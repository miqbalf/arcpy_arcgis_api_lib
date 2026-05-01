"""
Domain-specific joins, column lists, and helpers for the xls append / upload / delete flow.

Used by :class:`~arcpy_arcgis_api_lib.xls_append.main.ArcGISAPI`. Generic geometry
and feature-layer utilities live in :mod:`arcpy_arcgis_api_lib.xls_append.utils`.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import (
    DEFAULT_POINT_SPATIAL_REFERENCE,
    append_sdf_to_feature_layer,
    arcgis_geometry_to_shapely,
    delete_features_by_where,
    layer_field_map,
    merge_wrangled_columns,
    normalize_office_name,
    remove_suffix,
    shapely_to_arcgis_point_dict,
    spatial_reference_from_sdf,
    spatial_reference_to_dict,
    where_metric_year_filter,
)

__all__ = [
    "append_sdf_to_feature_layer",
    "arcgis_geometry_to_shapely",
    "columns_to_upload_for_joined_sdf",
    "delete_features_by_where",
    "layer_field_map",
    "matching_country_points",
    "matching_country_polygon",
    "merge_wrangled_columns",
    "normalize_office_name",
    "offices_from_andreas_gsheet",
    "point_layer_to_xy_country",
    "spatial_reference_from_sdf",
    "spatial_reference_to_dict",
    "where_metric_year_filter",
]


def offices_from_andreas_gsheet(andreas_data: Optional[pd.DataFrame]) -> Optional[List[str]]:
    if andreas_data is None or andreas_data.empty:
        return None
    source_column = None
    if "office_revision" in andreas_data.columns:
        source_column = "office_revision"
    elif "office_gsheet_andreas" in andreas_data.columns:
        source_column = "office_gsheet_andreas"
    elif "office_name_andreas" in andreas_data.columns:
        source_column = "office_name_andreas"
    if source_column is None:
        return None
    s = andreas_data[source_column].dropna().apply(normalize_office_name).dropna()
    if s.empty:
        return None
    s = s[s != "WWF Network Consolidated"]
    if s.empty:
        return None
    return sorted(s.unique().tolist())


def matching_country_polygon(
    input_df_revised: pd.DataFrame,
    regional_border: pd.DataFrame,
    world_df: pd.DataFrame,
    list_wwf_id_regional: List[str],
    list_wwf_id_non_regional: List[str],
    office_ref_df_no_regional: pd.DataFrame,
) -> pd.DataFrame:
    input_df_revised = input_df_revised.copy()
    regional_border = regional_border.copy()
    world_df = world_df.copy()

    input_df_revised_non_regional = input_df_revised.loc[
        input_df_revised["office"].isin(list_wwf_id_non_regional)
    ]
    input_df_revised_regional = input_df_revised.loc[
        input_df_revised["office"].isin(list_wwf_id_regional)
    ]

    if "office_name_andreas" in office_ref_df_no_regional.columns:
        office_key = "office_name_andreas"
    elif "office_andreas" in office_ref_df_no_regional.columns:
        office_key = "office_andreas"
    else:
        raise KeyError("office_name_andreas/office_andreas key not found in office reference data")

    df_data_country_joined_nonregional = pd.merge(
        input_df_revised_non_regional,
        office_ref_df_no_regional,
        left_on="office",
        right_on=office_key,
        how="left",
        suffixes=("_x", "_y"),
    )
    df_data_country_joined_regional = input_df_revised_regional.copy()
    df_data_country_joined_regional["country_andreas"] = df_data_country_joined_regional["region"]
    df_data_country_joined_regional["group_dataset"] = "regional_grouped"

    df_data_country_joined_nonregional = remove_suffix(df_data_country_joined_nonregional)
    df_data_country_joined_regional = remove_suffix(df_data_country_joined_regional)

    df_data_country_joined_nonregional["country"] = df_data_country_joined_nonregional["country_andreas"]
    df_data_country_joined_regional["country"] = df_data_country_joined_regional["country_andreas"]

    regional_border_use = regional_border[["office", "SHAPE"]].copy()
    world_df_use = world_df[["name", "SHAPE"]].copy()

    gdf_npo_regional = pd.merge(
        df_data_country_joined_regional, regional_border_use, on="office", how="left"
    )
    gdf_npo_non_regional = pd.merge(
        df_data_country_joined_nonregional,
        world_df_use,
        left_on="country_opensource_data_name",
        right_on="name",
        how="left",
    )

    all_npo = pd.concat([gdf_npo_regional, gdf_npo_non_regional], ignore_index=True)
    for c in ("name", "name_x", "name_y"):
        if c in all_npo.columns:
            all_npo = all_npo.drop(columns=c)

    sdf = all_npo.copy()
    sdf.spatial.set_geometry("SHAPE")
    return sdf


def point_layer_to_xy_country(point_layer_df: pd.DataFrame) -> pd.DataFrame:
    if point_layer_df is None or point_layer_df.empty:
        raise ValueError("Point reference layer is empty or unavailable")
    if "office" not in point_layer_df.columns:
        raise KeyError("Point reference layer must include an 'office' column")

    geom_col = "SHAPE"
    if hasattr(point_layer_df, "spatial") and getattr(point_layer_df.spatial, "name", None):
        geom_col = point_layer_df.spatial.name
    elif geom_col not in point_layer_df.columns and "geometry" in point_layer_df.columns:
        geom_col = "geometry"
    if geom_col not in point_layer_df.columns:
        raise KeyError("Point reference layer must include SHAPE/geometry")

    xy_country = point_layer_df[["office", geom_col]].copy()
    xy_country["office"] = xy_country["office"].apply(normalize_office_name)
    xy_country["geometry"] = xy_country[geom_col].apply(arcgis_geometry_to_shapely)
    xy_country = xy_country.drop(columns=[geom_col])
    xy_country = xy_country.dropna(subset=["office", "geometry"])
    return xy_country.drop_duplicates(subset=["office"], keep="first")


def matching_country_points(
    df_revised: pd.DataFrame,
    xy_country: pd.DataFrame,
    spatial_reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    df_data_country_joined = pd.merge(
        df_revised.copy(),
        xy_country.copy(),
        on="office",
        how="left",
    )

    sr = spatial_reference or DEFAULT_POINT_SPATIAL_REFERENCE.copy()
    if "geometry" in df_data_country_joined.columns:
        df_data_country_joined["SHAPE"] = df_data_country_joined["geometry"].apply(
            lambda geom: shapely_to_arcgis_point_dict(geom, sr)
        )
        df_data_country_joined = df_data_country_joined.drop(columns=["geometry"])
    else:
        df_data_country_joined["SHAPE"] = None

    sdf = df_data_country_joined.copy()
    if "SHAPE" in sdf.columns and sdf["SHAPE"].notna().any():
        sdf.spatial.set_geometry("SHAPE")
        sdf.spatial.sr = sr
    return {"df_data_country_joined": df_data_country_joined, "sdf": sdf}


def columns_to_upload_for_joined_sdf(df: pd.DataFrame) -> List[str]:
    names = [str(c) for c in df.columns]
    if "office_type" in df.columns and "metric_code" in df.columns:
        for extra in ("office_type_fe", "graph_fe"):
            if extra not in names:
                names.append(extra)
    return names
