import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
import re

from arcgis.gis import GIS
from arcgis.layers import Service
from dotenv import dotenv_values

import pandas as pd
import geopandas as gpd
# convert the feature layer arcgis data into geopandas object, it will takes some few minutes
from arcgis.features import GeoAccessor, GeoSeriesAccessor

from .data_val import (force_update_dtype_xls, map_df_to_esri,
                esri_to_df, compare_df_esri_types,
                pandas_to_esri,
                get_missing_columns, column_update_gap,
                validate_df_against_layer_with_details,
                multi_filter, graph_office_type, add_missing_fields_to_layer)
from .utils import validate_xls_path

# from arcgis.mapping import WebMap
from dotenv import load_dotenv
from .data_utils import (
    append_sdf_to_feature_layer,
    arcgis_geometry_to_shapely,
    columns_to_upload_for_joined_sdf,
    delete_features_by_where,
    layer_field_map,
    matching_country_points,
    matching_country_polygon,
    merge_wrangled_columns,
    normalize_office_name,
    offices_from_andreas_gsheet,
    point_layer_to_xy_country,
    spatial_reference_from_sdf,
    where_metric_year_filter,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# default var
api_key = os.getenv('API_KEY')
portal_url = os.getenv('PORTAL_URL')
query = os.getenv('QUERY')
item_type = os.getenv('ITEM_TYPE')


def _clean_env_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if v in ("", "''", '""'):
        return None
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1].strip()
    return v or None


def _pick_env_from_dotenv(names: Tuple[str, ...], file_env: Dict[str, Any]) -> Optional[str]:
    for name in names:
        val = _clean_env_str(os.getenv(name))
        if val:
            return val
    for name in names:
        val = _clean_env_str(file_env.get(name))
        if val:
            return val
    return None


class ArcGISAPI:
    """Optimized ArcGIS API wrapper class with improved error handling and performance."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        portal_url: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        query: Optional[str] = None,
        item_type: Optional[str] = None,
        xls_path: Optional[str] = None,
    ):
        """Initialize ArcGIS API connection with lazy loading for better performance."""
        file_env = dotenv_values(".env")
        self.portal_url = portal_url or _pick_env_from_dotenv(("ARCGIS_PORTAL_URL", "PORTAL_URL"), file_env) or "https://www.arcgis.com"
        self.api_key = api_key or _pick_env_from_dotenv(("ARCGIS_API_KEY", "API_KEY"), file_env)
        self.token = token
        self.username = username or _pick_env_from_dotenv(("ARCGIS_USERNAME", "USER_NAME"), file_env)
        self.password = password or _pick_env_from_dotenv(("ARCGIS_PASSWORD", "PASSWORD"), file_env)
        self.query = query or _pick_env_from_dotenv(("QUERY",), file_env) or ""
        self.item_type = item_type or _pick_env_from_dotenv(("ITEM_TYPE",), file_env) or "Web Map"
        
        # Initialize with None for lazy loading
        self._portal = None
        self._webmap_search_data = None
        self._webmap_data = None
        self._layer_groups = None
        self._list_layers = None
        self._layer_index = None
        
        # Feature layer tracking
        self._feature_layer_selected = None
        self._feature_layer_selected_item = None
        self.list_feature_layer = []  # Fixed: Initialize as empty list, not None
        
        if xls_path is None:
            self.xls_path = os.getenv('XLS_PATH')
        else:
            self.xls_path = xls_path     
        
        self._temp_file_path = None
        if self.xls_path:
            # Handle URL vs local file path using utils
            actual_xls_path, self._temp_file_path = validate_xls_path(self.xls_path)
            self.input_df = pd.read_excel(actual_xls_path)
            logger.info(
                f"Successfully loaded XLS with {len(self.input_df)} rows and {len(self.input_df.columns)} columns"
            )
        else:
            # Allow API-only usage where DataFrame input is supplied later (e.g. via API payload).
            self.input_df = pd.DataFrame()
            logger.info("ArcGISAPI initialized without XLS_PATH; waiting for runtime DataFrame input")

        logger.info("ArcGIS API initialized with lazy loading enabled")

    @property
    def portal(self) -> GIS:
        """Lazy-loaded portal connection."""
        if self._portal is None:
            self._portal = self._portal_login()
        return self._portal
    
    def _portal_login(self) -> GIS:
        """Create and return GIS portal connection with error handling."""
        try:
            if self.token:
                portal = GIS(self.portal_url, token=self.token)
            elif self.username and self.password:
                portal = GIS(self.portal_url, username=self.username, password=self.password)
            elif self.api_key:
                portal = GIS(self.portal_url, api_key=self.api_key)
            else:
                raise ValueError(
                    "Missing ArcGIS credentials. Set ARCGIS_API_KEY/API_KEY or "
                    "ARCGIS_USERNAME+ARCGIS_PASSWORD (or USER_NAME+PASSWORD), or pass a session token."
                )
            logger.info("Successfully connected to portal")
            return portal
        except Exception as e:
            logger.error(f"Failed to connect to portal: {e}")
            raise

    @property
    def webmap_search_data(self) -> List[Dict[str, Any]]:
        """Lazy-loaded webmap search results."""
        if self._webmap_search_data is None:
            self._webmap_search_data = self._webmap_search()
        return self._webmap_search_data
    
    def _webmap_search(self, query: Optional[str] = None, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for webmaps with error handling."""
        try:
            search_query = query or self.query
            search_item_type = item_type or self.item_type
            
            if not search_query or not search_item_type:
                raise ValueError("Query and item_type must be provided")
            
            webmap_search = self.portal.content.search(
                query=search_query,
                item_type=search_item_type
            )
            
            # logger.info(f"Found {len(webmap_search)} webmaps for query: {search_query}")
            logger.info(f"Webmap search successful with {len(webmap_search)} webmaps")
            return webmap_search
        except Exception as e:
            logger.error(f"Webmap search failed: {e}")
            raise

    @property
    def webmap_data(self) -> Dict[str, Any]:
        """Lazy-loaded webmap data."""
        if self._webmap_data is None:
            self._webmap_data = self._get_webmap_item_query()
        return self._webmap_data
    
    def _get_webmap_item_query(self, webmap_search_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Get webmap item data with improved error handling."""
        try:
            if webmap_search_data is None:
                webmap_search_data = self.webmap_search_data
            
            # Find matching webmaps
            matching_webmaps = [wm for wm in webmap_search_data if wm.get('title') == self.query]
            
            if not matching_webmaps:
                raise ValueError(f"No webmap found with title: {self.query}")
            
            if len(matching_webmaps) > 1:
                logger.warning(f"Multiple webmaps found with title '{self.query}', using the first one")
            
            webmap_item = matching_webmaps[0]
            webmap_data = webmap_item.get_data()
            
            # logger.info(f"Successfully loaded webmap data for: {self.query}")
            logger.info(f"Successfully loaded webmap data")
            return webmap_data
        except Exception as e:
            logger.error(f"Failed to get webmap item: {e}")
            raise

    @property
    def layer_groups(self) -> List[Dict[str, Any]]:
        """Lazy-loaded layer groups."""
        if self._layer_groups is None:
            self._layer_groups = self._get_layer_group()
        return self._layer_groups
    
    def _get_layer_group(self, webmap_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get layer groups from webmap data with error handling."""
        try:
            if webmap_data is None:
                webmap_data = self.webmap_data
            
            layer_groups = webmap_data.get('operationalLayers', [])
            
            if not layer_groups:
                logger.warning("No operational layers found in webmap data")
            else:
                logger.info(f"Found {len(layer_groups)} layer groups")
            
            return layer_groups
        except Exception as e:
            logger.error(f"Failed to get layer groups: {e}")
            raise

    @property
    def list_layers(self) -> List[Dict[str, Any]]:
        """Lazy-loaded list of all layers from webmap."""
        if self._list_layers is None:
            self._list_layers = self._layers_from_grouplayer_webmap()
        return self._list_layers
    
    def _layers_from_grouplayer_webmap(self, layer_groups: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract all layers from layer groups with error handling."""
        try:
            if layer_groups is None:
                layer_groups = self.layer_groups
            
            list_layers = []
            for layer_group in layer_groups:
                layers = layer_group.get('layers', [])
                for layer in layers:
                    list_layers.append(layer)
            
            logger.info(f"Extracted {len(list_layers)} layers from {len(layer_groups)} layer groups")
            return list_layers
        except Exception as e:
            logger.error(f"Failed to extract layers: {e}")
            raise

    def _build_layer_index(self) -> Dict[str, Dict[str, Any]]:
        """Build an index for faster layer lookups."""
        if self._layer_index is None:
            self._layer_index = {}
            if self.list_layers is not None:
                for layer in self.list_layers:
                    title = layer.get('title')
                    if title:
                        self._layer_index[title] = layer
            logger.info(f"Built layer index with {len(self._layer_index)} layers")
        return self._layer_index
    
    def get_feature_layer_tile_by_name(self, layer_name: str) -> Optional[Tuple[Any, Any]]:
        """Get feature layer by name from webmap data with caching."""
        try:
            # Use layer index for faster lookup
            layer_index = self._build_layer_index()
            
            if layer_name not in layer_index:
                raise ValueError(f"Layer '{layer_name}' not found in webmap")
            
            features = layer_index[layer_name]
            
            # Get the feature layer item
            feature_layer_item = self.portal.content.get(features.get('itemId'))
            
            if not feature_layer_item:
                raise ValueError(f"Could not retrieve feature layer item for {layer_name}")
            
            # Get the specific layer
            matching_layers = [layer for layer in feature_layer_item.layers 
                             if layer.url == features.get('url')]
            
            if not matching_layers:
                raise ValueError(f"Could not find matching layer URL for {layer_name}")
            
            feature_layer = matching_layers[0]
            
            # Update tracking
            self._feature_layer_selected = feature_layer
            self._feature_layer_selected_item = feature_layer_item
            
            # Add to list if not already present
            if feature_layer not in self.list_feature_layer:
                self.list_feature_layer.append(feature_layer)
            
            logger.info(f"Successfully retrieved feature layer: {layer_name}")
            return feature_layer_item, feature_layer

        except Exception as e:
            logger.error(f"Error getting feature layer for {layer_name}: {e}")
            return None

    def get_feature_tile_by_name(self, layer_name: str) -> Optional[Any]:
        """Get feature layer tile by name."""
        result = self.get_feature_layer_tile_by_name(layer_name)
        if result:
            return result[0]
        return None

    def layer_service_to_gdf(self, layer_name: str, feature_layer: Optional[Any] = None) -> Optional[gpd.GeoDataFrame]:
        """Convert layer service to GeoDataFrame with improved error handling."""
        try:
            if feature_layer is None:
                if (self._feature_layer_selected is None or 
                    self._feature_layer_selected.url != self.get_url_layer(layer_name)):
                    result = self.get_feature_layer_tile_by_name(layer_name)
                    if result:
                        feature_layer = result[1]
                    else:
                        raise ValueError(f"Could not retrieve feature layer for {layer_name}")
                else:
                    feature_layer = self._feature_layer_selected
            
            if not feature_layer:
                raise ValueError(f"Feature layer is None for {layer_name}")
            
            gdf = feature_layer.query().sdf
            logger.info(f"Successfully converted layer '{layer_name}' to GeoDataFrame with {len(gdf)} features")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to convert layer '{layer_name}' to GeoDataFrame: {e}")
            return None

    def get_url_layer(self, layer_name: str, list_layers: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Get URL for a specific layer by name."""
        try:
            if list_layers is None:
                list_layers = self.list_layers
            
            for layer in list_layers:
                if layer.get('title') == layer_name:
                    return layer.get('url')
            
            logger.warning(f"Layer '{layer_name}' not found in layer list")
            return None
        except Exception as e:
            logger.error(f"Error getting URL for layer '{layer_name}': {e}")
            return None

    def describe_layer_fields(self, layer_name: str, feature_layer: Optional[Any] = None) -> Optional[List[Dict[str, Any]]]:
        """Print and return a list of all fields (name, type, alias, length) from an ArcGIS FeatureLayer."""
        try:
            if feature_layer is None:
                if (self._feature_layer_selected is None or 
                    self._feature_layer_selected.url != self.get_url_layer(layer_name)):
                    result = self.get_feature_layer_tile_by_name(layer_name)
                    if result:
                        feature_layer = result[1]
                    else:
                        raise ValueError(f"Could not retrieve feature layer for {layer_name}")
                else:
                    feature_layer = self._feature_layer_selected
            
            if not feature_layer:
                raise ValueError(f"Feature layer is None for {layer_name}")
            
            fields = feature_layer.properties.fields
            info = []
            
            logger.info(f"Field info for: {feature_layer.properties.name}")
            logger.info("-" * 60)
            
            for f in fields:
                name = f.get('name')
                ftype = f.get('type')
                alias = f.get('alias', '')
                length = f.get('length', '')
                
                field_info = f"{name:30} | {ftype:20} | alias='{alias}' | length={length}"
                logger.info(field_info)
                
                info.append({
                    'name': name,
                    'type': ftype,
                    'alias': alias,
                    'length': length
                })
            
            logger.info(f"Found {len(info)} fields for layer '{layer_name}'")
            return info
            
        except Exception as e:
            logger.error(f"Failed to describe fields for layer '{layer_name}': {e}")
            return None


    def xls_wrangling(
        self,
        layer_name: Optional[str] = None,
        df_reference: Optional[pd.DataFrame] = None,
        reference_layer=None,
        xls_path: Optional[str] = None,
        query_filter: Optional[Dict[str, List]] = None,
        auto_add_fields: bool = True,
        df_input: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load and process XLS data with error handling.
        
        Args:
            layer_name: Name of the layer to process
            df_reference: Reference DataFrame for type matching
            reference_layer: Reference feature layer for validation
            xls_path: Optional path to XLS file (overrides instance path)
            query_filter: Optional filter conditions
            auto_add_fields: If True, automatically add missing fields to layer when validation fails. Default: True
            df_input: Optional input DataFrame (alternative to reading from xls_path)
        
        Returns:
            Dictionary with processing results or None if failed
        """
        try:
            if df_input is not None:
                self.input_df = df_input
            elif xls_path is not None:
                self.input_df = pd.read_excel(xls_path)
            logger.info(
                f"Successfully loaded XLS with {len(self.input_df)} rows and {len(self.input_df.columns)} columns"
            )

            if query_filter is not None:
                self.input_df = multi_filter(self.input_df, query_filter)

            if "office_type" in self.input_df.columns:
                self.input_df["office_type_fe"] = self.input_df["office_type"].apply(
                    lambda v: "Country Office" if v == "Country Office" else "National Office / Partners"
                )

            if "office_type_fe" not in self.input_df.columns or "metric_code" not in self.input_df.columns:
                logger.warning("office_type_fe or metric_code not found in the input dataframe")
                return None

            self.input_df["graph_fe"] = self.input_df.apply(
                lambda x: graph_office_type(x["metric_code"], x["office_type_fe"], is_polygon=True),
                axis=1,
            )

            if self.input_df.empty:
                logger.warning("XLS file is empty")

            esri_schema_api = self.describe_layer_fields(layer_name)
            if esri_schema_api is None:
                raise ValueError(f"Could not retrieve schema for layer: {layer_name}")
            esri_schema = {i["name"]: i["type"] for i in esri_schema_api}

            mapped_df_esri = map_df_to_esri(self.input_df, esri_schema)
            mapped_esri_df = esri_to_df(self.input_df, esri_schema)

            if reference_layer is None and layer_name is not None:
                result = self.get_feature_layer_tile_by_name(layer_name)
                if not result:
                    raise ValueError(f"Could not retrieve feature layer for: {layer_name}")
                reference_layer = result[1]

            col_not_exist_not_system = get_missing_columns(self.input_df, reference_layer)
            if col_not_exist_not_system and col_not_exist_not_system.get("missing_in_df"):
                new_updated_df_revised = column_update_gap(
                    self.input_df, col_not_exist_not_system["missing_in_df"]
                )
            else:
                new_updated_df_revised = self.input_df

            comparison_old = compare_df_esri_types(new_updated_df_revised, esri_schema, pandas_to_esri)

            updated_df = force_update_dtype_xls(
                new_updated_df_revised,
                df_reference,
                feature_layer=reference_layer,
            )
            if updated_df is None or updated_df.empty:
                logger.warning("force_update_dtype_xls returned None or empty data. Using original data.")
                logger.warning("This might indicate a mismatch between XLS data and target layer schema.")
                logger.warning("Please check if metric_code and other fields match the target layer requirements.")
                self.debug_data_mismatch(new_updated_df_revised, df_reference, layer_name)
            else:
                new_updated_df_revised = updated_df
                logger.info(f"Successfully updated data types. New shape: {new_updated_df_revised.shape}")

            comparison_new = compare_df_esri_types(new_updated_df_revised, esri_schema, pandas_to_esri)

            add_fields_result = None
            is_valid = None
            validation_report: Dict[str, Any] = {}

            # Joined polygon rows include many source columns not hosted on the layer.
            # strict=False validates overlapping layer fields only; extra join columns are not a failure.
            _validate_strict = False

            if reference_layer is None:
                print("No reference layer available for validation")
                validation_report = {"reason": "no_reference_layer"}
            elif new_updated_df_revised is None or new_updated_df_revised.empty:
                print("No valid data available for validation")
                validation_report = {"reason": "empty_dataframe"}
            else:
                col_diff = get_missing_columns(new_updated_df_revised, reference_layer)
                validation_report = {
                    "schema_allows_extra_dataframe_columns": not _validate_strict,
                    "columns_extra_in_dataframe": sorted(col_diff.get("extra_in_df") or []),
                    "columns_missing_in_dataframe": sorted(col_diff.get("missing_in_df") or []),
                }

                details = validate_df_against_layer_with_details(
                    new_updated_df_revised, reference_layer, strict=_validate_strict
                )

                if details["ok"]:
                    is_valid = True
                    validation_report["pandera_ok"] = True
                    if validation_report["columns_extra_in_dataframe"]:
                        validation_report["note"] = (
                            "Extra join columns are not fields on the hosted layer; "
                            "they are ignored for validation (schema strict=False)."
                        )
                    print("Valid!")
                else:
                    error_msg = details.get("error") or ""
                    is_valid = False
                    validation_report["pandera_ok"] = False
                    validation_report["pandera_error"] = details.get("pandera_error") or {
                        "message": error_msg
                    }

                    missing_cols_error = (
                        "not in DataFrameSchema" in error_msg
                        or "not in dataframe" in error_msg.lower()
                    )
                    if missing_cols_error:
                        print("\nValidation failed (schema); checking for fields to add on service...")
                        dry_run_result = add_missing_fields_to_layer(
                            new_updated_df_revised, reference_layer, dry_run=True
                        )

                        if dry_run_result.get("success") and dry_run_result.get("fields_to_add"):
                            for field in dry_run_result["fields_to_add"]:
                                print(
                                    f"   - {field['name']} ({field['type']}, length={field.get('length', 'N/A')})"
                                )
                            add_fields_result = dry_run_result

                            if auto_add_fields:
                                add_result = add_missing_fields_to_layer(
                                    new_updated_df_revised, reference_layer, dry_run=False
                                )
                                if add_result.get("success"):
                                    add_fields_result = add_result
                                    if hasattr(reference_layer, "properties"):
                                        _ = reference_layer.properties
                                    details2 = validate_df_against_layer_with_details(
                                        new_updated_df_revised, reference_layer, strict=_validate_strict
                                    )
                                    validation_report["after_auto_add_fields"] = details2
                                    if details2["ok"]:
                                        is_valid = True
                                        validation_report["pandera_ok"] = True
                                        validation_report["pandera_error"] = None
                                    else:
                                        is_valid = False
                                        validation_report["pandera_ok"] = False
                                        validation_report["pandera_error"] = details2.get(
                                            "pandera_error"
                                        ) or {"message": details2.get("error")}
                                else:
                                    print(f"Failed to add fields: {add_result.get('error')}")
                            else:
                                print("auto_add_fields=False; add fields manually if needed.")
                        else:
                            print("No service fields to add from dry run.")

            return {
                "mapped_df_esri": mapped_df_esri,
                "mapped_esri_df": mapped_esri_df,
                "comparison_old": comparison_old,
                "comparison_new": comparison_new,
                "is_valid": is_valid,
                "new_updated_df_revised": new_updated_df_revised,
                "data_processing_successful": new_updated_df_revised is not None and not new_updated_df_revised.empty,
                "add_fields_result": add_fields_result,
                "validation_report": validation_report,
            }
        except Exception as e:
            logger.error(f"Failed to process XLS file: {e}")
            return None
    
    def debug_data_mismatch(self, csv_df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug method to help identify data mismatches between XLS and target layer.
        
        Args:
            csv_df: The XLS DataFrame
            reference_df: Optional reference DataFrame from the layer
            layer_name: Optional layer name for context
            
        Returns:
            Dict with debugging information
        """
        debug_info = {
            'csv_shape': csv_df.shape,
            'csv_columns': list(csv_df.columns),
            'csv_dtypes': csv_df.dtypes.to_dict(),
            'csv_sample_data': csv_df.head(3).to_dict() if not csv_df.empty else None
        }
        
        if reference_df is not None:
            debug_info.update({
                'reference_shape': reference_df.shape,
                'reference_columns': list(reference_df.columns),
                'reference_dtypes': reference_df.dtypes.to_dict(),
                'common_columns': list(set(csv_df.columns) & set(reference_df.columns)),
                'csv_only_columns': list(set(csv_df.columns) - set(reference_df.columns)),
                'reference_only_columns': list(set(reference_df.columns) - set(csv_df.columns))
            })
        
        if layer_name:
            debug_info['layer_name'] = layer_name
            
        logger.info("=== DATA MISMATCH DEBUG INFO ===")
        for key, value in debug_info.items():
            logger.info(f"{key}: {value}")
        logger.info("=== END DEBUG INFO ===")
        
        return debug_info

    def list_webmaps(
        self,
        query: Optional[str] = None,
        item_type: Optional[str] = None,
        max_items: int = 20,
    ) -> List[Dict[str, Any]]:
        search_items = self._webmap_search(query=query or self.query, item_type=item_type or self.item_type)
        return [
            {"id": item.id, "title": item.title, "owner": item.owner, "type": item.type, "url": item.url}
            for item in search_items[:max_items]
        ]

    def list_webmap_layers(self) -> List[Dict[str, Any]]:
        return self.list_layers or []

    @staticmethod
    def _normalize_layer_name_for_backup(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    @staticmethod
    def _safe_backup_layer_filename(layer_name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", layer_name).strip("._")
        return cleaned or "layer"

    def backup_layers(
        self,
        default_layers: List[str],
        layer_names: Optional[List[str]] = None,
        backup_dir: str = "backup_data",
    ) -> Dict[str, Any]:
        available_layers = self.list_webmap_layers()
        available_titles = [layer.get("title") for layer in available_layers if layer.get("title")]
        normalized_map = {self._normalize_layer_name_for_backup(title): title for title in available_titles}
        selected_layers = layer_names or default_layers
        os.makedirs(backup_dir, exist_ok=True)
        result: Dict[str, Any] = {
            "backup_dir": os.path.abspath(backup_dir),
            "requested_layers": selected_layers,
            "available_layers": available_titles,
            "saved_files": [],
            "errors": [],
        }
        for requested_layer_name in selected_layers:
            try:
                layer_name = requested_layer_name
                if layer_name not in available_titles:
                    normalized = self._normalize_layer_name_for_backup(layer_name)
                    if normalized in normalized_map:
                        layer_name = normalized_map[normalized]
                    else:
                        result["errors"].append({"layer": requested_layer_name, "error": "Layer title not found in webmap"})
                        continue
                sdf = self.layer_service_to_gdf(layer_name=layer_name)
                if sdf is None or sdf.empty:
                    result["errors"].append(
                        {
                            "layer": requested_layer_name,
                            "resolved_layer": layer_name,
                            "error": "Layer dataframe is empty or unavailable",
                        }
                    )
                    continue
                df = sdf.copy()
                geom_col = sdf.spatial.name if hasattr(sdf, "spatial") else "SHAPE"
                if geom_col not in df.columns:
                    result["errors"].append(
                        {
                            "layer": requested_layer_name,
                            "resolved_layer": layer_name,
                            "error": f"No geometry column found: {geom_col}",
                        }
                    )
                    continue
                df["geometry"] = df[geom_col].apply(arcgis_geometry_to_shapely)
                df = df[df["geometry"].notna()].copy()
                if df.empty:
                    result["errors"].append(
                        {
                            "layer": requested_layer_name,
                            "resolved_layer": layer_name,
                            "error": "No valid geometries after conversion",
                        }
                    )
                    continue
                crs = "EPSG:4326"
                try:
                    wkid = sdf.spatial.sr.wkid if hasattr(sdf.spatial, "sr") else 4326
                    crs = f"EPSG:{wkid}"
                except Exception:
                    pass
                gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry="geometry", crs=crs)
                out_path = os.path.join(backup_dir, f"{self._safe_backup_layer_filename(layer_name)}.gpkg")
                gdf.to_file(out_path, driver="GPKG")
                result["saved_files"].append(
                    {
                        "layer": requested_layer_name,
                        "resolved_layer": layer_name,
                        "path": os.path.abspath(out_path),
                        "rows": int(len(gdf)),
                    }
                )
            except Exception as exc:
                result["errors"].append({"layer": requested_layer_name, "error": str(exc)})
        result["success_count"] = len(result["saved_files"])
        result["error_count"] = len(result["errors"])
        return result

    def _wrangle_and_sync_joined_metrics(
        self,
        metrics_sdf: Dict[str, pd.DataFrame],
        target_layers: List[str],
        list_year: Optional[List[int]],
        update_arcgis: bool,
        append_batch_size: int,
        delete_chunk_size: int,
    ) -> Dict[str, Any]:
        layer_name_flayer = {}
        wrangling_results = []
        if not update_arcgis:
            return {"layer_index_size": 0, "wrangling_results": wrangling_results}
        metric_to_layer = {}
        for metric_code in metrics_sdf.keys():
            matching_layers = [lyr_name for lyr_name in target_layers if metric_code in lyr_name]
            if matching_layers:
                metric_to_layer[metric_code] = matching_layers[0]
        for metrics_code, df_sdf in metrics_sdf.items():
            layer_name = metric_to_layer.get(metrics_code)
            if not layer_name:
                wrangling_results.append(
                    {
                        "metric_code": metrics_code,
                        "success": False,
                        "error": "No target ArcGIS layer mapping found for metric",
                        "arcgis_sync": None,
                    }
                )
                continue
            layer_info = self.get_feature_layer_tile_by_name(layer_name)
            if not layer_info:
                wrangling_results.append(
                    {
                        "metric_code": metrics_code,
                        "layer_name": layer_name,
                        "success": False,
                        "error": "Could not resolve ArcGIS feature layer",
                        "arcgis_sync": None,
                    }
                )
                continue
            reference_layer = layer_info[1]
            layer_name_flayer[layer_name] = reference_layer
            df_reference = self.layer_service_to_gdf(layer_name=layer_name)
            wrangling_sdf = self.xls_wrangling(
                layer_name=layer_name,
                reference_layer=reference_layer,
                df_reference=df_reference,
                auto_add_fields=True,
                df_input=df_sdf,
            )
            if not wrangling_sdf:
                wrangling_results.append(
                    {
                        "metric_code": metrics_code,
                        "layer_name": layer_name,
                        "success": False,
                        "error": "xls_wrangling returned None",
                        "arcgis_sync": None,
                    }
                )
                continue
            revised_df = wrangling_sdf.get("new_updated_df_revised")
            entry: Dict[str, Any] = {
                "metric_code": metrics_code,
                "layer_name": layer_name,
                "success": bool(wrangling_sdf.get("data_processing_successful")),
                "is_valid": wrangling_sdf.get("is_valid"),
                "output_rows": int(len(revised_df)) if revised_df is not None else 0,
                "add_fields_result": wrangling_sdf.get("add_fields_result"),
                "validation_report": wrangling_sdf.get("validation_report"),
                "arcgis_sync": None,
            }
            can_sync = revised_df is not None and not revised_df.empty and wrangling_sdf.get("data_processing_successful") and wrangling_sdf.get("is_valid") is True
            if can_sync:
                fm = layer_field_map(reference_layer)
                field_metric = fm.get("metric_code")
                field_year = fm.get("year")
                if not field_metric:
                    entry["arcgis_sync"] = {"success": False, "error": "Layer has no metric_code field; cannot scope delete"}
                elif list_year and not field_year:
                    entry["arcgis_sync"] = {"success": False, "error": "years_filter is set but layer has no year field"}
                else:
                    where = where_metric_year_filter(
                        metrics_code, list_year, field_metric, field_year if list_year else None
                    )
                    dels = delete_features_by_where(reference_layer, where, delete_chunk_size=delete_chunk_size)
                    if not dels.get("success"):
                        entry["arcgis_sync"] = {
                            "success": False,
                            "where": where,
                            "delete": dels,
                            "append": {"skipped": True, "reason": "delete had failures"},
                        }
                    else:
                        sdf_append = merge_wrangled_columns(df_sdf, revised_df)
                        app = append_sdf_to_feature_layer(reference_layer, sdf_append, batch_size=append_batch_size)
                        entry["arcgis_sync"] = {
                            "success": bool(dels.get("success")) and bool(app.get("success")),
                            "where": where,
                            "delete": dels,
                            "append": app,
                        }
            elif wrangling_sdf.get("data_processing_successful"):
                vr = wrangling_sdf.get("validation_report") or {}
                entry["arcgis_sync"] = {
                    "skipped": True,
                    "reason": (
                        "No reference layer inside wrangling; validation skipped (is_valid is null)."
                        if vr.get("reason") == "no_reference_layer"
                        else "Layer validation did not pass (is_valid is not True). See validation_report for columns_extra_in_dataframe, columns_missing_in_dataframe, and pandera_error."
                    ),
                    "validation_report": vr,
                }
            sync = entry.get("arcgis_sync")
            if isinstance(sync, dict) and sync.get("success") is False and not sync.get("skipped"):
                entry["success"] = False
            wrangling_results.append(entry)
        return {"layer_index_size": int(len(layer_name_flayer)), "wrangling_results": wrangling_results}

    def data_join_polygon(
        self,
        npo_data: Optional[pd.DataFrame],
        andreas_data: Optional[pd.DataFrame],
        default_metric_list: List[str],
        default_polygon_target_layers: List[str],
        list_year: Optional[List[int]] = None,
        list_metric: Optional[List[str]] = None,
        update_arcgis: bool = False,
        append_batch_size: int = 400,
        delete_chunk_size: int = 2000,
    ) -> Dict[str, Any]:
        if npo_data is None:
            raise ValueError("npo_data is required; pass a DataFrame built from your JSON payload")
        if andreas_data is None:
            raise ValueError("andreas_data is required; pass a DataFrame built from your JSON payload")
        list_metric = list_metric or list(default_metric_list)
        if not list_metric:
            raise ValueError("list_metric is required when no default_metric_list is configured")
        country_gdf = self.layer_service_to_gdf(layer_name="Other countries")
        regional_border = self.layer_service_to_gdf(layer_name="WWF Country Regional")
        df_input = npo_data.copy()
        office_ref_df = andreas_data.copy()
        office_ref_df["office_name_andreas"] = office_ref_df["office_revision"]
        office_ref_df["office_type_andreas"] = office_ref_df["office_type"]
        office_ref_df["country_andreas"] = office_ref_df["country_wwf_name"]
        office_ref_df["region_andreas"] = office_ref_df["region"]
        office_ref_df = office_ref_df[
            [
                "office_name_andreas",
                "office_type_andreas",
                "country_andreas",
                "country_opensource_data_name",
                "region_andreas",
                "group_dataset",
            ]
        ].copy()
        office_ref_df_no_regional = office_ref_df[office_ref_df["group_dataset"] == "non_regional"].copy()
        office_ref_df_regional = office_ref_df[office_ref_df["group_dataset"] == "regional_grouped"].copy()
        list_wwf_id_regional = sorted(office_ref_df_regional["office_name_andreas"].unique())
        list_wwf_id_non_regional = sorted(office_ref_df_no_regional["office_name_andreas"].unique())
        if list_year:
            df_input = df_input[df_input["year"].isin(list_year)]
        if list_metric:
            df_input = df_input[df_input["metric_code"].isin(list_metric)]
        metric_rows: Dict[str, int] = {}
        metrics_sdf: Dict[str, pd.DataFrame] = {}
        for metric_code in list_metric:
            input_df_revised = df_input[df_input["metric_code"] == metric_code]
            sdf = matching_country_polygon(
                input_df_revised,
                regional_border,
                country_gdf,
                list_wwf_id_regional,
                list_wwf_id_non_regional,
                office_ref_df_no_regional,
            )
            metric_rows[metric_code] = int(len(sdf))
            metrics_sdf[metric_code] = sdf
        sync_result = self._wrangle_and_sync_joined_metrics(
            metrics_sdf=metrics_sdf,
            target_layers=default_polygon_target_layers,
            list_year=list_year,
            update_arcgis=update_arcgis,
            append_batch_size=append_batch_size,
            delete_chunk_size=delete_chunk_size,
        )
        return {
            "update_arcgis": update_arcgis,
            "years_filter": list_year,
            "metrics_filter": list_metric,
            "source_rows_npo": int(len(df_input)),
            "source_rows_andreas": int(len(office_ref_df)),
            "joined_batches": int(len(metrics_sdf.keys())),
            "joined_rows_by_metric": metric_rows,
            "columns_to_upload": {
                metric_code: columns_to_upload_for_joined_sdf(sdf) for metric_code, sdf in metrics_sdf.items()
            },
            "layer_index_size": sync_result["layer_index_size"],
            "wrangling_results": sync_result["wrangling_results"],
        }

    def data_join_point(
        self,
        npo_data: Optional[pd.DataFrame],
        andreas_data: Optional[pd.DataFrame],
        default_metric_list: List[str],
        default_point_target_layers: List[str],
        default_point_offices: List[str],
        default_point_reference_layer: str,
        list_year: Optional[List[int]] = None,
        list_metric: Optional[List[str]] = None,
        list_office_required: Optional[List[str]] = None,
        point_reference_layer_name: Optional[str] = None,
        update_arcgis: bool = False,
        append_batch_size: int = 400,
        delete_chunk_size: int = 2000,
    ) -> Dict[str, Any]:
        if npo_data is None:
            raise ValueError("npo_data is required; pass a DataFrame built from your JSON payload")
        list_metric = list_metric or list(default_metric_list)
        if not list_metric:
            raise ValueError("list_metric is required when no default_metric_list is configured")
        if list_office_required is None:
            derived_offices = offices_from_andreas_gsheet(andreas_data)
            if derived_offices is not None:
                list_office_required = derived_offices
                list_office_source = "andreas_data.office_revision"
            else:
                list_office_required = list(default_point_offices)
                list_office_source = "default"
        else:
            list_office_source = "request"
        point_reference_layer_name = point_reference_layer_name or default_point_reference_layer
        point_layer_df = self.layer_service_to_gdf(layer_name=point_reference_layer_name)
        point_spatial_reference = spatial_reference_from_sdf(point_layer_df)
        xy_country = point_layer_to_xy_country(point_layer_df)
        df_input = npo_data.copy()
        if "office" in df_input.columns:
            df_input["office"] = df_input["office"].apply(normalize_office_name)
        if list_year:
            df_input = df_input[df_input["year"].isin(list_year)]
        if list_metric:
            df_input = df_input[df_input["metric_code"].isin(list_metric)]
        if list_office_required:
            df_input = df_input[df_input["office"].isin(list_office_required)]
        metric_rows: Dict[str, int] = {}
        metrics_sdf: Dict[str, pd.DataFrame] = {}
        missing_point_offices_by_metric: Dict[str, List[str]] = {}
        for metric_code in list_metric:
            input_df_revised = df_input[df_input["metric_code"] == metric_code]
            joining = matching_country_points(input_df_revised, xy_country, spatial_reference=point_spatial_reference)
            joined_df = joining["df_data_country_joined"]
            sdf = joining["sdf"]
            metric_rows[metric_code] = int(len(sdf))
            metrics_sdf[metric_code] = sdf
            missing_offices = []
            if "SHAPE" in joined_df.columns and "office" in joined_df.columns:
                missing_offices = sorted(
                    joined_df.loc[joined_df["SHAPE"].isna(), "office"].dropna().astype(str).unique().tolist()
                )
            missing_point_offices_by_metric[metric_code] = missing_offices
        sync_result = self._wrangle_and_sync_joined_metrics(
            metrics_sdf=metrics_sdf,
            target_layers=default_point_target_layers,
            list_year=list_year,
            update_arcgis=update_arcgis,
            append_batch_size=append_batch_size,
            delete_chunk_size=delete_chunk_size,
        )
        return {
            "update_arcgis": update_arcgis,
            "years_filter": list_year,
            "metrics_filter": list_metric,
            "list_office_required": list(list_office_required),
            "list_office_source": list_office_source,
            "point_reference_layer": point_reference_layer_name,
            "point_spatial_reference": point_spatial_reference,
            "target_layers": default_point_target_layers,
            "source_rows_npo": int(len(df_input)),
            "source_rows_andreas": int(len(andreas_data)) if andreas_data is not None else 0,
            "point_reference_rows": int(len(xy_country)),
            "joined_batches": int(len(metrics_sdf.keys())),
            "joined_rows_by_metric": metric_rows,
            "missing_point_offices_by_metric": missing_point_offices_by_metric,
            "columns_to_upload": {
                metric_code: columns_to_upload_for_joined_sdf(sdf) for metric_code, sdf in metrics_sdf.items()
            },
            "layer_index_size": sync_result["layer_index_size"],
            "wrangling_results": sync_result["wrangling_results"],
        }

    def cleanup(self):
        """Clean up resources and clear caches."""
        try:
            # Reset lazy-loaded properties
            self._webmap_search_data = None
            self._webmap_data = None
            self._layer_groups = None
            self._list_layers = None
            self._layer_index = None
            
            # Clear feature layer tracking
            self._feature_layer_selected = None
            self._feature_layer_selected_item = None
            self.list_feature_layer.clear()
            
            logger.info("Successfully cleaned up ArcGIS API resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup

