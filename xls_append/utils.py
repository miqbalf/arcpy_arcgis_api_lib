"""
Utility functions for the ArcGIS API library.
"""

import json
import logging
import math
import os
import tempfile
import requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import shapely.geometry.base
from arcgis.features import GeoAccessor
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon, shape

# Configure logging
logger = logging.getLogger(__name__)

DEFAULT_POINT_SPATIAL_REFERENCE = {"wkid": 102100, "latestWkid": 3857}


def remove_suffix(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in df.columns if col.endswith("_x")]
    df = df.drop(columns=cols_to_drop)
    df.columns = [col.replace("_y", "") if col.endswith("_y") else col for col in df.columns]
    return df


def normalize_office_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        cleaned = str(value).strip()
    return cleaned or None


def spatial_reference_to_dict(sr: Any) -> Dict[str, Any]:
    if sr is None:
        return DEFAULT_POINT_SPATIAL_REFERENCE.copy()
    if isinstance(sr, dict):
        out = dict(sr)
    elif hasattr(sr, "to_dict"):
        out = sr.to_dict()
    else:
        out = {}
        for key in ("wkid", "latestWkid", "wkt"):
            value = getattr(sr, key, None)
            if value:
                out[key] = value
    return out or DEFAULT_POINT_SPATIAL_REFERENCE.copy()


def spatial_reference_from_sdf(sdf: pd.DataFrame) -> Dict[str, Any]:
    try:
        if hasattr(sdf, "spatial") and getattr(sdf.spatial, "sr", None):
            return spatial_reference_to_dict(sdf.spatial.sr)
    except Exception:
        pass
    return DEFAULT_POINT_SPATIAL_REFERENCE.copy()


def arcgis_geometry_to_shapely(value: Any) -> Any:
    if value is None or isinstance(value, shapely.geometry.base.BaseGeometry):
        return value
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return Point(value["x"], value["y"])
        if "points" in value:
            return MultiPoint(value["points"])
        if "paths" in value:
            paths = value["paths"]
            if len(paths) == 1:
                return LineString(paths[0])
            return MultiLineString(paths)
        if "rings" in value:
            rings = value["rings"]
            if not rings:
                return None
            return Polygon(rings[0], rings[1:])
        try:
            return shape(value)
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return arcgis_geometry_to_shapely(json.loads(raw))
        except Exception:
            return None
    return None


def shapely_to_arcgis_point_dict(
    geom: Any,
    spatial_reference: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if geom is None:
        return None
    if not isinstance(geom, shapely.geometry.base.BaseGeometry):
        geom = arcgis_geometry_to_shapely(geom)
    if geom is None or geom.is_empty:
        return None

    sr = spatial_reference or DEFAULT_POINT_SPATIAL_REFERENCE
    if isinstance(geom, Point):
        point = geom
    elif isinstance(geom, MultiPoint) and len(geom.geoms) > 0:
        point = geom.geoms[0]
    else:
        return None

    return {"x": point.x, "y": point.y, "spatialReference": dict(sr)}


def layer_field_map(feature_layer) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in feature_layer.properties.fields or []:
        n = f.get("name")
        if n:
            out[str(n).lower()] = n
    return out


def sql_quote_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def where_metric_year_filter(
    metric_code: str,
    list_year: Optional[List[int]],
    field_metric: str,
    field_year: Optional[str],
) -> str:
    parts = [f"{field_metric} = {sql_quote_str(metric_code)}"]
    if list_year and field_year:
        ys = ",".join(str(int(y)) for y in sorted(set(list_year)))
        parts.append(f"{field_year} IN ({ys})")
    return " AND ".join(parts)


def delete_features_by_where(
    feature_layer,
    where: str,
    delete_chunk_size: int = 2000,
) -> Dict[str, Any]:
    oid_field = feature_layer.properties.objectIdField
    all_oids: List[Any] = []
    offset = 0
    page = 2000
    while True:
        fs = feature_layer.query(
            where=where,
            out_fields=[oid_field],
            return_geometry=False,
            result_offset=offset,
            result_record_count=page,
        )
        batch = fs.features or []
        if not batch:
            break
        for f in batch:
            all_oids.append(f.attributes[oid_field])
        if len(batch) < page:
            break
        offset += page

    if not all_oids:
        return {"success": True, "deleted": 0, "failed": 0, "attempted": 0, "where": where}

    total_deleted = 0
    failed = 0
    batches = 0
    for i in range(0, len(all_oids), delete_chunk_size):
        chunk = all_oids[i : i + delete_chunk_size]
        dr = feature_layer.edit_features(deletes=chunk)
        batches += 1
        for r in dr.get("deleteResults", []) or []:
            if r.get("success"):
                total_deleted += 1
            else:
                failed += 1

    return {
        "success": failed == 0,
        "deleted": total_deleted,
        "failed": failed,
        "attempted": len(all_oids),
        "where": where,
        "delete_batches": batches,
    }


def merge_wrangled_columns(join_sdf: pd.DataFrame, revised_df: pd.DataFrame) -> pd.DataFrame:
    if len(revised_df) != len(join_sdf):
        crs = "EPSG:3857"
        if hasattr(join_sdf, "spatial") and getattr(join_sdf.spatial, "sr", None):
            crs = join_sdf.spatial.sr
        if "SHAPE" not in revised_df.columns:
            raise ValueError("Wrangled row count differs from join and revised_df has no SHAPE for rebuild")
        gdf = gpd.GeoDataFrame(revised_df.copy(), geometry="SHAPE", crs=crs)
        return GeoAccessor.from_geodataframe(gdf)

    out = join_sdf.copy()
    for col in revised_df.columns:
        out[col] = revised_df[col].values
    if hasattr(out, "spatial"):
        out.spatial.set_geometry("SHAPE")
    return out


def append_sdf_to_feature_layer(
    feature_layer,
    sdf: pd.DataFrame,
    batch_size: int = 400,
) -> Dict[str, Any]:
    if sdf is None or sdf.empty:
        return {"success": False, "error": "empty dataframe", "added": 0, "failed": 0}
    if not hasattr(sdf, "spatial"):
        if "SHAPE" not in sdf.columns:
            return {"success": False, "error": "missing SHAPE", "added": 0, "failed": 0}
        sdf = sdf.copy()
        sdf.spatial.set_geometry("SHAPE")
    elif getattr(sdf.spatial, "name", None) != "SHAPE" and "SHAPE" in sdf.columns:
        sdf = sdf.copy()
        sdf.spatial.set_geometry("SHAPE")
    if sdf["SHAPE"].isnull().any():
        return {"success": False, "error": "null SHAPE geometry", "added": 0, "failed": 0}

    feats = sdf.spatial.to_featureset().features
    total = len(feats)
    if total == 0:
        return {"success": True, "added": 0, "failed": 0, "batches": 0}
    n_batches = math.ceil(total / batch_size)
    added = 0
    failed = 0
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch = feats[start:end]
        try:
            res = feature_layer.edit_features(adds=batch)
            for r in res.get("addResults", []) or []:
                if r.get("success"):
                    added += 1
                else:
                    failed += 1
        except Exception as exc:
            logger.exception("append batch failed")
            failed += len(batch)
            return {
                "success": False,
                "error": str(exc),
                "added": added,
                "failed": failed,
                "batches_done": i,
            }
    return {"success": failed == 0, "added": added, "failed": failed, "batches": n_batches}


######### BACKING UP DATA to GPKG - some data geometry is missing though :( #########
def backup_data_to_gpkg(list_gdf_to_update, list_layer_to_update, output_dir: str = 'backup_data'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Backing up data to {os.path.join(os.getcwd(), output_dir)}")
    for i in range(len(list_gdf_to_update)):
        layer_name = list_layer_to_update[i].replace(' ', '_').replace('-', '_').replace('(', '_').replace(')', '_')
        gdf = list_gdf_to_update[i].copy()
        gdf.set_geometry('geometry', inplace=True)
        gdf = gdf.drop(columns=["SHAPE"])
        # Ensure CRS is set properly before saving
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:3857') # hardcoded for now
            print(f"  -> Set CRS to EPSG:3857 for layer '{layer_name}'")

        geom_type = None
        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                if isinstance(geom, Point):
                    geom_type = 'point'
                elif isinstance(geom, Polygon):
                    geom_type = 'polygon'
                elif isinstance(geom, LineString):
                    geom_type = 'line'
                else:
                    # For MultiPoint, MultiPolygon, MultiLineString, etc.
                    geom_type = geom.geom_type.lower()
                break

        layer_path = os.path.join(output_dir, f"layers_{layer_name}.gpkg")
        gdf.to_file(layer_path, driver='GPKG', layer=layer_name)
        print(f"Saved layer '{layer_name}' as {geom_type} geometry to {layer_path}")
    
    print(f"Backed up all the layers to {os.path.join(os.getcwd(), output_dir)}")

def download_file_from_url(url: str, temp_dir: Optional[str] = None) -> str:
    """
    Download a file from a URL (including Google Drive) and return the local file path.
    
    Args:
        url: The URL to download from
        temp_dir: Optional temporary directory to save the file
        
    Returns:
        str: Path to the downloaded file
        
    Raises:
        requests.RequestException: If download fails
        ValueError: If URL is invalid
    """
    try:
        # Create a temporary file
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        # Generate a temporary filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "downloaded_file.xls"
        temp_file_path = os.path.join(temp_dir, f"temp_{filename}")
        
        logger.info(f"Downloading file from url")
        
        # Handle Google Drive URLs specifically
        if 'drive.google.com' in url or 'drive.usercontent.google.com' in url:
            # For Google Drive, we need to modify the URL to force download
            if 'export=download' not in url:
                if '?' in url:
                    url += '&export=download'
                else:
                    url += '?export=download'
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded file to: {temp_file_path}")
        return temp_file_path
        
    except requests.RequestException as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading file: {e}")
        raise


def is_url(path: str) -> bool:
    """
    Check if the given path is a URL.
    
    Args:
        path: The path to check
        
    Returns:
        bool: True if the path is a URL, False otherwise
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False


def validate_xls_path(xls_path: str) -> Tuple[str, Optional[str]]:
    """
    Validate and process a xls path (URL or local file).
    
    Args:
        xls_path: The xls path (URL or local file path)
        
    Returns:
        tuple: (actual_xls_path, temp_file_path)
            - actual_xls_path: The path to use for reading the xls
            - temp_file_path: The temporary file path if downloaded, None otherwise
            
    Raises:
        FileNotFoundError: If local file doesn't exist
        requests.RequestException: If URL download fails
    """
    if is_url(xls_path):
        logger.info("xls path is a URL, downloading file...")
        temp_file_path = download_file_from_url(xls_path)
        return temp_file_path, temp_file_path
    else:
        # Local file path
        if not os.path.exists(xls_path):
            raise FileNotFoundError(f"xls file not found: {xls_path}")
        return xls_path, None
