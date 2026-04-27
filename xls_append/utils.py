"""
Utility functions for the ArcGIS API library.
"""

import os
import tempfile
import requests
import logging
from urllib.parse import urlparse
from typing import Optional


import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point, Polygon, LineString

# Configure logging
logger = logging.getLogger(__name__)

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


def validate_xls_path(xls_path: str) -> tuple[str, Optional[str]]:
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
