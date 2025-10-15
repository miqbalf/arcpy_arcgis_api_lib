import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache

from arcgis.gis import GIS
from arcgis.layers import Service

import pandas as pd
import geopandas as gpd
# convert the feature layer arcgis data into geopandas object, it will takes some few minutes
from arcgis.features import GeoAccessor, GeoSeriesAccessor

from .data_val import (map_df_to_esri, 
                esri_to_df, compare_df_esri_types, 
                pandas_to_esri , build_pandera_schema_from_layer, 
                force_update_dtype_csv, get_missing_columns, column_update_gap, validate_df_against_layer)
from .utils import validate_csv_path

# from arcgis.mapping import WebMap
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# default var
api_key = os.getenv('API_KEY')
portal_url = os.getenv('PORTAL_URL')
query = os.getenv('QUERY')
item_type = os.getenv('ITEM_TYPE')


class ArcGISAPI:
    """Optimized ArcGIS API wrapper class with improved error handling and performance."""
    
    def __init__(self, api_key: Optional[str] = None, portal_url: Optional[str] = None, 
                 query: Optional[str] = None, item_type: Optional[str] = None, csv_path: Optional[str] = None):
        """Initialize ArcGIS API connection with lazy loading for better performance."""
        self.portal_url = portal_url or os.getenv('PORTAL_URL')
        self.api_key = api_key or os.getenv('API_KEY')
        self.query = query or os.getenv('QUERY')
        self.item_type = item_type or os.getenv('ITEM_TYPE')
        
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
        
        if csv_path is None:
            self.csv_path = os.getenv('CSV_PATH')
        else:
            self.csv_path = csv_path     
        
        if not self.csv_path:
            raise ValueError("CSV path must be provided either as parameter or CSV_PATH environment variable")
        
        # Handle URL vs local file path using utils
        actual_csv_path, self._temp_file_path = validate_csv_path(self.csv_path)
        
        self.input_df = pd.read_csv(actual_csv_path)
        logger.info(f"Successfully loaded CSV with {len(self.input_df)} rows and {len(self.input_df.columns)} columns")

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
            if not self.api_key or not self.portal_url:
                raise ValueError("API_KEY and PORTAL_URL must be provided")
            
            portal = GIS(self.portal_url, api_key=self.api_key)
            # logger.info(f"Successfully connected to portal: {self.portal_url}")
            logger.info(f"Successfully connected to portal")
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
            
            logger.info(f"📘 Field info for: {feature_layer.properties.name}")
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


    def csv_wrangling(self, layer_name: Optional[str] = None, df_reference: Optional[pd.DataFrame] = None, reference_layer=None) -> Optional[pd.DataFrame]:
        """Load and process CSV data with error handling."""
        try:
            
            # Add basic data quality checks
            if self.input_df.empty:
                logger.warning("CSV file is empty")
            
            esri_schema_api = self.describe_layer_fields(layer_name)
            if esri_schema_api is None:
                raise ValueError(f"Could not retrieve schema for layer: {layer_name}")
            esri_schema = {i['name']:i['type'] for i in esri_schema_api}

            mapped_df_esri = map_df_to_esri(self.input_df, esri_schema)
            # mapped_df_esri

            mapped_esri_df = esri_to_df(self.input_df, esri_schema)

            # Get the feature layer for reference if not provided
            if reference_layer is None and layer_name is not None:
                result = self.get_feature_layer_tile_by_name(layer_name)
                if result:
                    reference_layer = result[1]  # Get the feature layer (second element of tuple)
                else:
                    raise ValueError(f"Could not retrieve feature layer for: {layer_name}")

            col_not_exist_not_system = get_missing_columns(self.input_df, reference_layer)

            # repair, adding the data first, if there is any col missing
            if col_not_exist_not_system and col_not_exist_not_system.get('missing_in_df'):
                new_updated_df_revised = column_update_gap(self.input_df,col_not_exist_not_system['missing_in_df'])
            else:
                new_updated_df_revised = self.input_df

            # new_updated_df_revised.columns

            comparison_old = compare_df_esri_types(new_updated_df_revised, esri_schema, pandas_to_esri)
        
            # Apply dtype updates with error handling
            updated_df = force_update_dtype_csv(new_updated_df_revised, df_reference)
            
            # Check if force_update_dtype_csv returned valid data
            if updated_df is None or updated_df.empty:
                logger.warning("force_update_dtype_csv returned None or empty data. Using original data.")
                logger.warning("This might indicate a mismatch between CSV data and target layer schema.")
                logger.warning("Please check if metric_code and other fields match the target layer requirements.")
                
                # Use the debug method to get detailed information
                self.debug_data_mismatch(new_updated_df_revised, df_reference, layer_name)
                
                # Keep the original data instead of None
                new_updated_df_revised = new_updated_df_revised
            else:
                new_updated_df_revised = updated_df
                logger.info(f"Successfully updated data types. New shape: {new_updated_df_revised.shape}")
                
            comparison_new = compare_df_esri_types(new_updated_df_revised, esri_schema, pandas_to_esri)
            
            # Only perform validation if reference_layer is available and data is valid
            if reference_layer is not None and new_updated_df_revised is not None and not new_updated_df_revised.empty:
                try:
                    validation_schema = build_pandera_schema_from_layer(reference_layer)
                    validation_schema.validate(new_updated_df_revised)
                    print("✅ Valid!")
                    
                    is_valid = validate_df_against_layer(new_updated_df_revised, reference_layer)
                    print(f"Validation result: {is_valid}")
                except Exception as e:
                    print(f"❌ Invalid: {e}")
                    is_valid = False
            else:
                if reference_layer is None:
                    print("⚠️ No reference layer available for validation")
                elif new_updated_df_revised is None or new_updated_df_revised.empty:
                    print("⚠️ No valid data available for validation")
                is_valid = None

            # Only update self.input_df if we have valid data
            if new_updated_df_revised is not None and not new_updated_df_revised.empty:
                self.input_df = new_updated_df_revised
            else:
                logger.warning("Keeping original input_df due to invalid processed data")
                # Keep the original input_df unchanged

            return {'mapped_df_esri': mapped_df_esri,
                    'mapped_esri_df':mapped_esri_df,
                    'comparison_old':comparison_old,
                    'comparison_new':comparison_new,
                    'is_valid':is_valid,
                    'new_updated_df_revised':new_updated_df_revised,
                    'data_processing_successful': new_updated_df_revised is not None and not new_updated_df_revised.empty

                    }
            
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            return None
    
    def debug_data_mismatch(self, csv_df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug method to help identify data mismatches between CSV and target layer.
        
        Args:
            csv_df: The CSV DataFrame
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
