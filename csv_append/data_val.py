import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
import logging
import json
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pandera_type(esri_type_str: str) -> pa.DataType:
    """Maps an Esri field type string to a Pandera DataType."""
    if not isinstance(esri_type_str, str):
        logger.warning(f"Expected string for esri_type_str, got {type(esri_type_str)}")
        return pa.Object
    
    mapping = {
        'esriFieldTypeString': pa.String,
        'esriFieldTypeInteger': pa.Int32,
        'esriFieldTypeSmallInteger': pa.Int16,
        'esriFieldTypeBigInteger': pa.Int64,
        'esriFieldTypeDouble': pa.Float64,
        'esriFieldTypeSingle': pa.Float32,
        'esriFieldTypeDate': pa.DateTime,
        'esriFieldTypeGUID': pa.String,
        'esriFieldTypeGlobalID': pa.String,
        'esriFieldTypeOID': pa.Int64,
        'esriFieldTypeGeometry': pa.Object,
        'esriFieldTypeBoolean': pa.Bool,  # Added boolean type
    }
    
    result = mapping.get(esri_type_str, pa.Object)
    if result == pa.Object:
        logger.warning(f"Unknown Esri field type: {esri_type_str}, defaulting to Object")
    
    return result

def build_pandera_schema_from_layer(feature_layer) -> Optional[pa.DataFrameSchema]:
    """
    Dynamically builds a Pandera DataFrameSchema from an ArcGIS Feature Layer.

    Args:
        feature_layer: The target Esri Feature Layer.

    Returns:
        pandera.DataFrameSchema: A schema object for validating DataFrames.
        None: If feature_layer is invalid or has no properties.
    """
    try:
        if feature_layer is None:
            logger.error("Feature layer is None")
            return None
        
        if not hasattr(feature_layer, 'properties'):
            logger.error("Feature layer does not have properties attribute")
            return None
        
        layer_properties = feature_layer.properties
        if not layer_properties:
            logger.error("Layer properties are empty")
            return None
        
        columns_to_validate = {}

        # These fields are managed by the server and should not be validated on input
        server_managed_fields = []
        
        # Safely get server-managed fields
        object_id_field = layer_properties.get('objectIdField')
        global_id_field = layer_properties.get('globalIdField')
        
        if object_id_field:
            server_managed_fields.append(object_id_field)
        if global_id_field:
            server_managed_fields.append(global_id_field)
        
        # Shape__Area and Shape__Length are also server-managed
        server_managed_fields.extend(['Shape__Area', 'Shape__Length', 'SHAPE__AREA', 'SHAPE__LENGTH'])

        logger.info("Building validation schema from layer properties...")
        
        if not hasattr(layer_properties, 'fields') or not layer_properties.fields:
            logger.warning("No fields found in layer properties")
            return pa.DataFrameSchema({}, strict=False, ordered=False)
        
        logger.info(f"Found {len(layer_properties.fields)} total fields in layer")
        
        for field in layer_properties.fields:
            # Handle PropertyMap, dict, and JSON string formats
            if hasattr(field, 'get'):  # PropertyMap or dict-like object
                # PropertyMap objects have a get method, so we can use them directly
                pass
            elif isinstance(field, str):
                try:
                    field = json.loads(field)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON field: {field}")
                    continue
            elif isinstance(field, dict):
                # Already a dict, use as is
                pass
            else:
                logger.warning(f"Skipping invalid field type: {type(field)} - {field}")
                continue
                
            field_name = field.get('name')
            if not field_name:
                logger.warning("Skipping field with no name")
                continue
            
            field_type = field.get('type')
            if not field_type:
                logger.warning(f"Skipping field '{field_name}' with no type")
                continue
            
            # Log every field we encounter
            logger.info(f"  -> Processing field: '{field_name}' (type: {field_type})")
            
            # Skip server-managed fields
            if field_type in ['esriFieldTypeOID', 'esriFieldTypeGeometry'] or field_name in server_managed_fields:
                logger.info(f"  -> Skipping server-managed field: '{field_name}' (type: {field_type})")
                continue
                
            # Get the corresponding Pandera data type
            pandera_type = get_pandera_type(field_type)
            
            # Get nullable property with default
            nullable = field.get('nullable', True)
            
            # Create a Pandera Column definition
            column_checks = []
            
            # Add string length check if applicable
            if pandera_type == pa.String and field.get('length'):
                max_length = field.get('length')
                if max_length and max_length > 0:  # Avoid division by zero
                    column_checks.append(pa.Check.str_length(max_value=max_length))
            
            columns_to_validate[field_name] = pa.Column(
                dtype=pandera_type,
                nullable=nullable,
                checks=column_checks if column_checks else None
            )
            logger.info(f"  -> Adding rule for field '{field_name}': Type={pandera_type}, Nullable={nullable}")

        # strict=True will fail if DataFrame has columns not in the target layer
        # This ensures validation catches mismatches between DataFrame and layer schema
        schema = pa.DataFrameSchema(columns_to_validate, strict=True, ordered=False)
        logger.info(f"Successfully built schema with {len(columns_to_validate)} fields")
        
        # Debug: Show what fields were included/excluded
        if len(columns_to_validate) == 0:
            logger.warning("⚠️ No fields were added to validation schema!")
            logger.warning(f"Server-managed fields that were skipped: {server_managed_fields}")
            logger.warning("This might mean all fields are being filtered out as server-managed")
        
        return schema
        
    except Exception as e:
        logger.error(f"Error building Pandera schema: {e}")
        return None

def compare_df_esri_types(df: pd.DataFrame, esri_schema: Dict[str, str], pandas_to_esri: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Compare DataFrame column types with expected Esri types.
    
    Args:
        df: Pandas DataFrame to analyze
        esri_schema: Dictionary mapping column names to expected Esri types
        pandas_to_esri: Dictionary mapping pandas dtypes to Esri types
    
    Returns:
        DataFrame with comparison results, or None if error
    """
    try:
        if df is None:
            logger.error("DataFrame is None")
            return None
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            return None
        
        if not isinstance(esri_schema, dict):
            logger.error(f"Expected dict for esri_schema, got {type(esri_schema)}")
            return None
        
        if not isinstance(pandas_to_esri, dict):
            logger.error(f"Expected dict for pandas_to_esri, got {type(pandas_to_esri)}")
            return None
        
        if df.empty:
            logger.warning("DataFrame is empty")
            return pd.DataFrame(columns=["column", "pandas_dtype", "inferred_esri_type", "esri_expected", "status"])
        
        results = []
        
        for col in df.columns:
            try:
                df_dtype = str(df[col].dtype)
                esri_expected = esri_schema.get(col, None)
                esri_inferred = pandas_to_esri.get(df_dtype, "UNKNOWN")
                
                if esri_expected is None:
                    status = "⚠️ Missing in ESRI schema"
                elif esri_expected == esri_inferred:
                    status = "✅ Match"
                else:
                    status = f"❌ Mismatch (expected {esri_expected}, got {esri_inferred})"
                
                results.append({
                    "column": col,
                    "pandas_dtype": df_dtype,
                    "inferred_esri_type": esri_inferred,
                    "esri_expected": esri_expected,
                    "status": status
                })
            except Exception as e:
                logger.warning(f"Error processing column '{col}': {e}")
                results.append({
                    "column": col,
                    "pandas_dtype": "ERROR",
                    "inferred_esri_type": "ERROR",
                    "esri_expected": esri_schema.get(col, None),
                    "status": f"❌ Error: {e}"
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in compare_df_esri_types: {e}")
        return None

# Map pandas dtype to ESRI type (comprehensive mapping)
pandas_to_esri = {
    # String types
    "object": "esriFieldTypeString",
    "string": "esriFieldTypeString",
    
    # Integer types
    "int8": "esriFieldTypeSmallInteger",
    "int16": "esriFieldTypeSmallInteger",
    "int32": "esriFieldTypeInteger",
    "int64": "esriFieldTypeInteger",
    "Int8": "esriFieldTypeSmallInteger",
    "Int16": "esriFieldTypeSmallInteger",
    "Int32": "esriFieldTypeInteger",
    "Int64": "esriFieldTypeBigInteger",
    
    # Float types
    "float32": "esriFieldTypeSingle",
    "float64": "esriFieldTypeDouble",
    "Float32": "esriFieldTypeSingle",
    "Float64": "esriFieldTypeDouble",
    
    # Boolean types
    "bool": "esriFieldTypeSmallInteger",
    "boolean": "esriFieldTypeSmallInteger",
    
    # Date/Time types
    "datetime64[ns]": "esriFieldTypeDate",
    "datetime64[ns, UTC]": "esriFieldTypeDate",
    "datetime64[us]": "esriFieldTypeDate",  # microseconds
    "datetime64[ms]": "esriFieldTypeDate",  # milliseconds
    "datetime64[s]": "esriFieldTypeDate",   # seconds
    "datetime64[D]": "esriFieldTypeDate",   # days
    "datetime64": "esriFieldTypeDate",
    
    # Category types
    "category": "esriFieldTypeString",
}

# comparison = compare_df_esri_types(update_test, esri_schema, pandas_to_esri)
# print(comparison.to_string(index=False))


# --- Check and map DataFrame columns ---
def map_df_to_esri(df: pd.DataFrame, schema: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Map DataFrame columns to Esri schema.
    
    Args:
        df: Pandas DataFrame to analyze
        schema: Dictionary mapping column names to Esri types
    
    Returns:
        DataFrame with mapping results, or None if error
    """
    try:
        if df is None:
            logger.error("DataFrame is None")
            return None
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            return None
        
        if not isinstance(schema, dict):
            logger.error(f"Expected dict for schema, got {type(schema)}")
            return None
        
        if df.empty:
            logger.warning("DataFrame is empty")
            return pd.DataFrame(columns=["column", "esri_type", "status"])
        
        results = []
        for col in df.columns:
            if col in schema:
                results.append({
                    "column": col,
                    "esri_type": schema[col],
                    "status": "✅ exists"
                })
            else:
                results.append({
                    "column": col,
                    "esri_type": None,
                    "status": "⚠️ missing from ESRI schema"
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in map_df_to_esri: {e}")
        return None

# mapped = map_df_to_esri(update_test, esri_schema)

# --- Check and map DataFrame columns ---
def esri_to_df(df: pd.DataFrame, schema: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Check which Esri schema columns exist in DataFrame.
    
    Args:
        df: Pandas DataFrame to analyze
        schema: Dictionary mapping column names to Esri types
    
    Returns:
        DataFrame with mapping results, or None if error
    """
    try:
        if df is None:
            logger.error("DataFrame is None")
            return None
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            return None
        
        if not isinstance(schema, dict):
            logger.error(f"Expected dict for schema, got {type(schema)}")
            return None
        
        if not schema:
            logger.warning("Schema is empty")
            return pd.DataFrame(columns=["column", "esri_type", "status"])
        
        results = []
        for col in schema:
            if col in df.columns:
                results.append({
                    "column": col,
                    "esri_type": schema[col],
                    "status": "✅ exists"
                })
            else:
                results.append({
                    "column": col,
                    "esri_type": schema[col],
                    "status": "⚠️ missing from INPUT schema"
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in esri_to_df: {e}")
        return None

# mapped = esri_to_df(update_test, esri_schema)
# mapped

def validate_dataframe_with_schema(df: pd.DataFrame, schema: pa.DataFrameSchema) -> Dict[str, Any]:
    """
    Validate a DataFrame against a Pandera schema and return detailed results.
    
    Args:
        df: Pandas DataFrame to validate
        schema: Pandera DataFrameSchema to validate against
    
    Returns:
        Dictionary with validation results including success status, errors, and warnings
    """
    try:
        if df is None:
            logger.error("DataFrame is None")
            return {"success": False, "error": "DataFrame is None", "details": None}
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            return {"success": False, "error": f"Expected DataFrame, got {type(df)}", "details": None}
        
        if schema is None:
            logger.error("Schema is None")
            return {"success": False, "error": "Schema is None", "details": None}
        
        if df.empty:
            logger.warning("DataFrame is empty")
            return {"success": True, "warning": "DataFrame is empty", "details": None}
        
        # Perform validation
        try:
            validated_df = schema.validate(df)
            logger.info("✅ DataFrame validation successful!")
            return {
                "success": True, 
                "error": None, 
                "warning": None,
                "details": {
                    "validated_rows": len(validated_df),
                    "validated_columns": len(validated_df.columns),
                    "schema_fields": len(schema.columns) if hasattr(schema, 'columns') else 0
                }
            }
        except SchemaError as e:
            logger.error(f"❌ DataFrame validation failed: {e}")
            return {
                "success": False, 
                "error": str(e), 
                "warning": None,
                "details": {
                    "schema_error": True,
                    "error_type": type(e).__name__,
                    "failed_checks": getattr(e, 'failure_cases', None)
                }
            }
        except Exception as e:
            logger.error(f"❌ Unexpected validation error: {e}")
            return {
                "success": False, 
                "error": f"Unexpected error: {str(e)}", 
                "warning": None,
                "details": {
                    "unexpected_error": True,
                    "error_type": type(e).__name__
                }
            }
            
    except Exception as e:
        logger.error(f"Error in validate_dataframe_with_schema: {e}")
        return {"success": False, "error": f"Function error: {str(e)}", "details": None}

def validate_dataframe_against_layer(df: pd.DataFrame, feature_layer) -> Dict[str, Any]:
    """
    Validate a DataFrame against an ArcGIS Feature Layer schema.
    
    Args:
        df: Pandas DataFrame to validate
        feature_layer: ArcGIS Feature Layer to get schema from
    
    Returns:
        Dictionary with validation results
    """
    try:
        # Build schema from layer
        schema = build_pandera_schema_from_layer(feature_layer)
        
        if schema is None:
            return {
                "success": False, 
                "error": "Failed to build schema from feature layer", 
                "details": None
            }
        
        # Validate DataFrame
        return validate_dataframe_with_schema(df, schema)
        
    except Exception as e:
        logger.error(f"Error in validate_dataframe_against_layer: {e}")
        return {"success": False, "error": f"Function error: {str(e)}", "details": None}

def test_validation_with_sample_data():
    """
    Test function to demonstrate Pandera validation working with sample data.
    This creates a simple test case to verify validation is functioning.
    """
    try:
        # Create sample DataFrame with intentional validation issues
        test_data = {
            'name': ['John', 'Jane', 'Bob'],  # String data - should be fine
            'age': [25, 30, 'invalid_age'],   # Mixed types - should fail
            'score': [85.5, 92.0, 78.3],     # Float data - should be fine
            'active': [True, False, True]     # Boolean data - should be fine
        }
        test_df = pd.DataFrame(test_data)
        
        # Create a simple schema for testing
        test_schema = pa.DataFrameSchema({
            'name': pa.Column(pa.String, nullable=False),
            'age': pa.Column(pa.Int32, nullable=False),
            'score': pa.Column(pa.Float64, nullable=True),
            'active': pa.Column(pa.Bool, nullable=True)
        })
        
        print("🧪 Testing Pandera validation with sample data...")
        print(f"Test DataFrame:\n{test_df}")
        print(f"Test DataFrame dtypes:\n{test_df.dtypes}")
        
        # Test validation
        result = validate_dataframe_with_schema(test_df, test_schema)
        
        print(f"\n📊 Validation Result:")
        print(f"Success: {result['success']}")
        if result['error']:
            print(f"Error: {result['error']}")
        if result['warning']:
            print(f"Warning: {result['warning']}")
        if result['details']:
            print(f"Details: {result['details']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in test_validation_with_sample_data: {e}")
        return {"success": False, "error": f"Test error: {str(e)}", "details": None}

def debug_layer_fields(feature_layer):
    """
    Debug function to show what fields are in the layer and why they might be filtered out.
    """
    try:
        if not hasattr(feature_layer, 'properties'):
            print("❌ Feature layer has no properties")
            return
        
        props = feature_layer.properties
        print(f"Layer name: {props.get('name', 'Unknown')}")
        print(f"Total fields: {len(props.fields) if hasattr(props, 'fields') else 0}")
        
        if hasattr(props, 'fields'):
            print("\nAll fields in layer:")
            for i, field in enumerate(props.fields):
                name = field.get('name', 'No name')
                ftype = field.get('type', 'No type')
                nullable = field.get('nullable', True)
                print(f"  {i+1}. {name} | {ftype} | nullable={nullable}")
        
        # Show server-managed fields
        server_managed = []
        object_id_field = props.get('objectIdField')
        global_id_field = props.get('globalIdField')
        if object_id_field:
            server_managed.append(object_id_field)
        if global_id_field:
            server_managed.append(global_id_field)
        server_managed.extend(['Shape__Area', 'Shape__Length', 'SHAPE__AREA', 'SHAPE__LENGTH'])
        
        print(f"\nServer-managed fields that will be skipped: {server_managed}")
        
    except Exception as e:
        print(f"Error debugging layer fields: {e}")

def column_update_gap(df, list_column_to_add):
    df = df.copy()
    for col in list_column_to_add:
        if col == 'value_us_consolidated_float':
            df[col] = df['value_us_consolidated'].astype(float)
        elif col == 'value_non_consolidated_float':
            df[col] = df['value_non_consolidated'].astype(float)
        elif col == 'year_time':
            df[col] = pd.to_datetime(df['year'].astype(str) + '-01-01')
        else:
            df[col] = None
    
    return df

def get_missing_columns(df: pd.DataFrame, feature_layer) -> Dict[str, List[str]]:
    """
    Get a list of columns that are missing from DataFrame compared to the layer schema.
    
    Args:
        df: Pandas DataFrame to check
        feature_layer: ArcGIS Feature Layer to compare against
    
    Returns:
        Dictionary with 'missing_in_df' and 'extra_in_df' lists
    """
    try:
        schema = build_pandera_schema_from_layer(feature_layer)
        if schema is None:
            return {"missing_in_df": [], "extra_in_df": []}
        
        df_cols = set(df.columns)
        schema_cols = set(schema.columns.keys()) if hasattr(schema, 'columns') else set()
        
        missing_in_df = list(schema_cols - df_cols)
        extra_in_df = list(df_cols - schema_cols)
        
        return {
            "missing_in_df": missing_in_df,
            "extra_in_df": extra_in_df
        }
        
    except Exception as e:
        logger.error(f"Error in get_missing_columns: {e}")
        return {"missing_in_df": [], "extra_in_df": []}

def validate_df_against_layer(df: pd.DataFrame, feature_layer) -> bool:
    """
    Simple validation function that returns True if DataFrame matches layer schema, False otherwise.
    
    Args:
        df: Pandas DataFrame to validate
        feature_layer: ArcGIS Feature Layer to validate against
    
    Returns:
        bool: True if valid, False if invalid
    """
    try:
        schema = build_pandera_schema_from_layer(feature_layer)
        if schema is None:
            return False
        
        schema.validate(df)
        return True
    except Exception:
        return False

def force_update_dtype_csv(df_csv, server_gdb):
    print('force_update_dtype_csv')
    reference_gdb = server_gdb.copy()
    new_updated_df = df_csv.copy()

    try:
        unique_columns = reference_gdb.metric_code.unique()
        new_updated_df = new_updated_df[new_updated_df.metric_code.isin(unique_columns)]
        if new_updated_df.empty:
            raise ValueError('empty filtered data, check the update csv data, metric_code, is it the same with target layer?')
        
        for col in new_updated_df.columns:
            if col in reference_gdb.columns:
                print(f'found column {col} in reference')
                new_updated_df[col] = new_updated_df[col].astype(reference_gdb[col].dtype)
            else:
                error_v = f"can't find column {col} in reference, mapping is needed"
                # print(error_v)
                raise ValueError(error_v)

    except Exception as e:
            print(f'something went wrong {e}')
        
    return new_updated_df