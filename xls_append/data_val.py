import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
import logging
import json
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pandera_type(esri_type_str: str, coerce: bool = True) -> pa.DataType:
    """
    Maps an Esri field type string to a Pandera DataType.
    
    Args:
        esri_type_str: Esri field type string (e.g., 'esriFieldTypeInteger')
        coerce: If True, allows type coercion during validation (e.g., int64 -> int32)
    
    Returns:
        Pandera DataType
    """
    if not isinstance(esri_type_str, str):
        logger.warning(f"Expected string for esri_type_str, got {type(esri_type_str)}")
        return pa.Object
    
    # Note: For integers, we use Int32/Int64 (nullable) instead of int32/int64
    # This allows for better handling of missing values and type coercion
    mapping = {
        'esriFieldTypeString': pa.String,
        'esriFieldTypeInteger': pa.Int64 if not coerce else pa.Int,  # Allow int64 for compatibility
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

def build_pandera_schema_from_layer(feature_layer, coerce: bool = True) -> Optional[pa.DataFrameSchema]:
    """
    Dynamically builds a Pandera DataFrameSchema from an ArcGIS Feature Layer.

    Args:
        feature_layer: The target Esri Feature Layer.
        coerce: If True, enables type coercion during validation (recommended). Default: True

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
            pandera_type = get_pandera_type(field_type, coerce=coerce)
            
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
                checks=column_checks if column_checks else None,
                coerce=coerce  # Enable type coercion for better compatibility
            )
            logger.info(f"  -> Adding rule for field '{field_name}': Type={pandera_type}, Nullable={nullable}, Coerce={coerce}")

        # strict=True will fail if DataFrame has columns not in the target layer
        # This ensures validation catches mismatches between DataFrame and layer schema
        # coerce=True allows automatic type conversion (e.g., int64 -> int32)
        schema = pa.DataFrameSchema(columns_to_validate, strict=True, ordered=False, coerce=coerce)
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

def clean_numeric_string(value):
    """
    Clean numeric string by removing commas and other non-numeric characters.
    
    Args:
        value: String value that may contain commas (e.g., "350,000", "$1,234.56", "1,000,000")
        
    Returns:
        Cleaned string ready for float conversion, or None if invalid
    """
    if pd.isna(value) or value is None:
        return None
    
    # Convert to string and strip whitespace
    str_value = str(value).strip()
    
    # If already a number, return as is
    try:
        float(str_value)
        return str_value
    except ValueError:
        pass
    
    # Remove common currency symbols and other non-numeric characters
    import re
    # Remove commas, currency symbols, spaces, and other non-numeric characters
    # Keep only digits, decimal point, and minus sign
    cleaned = re.sub(r'[^\d.-]', '', str_value)
    
    # Handle edge cases
    if not cleaned or cleaned == '.' or cleaned == '-':
        return None
    
    # Ensure only one decimal point
    if cleaned.count('.') > 1:
        # Keep only the first decimal point
        parts = cleaned.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    
    # Ensure minus sign is at the beginning
    if '-' in cleaned and not cleaned.startswith('-'):
        cleaned = '-' + cleaned.replace('-', '')
    
    return cleaned if cleaned else None


def column_update_gap(df, list_column_to_add):
    """
    Add missing columns to DataFrame with default values.
    Skips geometry/SHAPE fields.
    
    Args:
        df: DataFrame to add columns to
        list_column_to_add: List of column names to add
    
    Returns:
        DataFrame with added columns
    """
    df = df.copy()
    
    # Skip geometry/SHAPE fields
    geometry_patterns = ['SHAPE', 'Shape', 'shape', 'GEOMETRY', 'geometry']
    
    for col in list_column_to_add:
        # Skip geometry fields
        if any(col.startswith(pattern) or col == pattern for pattern in geometry_patterns):
            print(f"⊘ Skipping geometry field: {col}")
            continue
            
        # Handle specific transformations
        if col == 'value_us_consolidated_float':
            # Clean the string values before converting to float
            if 'value_us_consolidated' in df.columns:
                df[col] = df['value_us_consolidated'].apply(clean_numeric_string).astype(float)
            else:
                df[col] = None
        elif col == 'value_non_consolidated_float':
            # Clean the string values before converting to float
            if 'value_non_consolidated' in df.columns:
                df[col] = df['value_non_consolidated'].apply(clean_numeric_string).astype(float)
            else:
                df[col] = None
        elif col == 'year_time':
            if 'year' in df.columns:
                df[col] = pd.to_datetime(df['year'].astype(str) + '-01-01')
            else:
                df[col] = None
        else:
            # Add placeholder with None
            df[col] = None
            print(f"  + Added placeholder column: {col}")
    
    return df

def get_missing_columns(df: pd.DataFrame, feature_layer) -> Dict[str, List[str]]:
    """
    Get a list of columns that are missing from DataFrame compared to the layer schema.
    Excludes server-managed fields and geometry fields.
    
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
        
        # Remove geometry/SHAPE columns from comparison
        geometry_patterns = {'SHAPE', 'Shape', 'shape', 'geometry', 'GEOMETRY'}
        df_cols_filtered = {col for col in df_cols if not any(
            col.startswith(pattern) or col == pattern 
            for pattern in geometry_patterns
        )}
        
        missing_in_df = list(schema_cols - df_cols_filtered)
        extra_in_df = list(df_cols_filtered - schema_cols)
        
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

def graph_office_type(metric_code, office_type_fe,is_polygon=True):

    if is_polygon:
        placeholder = 'WWF Office'
    else:
        if office_type_fe == 'Country Office':
            placeholder = 'Country Office'
        elif office_type_fe == 'National Office / Partners':
            placeholder = 'National Office'
        else:
            placeholder = 'WWF Office'
    

    if metric_code == 'STA100':
        return f'Staff and Headcount by {placeholder}'
    elif metric_code == 'F1':
        return f'CO2 Emissions (mTCO2e) by {placeholder}'
    elif metric_code == 'E1':
        return f'Staff (FTE) by {placeholder}'
    elif metric_code == 'DDD400':
        return f'Supporters & Partners (EUR) by {placeholder}'
    elif metric_code == 'BBB110':
        return f'Conservation Spend (EUR) by {placeholder}'
    elif metric_code == 'BB3':
        return f'Conservation Spend (EUR) by {placeholder}'
    elif metric_code == 'AAA100':
        return f'Donated Income (EUR) by  {placeholder}'
    else:
        return None


def multi_filter(df: pd.DataFrame, conditions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Applies a list of complex filter conditions to a DataFrame using df.query().

    Args:
        df: The pandas DataFrame to filter.
        conditions: A list of dictionaries, where each dict defines one filter:
            [{
                "column": "column_name",
                "operator": "IN" or "NOT IN" or "=" or "!=",
                "values": list_of_values (for IN/NOT IN) or single_value (for =/!=)
            }, ...]

    Returns:
        The filtered DataFrame.
    """

    query_parts = []
    local_vars = {}

    for i, condition in enumerate(conditions):
        col = condition["column"]
        op = condition["operator"].upper()
        values = condition["values"]

        # Unique variable name for each condition
        var_name = f"val_{i}"
        local_vars[var_name] = values

        if op == "IN":
            query_parts.append(f"`{col}` in @{var_name}")
        elif op == "NOT IN":
            query_parts.append(f"`{col}` not in @{var_name}")
        elif op == "=":
            query_parts.append(f"`{col}` == @{var_name}")
        elif op == "!=":
            query_parts.append(f"`{col}` != @{var_name}")
        else:
            raise ValueError(f"Unsupported operator: {op}")

    # Combine with AND logic (you can change to ' or ' if desired)
    query_str = " and ".join(f"({q})" for q in query_parts)

    print(f"Generated query: {query_str}")

    return df.query(query_str, local_dict=local_vars)

def pandas_dtype_to_esri_field_def(col_name: str, dtype, sample_value=None, max_length: int = 255) -> Dict[str, Any]:
    """
    Convert pandas dtype to Esri field definition dictionary.
    
    Args:
        col_name: Column name
        dtype: pandas dtype
        sample_value: Sample value from the column (for better type inference)
        max_length: Maximum length for string fields
    
    Returns:
        Dictionary with Esri field definition
    """
    dtype_str = str(dtype)
    
    # Map pandas dtype to Esri field type
    if dtype_str in ['object', 'string'] or 'str' in dtype_str:
        return {
            'name': col_name,
            'type': 'esriFieldTypeString',
            'alias': col_name,
            'length': max_length,
            'nullable': True
        }
    elif dtype_str in ['int8', 'int16', 'Int8', 'Int16']:
        return {
            'name': col_name,
            'type': 'esriFieldTypeSmallInteger',
            'alias': col_name,
            'nullable': True
        }
    elif dtype_str in ['int32', 'Int32']:
        return {
            'name': col_name,
            'type': 'esriFieldTypeInteger',
            'alias': col_name,
            'nullable': True
        }
    elif dtype_str in ['int64', 'Int64']:
        return {
            'name': col_name,
            'type': 'esriFieldTypeBigInteger',
            'alias': col_name,
            'nullable': True
        }
    elif dtype_str in ['float32', 'Float32']:
        return {
            'name': col_name,
            'type': 'esriFieldTypeSingle',
            'alias': col_name,
            'nullable': True
        }
    elif dtype_str in ['float64', 'Float64']:
        return {
            'name': col_name,
            'type': 'esriFieldTypeDouble',
            'alias': col_name,
            'nullable': True
        }
    elif dtype_str in ['bool', 'boolean']:
        return {
            'name': col_name,
            'type': 'esriFieldTypeSmallInteger',
            'alias': col_name,
            'nullable': True
        }
    elif 'datetime' in dtype_str:
        return {
            'name': col_name,
            'type': 'esriFieldTypeDate',
            'alias': col_name,
            'nullable': True
        }
    else:
        # Default to string for unknown types
        logger.warning(f"Unknown dtype '{dtype_str}' for column '{col_name}', defaulting to String")
        return {
            'name': col_name,
            'type': 'esriFieldTypeString',
            'alias': col_name,
            'length': max_length,
            'nullable': True
        }

def add_missing_fields_to_layer(df: pd.DataFrame, feature_layer, dry_run: bool = True) -> Dict[str, Any]:
    """
    Add missing fields from DataFrame to ArcGIS Feature Layer.
    Automatically skips geometry and server-managed fields.
    
    Args:
        df: Pandas DataFrame with columns to add
        feature_layer: ArcGIS Feature Layer to update
        dry_run: If True, only show what would be added without making changes
    
    Returns:
        Dictionary with results:
            - success: bool
            - fields_to_add: list of field definitions
            - fields_added: list of field names (if dry_run=False)
            - error: error message if any
    """
    try:
        if df is None or df.empty:
            return {"success": False, "error": "DataFrame is empty", "fields_to_add": []}
        
        if feature_layer is None:
            return {"success": False, "error": "Feature layer is None", "fields_to_add": []}
        
        # Get current layer schema and properties
        layer_fields = {field.get('name'): field for field in feature_layer.properties.fields}
        layer_properties = feature_layer.properties
        
        # Build set of fields to skip (server-managed and geometry)
        skip_fields = set()
        
        # Add server-managed field names
        object_id_field = layer_properties.get('objectIdField')
        global_id_field = layer_properties.get('globalIdField')
        if object_id_field:
            skip_fields.add(object_id_field)
        if global_id_field:
            skip_fields.add(global_id_field)
        
        # Add geometry field patterns
        skip_fields.update([
            'SHAPE', 'Shape', 'shape',
            'SHAPE__Area', 'SHAPE__Length',
            'Shape__Area', 'Shape__Length',
            'Shape_Area', 'Shape_Length',
            'SHAPE_Area', 'SHAPE_Length',
            'geometry', 'GEOMETRY', 'Geometry'
        ])
        
        # Find missing columns, excluding server-managed and geometry fields
        missing_cols = []
        skipped_cols = []
        
        for col in df.columns:
            # Skip if already exists
            if col in layer_fields:
                continue
            
            # Skip server-managed and geometry fields
            if col in skip_fields:
                skipped_cols.append(col)
                logger.info(f"Skipping server-managed/geometry field: {col}")
                continue
            
            # Skip fields that match geometry patterns
            if any(col.startswith(pattern) or col == pattern for pattern in ['SHAPE', 'Shape', 'shape', 'geometry', 'GEOMETRY']):
                skipped_cols.append(col)
                logger.info(f"Skipping geometry field: {col}")
                continue
            
            missing_cols.append(col)
        
        if skipped_cols:
            logger.info(f"Skipped {len(skipped_cols)} server-managed/geometry fields: {skipped_cols}")
        
        if not missing_cols:
            logger.info("No missing fields to add")
            message = "No fields to add"
            if skipped_cols:
                message += f" (skipped {len(skipped_cols)} server-managed/geometry fields)"
            return {"success": True, "fields_to_add": [], "message": message}
        
        # Build field definitions for missing columns
        fields_to_add = []
        for col in missing_cols:
            # Get sample value for better type inference
            sample_value = df[col].iloc[0] if not df[col].empty else None
            field_def = pandas_dtype_to_esri_field_def(col, df[col].dtype, sample_value)
            fields_to_add.append(field_def)
            logger.info(f"Field to add: {field_def}")
        
        if dry_run:
            logger.info("DRY RUN: The following fields would be added:")
            for field in fields_to_add:
                logger.info(f"  - {field['name']} ({field['type']})")
            return {
                "success": True,
                "dry_run": True,
                "fields_to_add": fields_to_add,
                "message": f"Dry run complete. {len(fields_to_add)} fields would be added."
            }
        
        # Actually add fields to the layer
        try:
            # Add fields using the manager's add_to_definition method
            add_result = feature_layer.manager.add_to_definition({
                "fields": fields_to_add
            })
            
            if add_result.get('success', False):
                logger.info(f"Successfully added {len(fields_to_add)} fields to layer")
                return {
                    "success": True,
                    "dry_run": False,
                    "fields_to_add": fields_to_add,
                    "fields_added": [f['name'] for f in fields_to_add],
                    "message": f"Successfully added {len(fields_to_add)} fields"
                }
            else:
                error_msg = add_result.get('error', 'Unknown error')
                logger.error(f"Failed to add fields: {error_msg}")
                return {
                    "success": False,
                    "fields_to_add": fields_to_add,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Error calling add_to_definition: {e}")
            return {
                "success": False,
                "fields_to_add": fields_to_add,
                "error": f"API error: {str(e)}"
            }
    
    except Exception as e:
        logger.error(f"Error in add_missing_fields_to_layer: {e}")
        return {"success": False, "error": str(e), "fields_to_add": []}

def force_update_dtype_xls(df_xls, server_gdb=None, feature_layer=None):
    """
    Force update DataFrame dtypes to match the target layer schema.
    
    Args:
        df_xls: Input DataFrame to convert
        server_gdb: Optional reference DataFrame with existing data (used for unique_columns filtering)
        feature_layer: ArcGIS Feature Layer object (used to get schema information)
    
    Returns:
        DataFrame with converted dtypes
    """
    print('force_update_dtype_xls')
    new_updated_df = df_xls.copy()
    
    # Determine unique_columns for filtering
    # If server_gdb is empty/None, use df_xls; otherwise use server_gdb
    if server_gdb is None or (isinstance(server_gdb, pd.DataFrame) and server_gdb.empty):
        # No reference data - use input df_xls to get unique metric_codes
        print("ℹ️ No reference data provided, using input df for unique columns")
        if 'metric_code' in new_updated_df.columns:
            unique_columns = new_updated_df.metric_code.unique()
            print(f"Found {len(unique_columns)} unique metric_codes in input data")
        else:
            print("⚠️ metric_code column not found")
            unique_columns = None
    else:
        # Use server_gdb to get unique metric_codes
        server_gdb = server_gdb.copy()
        if 'metric_code' in server_gdb.columns:
            unique_columns = server_gdb.metric_code.unique()
            print(f"Found {len(unique_columns)} unique metric_codes in reference data")
        else:
            print("⚠️ metric_code column not found in reference")
            unique_columns = None
    
    # Get schema from feature layer (this is the source of truth for dtypes)
    target_schema = None
    if feature_layer is not None:
        try:
            print("📋 Getting schema from feature layer...")
            layer_fields = feature_layer.properties.fields
            layer_properties = feature_layer.properties
            target_schema = {}
            
            # Fields to skip (server-managed and geometry fields)
            skip_fields = set()
            
            # Get server-managed field names
            object_id_field = layer_properties.get('objectIdField')
            global_id_field = layer_properties.get('globalIdField')
            if object_id_field:
                skip_fields.add(object_id_field)
            if global_id_field:
                skip_fields.add(global_id_field)
            
            # Add common geometry and server-managed field patterns
            skip_fields.update([
                'SHAPE', 'Shape', 'shape',
                'SHAPE__Area', 'SHAPE__Length', 
                'Shape__Area', 'Shape__Length',
                'Shape_Area', 'Shape_Length',
                'SHAPE_Area', 'SHAPE_Length'
            ])
            
            # Map Esri types to pandas dtypes
            esri_to_pandas = {
                'esriFieldTypeString': 'object',
                'esriFieldTypeInteger': 'int32',
                'esriFieldTypeSmallInteger': 'int16',
                'esriFieldTypeBigInteger': 'int64',
                'esriFieldTypeDouble': 'float64',
                'esriFieldTypeSingle': 'float32',
                'esriFieldTypeDate': 'datetime64[ns]',
                'esriFieldTypeOID': 'int64',
                'esriFieldTypeGlobalID': 'object',
                'esriFieldTypeGUID': 'object',
            }
            
            skipped_count = 0
            for field in layer_fields:
                field_name = field.get('name')
                field_type = field.get('type')
                
                # Skip geometry and server-managed fields
                if field_type in ['esriFieldTypeOID', 'esriFieldTypeGeometry', 'esriFieldTypeGlobalID']:
                    print(f"  ⊘ Skipping server-managed/geometry field: {field_name} ({field_type})")
                    skipped_count += 1
                    continue
                
                if field_name in skip_fields:
                    print(f"  ⊘ Skipping field: {field_name} (in skip list)")
                    skipped_count += 1
                    continue
                
                if field_name and field_type:
                    pandas_dtype = esri_to_pandas.get(field_type, 'object')
                    target_schema[field_name] = pandas_dtype
                    
            print(f"✅ Retrieved schema with {len(target_schema)} editable fields from layer (skipped {skipped_count} server-managed/geometry fields)")
            
        except Exception as e:
            print(f"⚠️ Could not get schema from feature layer: {e}")
            target_schema = None
    
    # If no feature layer provided, try to build schema from server_gdb
    if target_schema is None and server_gdb is not None and not server_gdb.empty:
        print("📋 Using reference data dtypes as schema")
        target_schema = {col: str(dtype) for col, dtype in server_gdb.dtypes.items()}
    
    if target_schema is None:
        print("⚠️ No schema available for dtype conversion, returning original DataFrame")
        return new_updated_df
    
    try:
        # Filter by metric_code if unique_columns available
        if unique_columns is not None and 'metric_code' in new_updated_df.columns:
            print(f"Filtering data by {len(unique_columns)} unique metric_codes...")
            new_updated_df = new_updated_df[new_updated_df.metric_code.isin(unique_columns)]
            if new_updated_df.empty:
                raise ValueError('empty filtered data, check the update xls data, metric_code, is it the same with target layer?')
            print(f"✅ Filtered to {len(new_updated_df)} rows")
        
        # Convert dtypes to match target schema
        print("\n🔄 Converting dtypes to match target schema...")
        for col in new_updated_df.columns:
            if col in target_schema:
                print(f'Processing column: {col}')
                target_dtype = target_schema[col]
                current_dtype = str(new_updated_df[col].dtype)
                
                # Skip if dtypes already match
                if current_dtype == target_dtype:
                    print(f'  ✓ {col} already has correct dtype: {target_dtype}')
                    continue
                
                # Handle float conversion with string cleaning
                if 'float' in target_dtype:
                    print(f'  → Converting {col} from {current_dtype} to {target_dtype} (with string cleaning)')
                    try:
                        new_updated_df[col] = new_updated_df[col].apply(clean_numeric_string).astype(target_dtype)
                        print(f'  ✅ Converted successfully')
                    except Exception as e:
                        print(f'  ⚠️ Warning: Could not convert {col} to {target_dtype}: {e}')
                        continue
                
                # Handle integer type conversions (int64 -> int32, etc)
                elif 'int' in target_dtype:
                    print(f'  → Converting {col} from {current_dtype} to {target_dtype}')
                    try:
                        # First handle any NaN values
                        if new_updated_df[col].isna().any():
                            nan_count = new_updated_df[col].isna().sum()
                            print(f'     ⚠️ {col} contains {nan_count} NaN values, filling with 0')
                            new_updated_df[col] = new_updated_df[col].fillna(0)
                        
                        # For int64 -> int32 conversion, check if values are in range
                        if target_dtype == 'int32' and 'int64' in current_dtype:
                            max_val = new_updated_df[col].max()
                            min_val = new_updated_df[col].min()
                            if max_val > 2147483647 or min_val < -2147483648:
                                print(f'     ⚠️ Warning: {col} values exceed int32 range (min={min_val}, max={max_val})')
                                print(f'     Keeping as int64 to avoid data loss')
                                continue  # Skip conversion if out of range
                        
                        new_updated_df[col] = new_updated_df[col].astype(target_dtype)
                        print(f'  ✅ Converted successfully')
                    except (ValueError, OverflowError) as e:
                        print(f'  ⚠️ Warning: Could not convert {col} to {target_dtype}: {e}')
                        print(f'     Keeping original dtype: {current_dtype}')
                        continue
                
                # Handle datetime conversions
                elif 'datetime' in target_dtype:
                    print(f'  → Converting {col} from {current_dtype} to {target_dtype}')
                    try:
                        new_updated_df[col] = pd.to_datetime(new_updated_df[col])
                        print(f'  ✅ Converted successfully')
                    except Exception as e:
                        print(f'  ⚠️ Warning: Could not convert {col} to {target_dtype}: {e}')
                        continue
                
                # Handle other dtype conversions
                else:
                    print(f'  → Converting {col} from {current_dtype} to {target_dtype}')
                    try:
                        new_updated_df[col] = new_updated_df[col].astype(target_dtype)
                        print(f'  ✅ Converted successfully')
                    except Exception as e:
                        print(f'  ⚠️ Warning: Could not convert {col} to {target_dtype}: {e}')
                        print(f'     Keeping original dtype: {current_dtype}')
                        continue
            else:
                # Column not in schema - might be a new column
                print(f'ℹ️ Column {col} not in target schema (might be a new field)')
                # Don't raise error, just inform

    except Exception as e:
        print(f'❌ something went wrong in force_update_dtype_xls: {e}')
        import traceback
        traceback.print_exc()
        return df_xls  # Return original on error
        
    return new_updated_df