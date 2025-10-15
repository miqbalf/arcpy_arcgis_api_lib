"""
Utility functions for the ArcGIS API library.
"""

import os
import tempfile
import requests
import logging
from urllib.parse import urlparse
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


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
        filename = os.path.basename(parsed_url.path) or "downloaded_file.csv"
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


def validate_csv_path(csv_path: str) -> tuple[str, Optional[str]]:
    """
    Validate and process a CSV path (URL or local file).
    
    Args:
        csv_path: The CSV path (URL or local file path)
        
    Returns:
        tuple: (actual_csv_path, temp_file_path)
            - actual_csv_path: The path to use for reading the CSV
            - temp_file_path: The temporary file path if downloaded, None otherwise
            
    Raises:
        FileNotFoundError: If local file doesn't exist
        requests.RequestException: If URL download fails
    """
    if is_url(csv_path):
        logger.info("CSV path is a URL, downloading file...")
        temp_file_path = download_file_from_url(csv_path)
        return temp_file_path, temp_file_path
    else:
        # Local file path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return csv_path, None
