import os
import requests
import zipfile
from tqdm import tqdm

def download_and_extract_pretrained_zip(url, target_dir="agents", zip_filename="pretrained.zip"):
    """
    Downloads a ZIP file from a URL, extracts it into a target directory, and deletes the ZIP.

    Args:
        url (str): The URL of the .zip file to download.
        target_dir (str): The directory where contents will be extracted (default: "agents").
        zip_filename (str): The name to save the downloaded zip file as.
    """
    
    # 1. Define file paths
    current_dir = os.getcwd()
    zip_filepath = os.path.join(current_dir, zip_filename)
    extract_dir = os.path.join(current_dir, target_dir)

    print(f"Starting download from: {url}")
    
    # --- 2. Download the ZIP file ---
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise exception for bad status codes
            
            # Get the total size of the file from the headers
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 8192
            
            # Initialize tqdm with total size and description
            with open(zip_filepath, 'wb') as f, tqdm(
                desc=zip_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    bar.update(size) # Update the progress bar
                    
        print(f"Successfully downloaded {zip_filename}.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download the file. Check the URL and connection: {e}")
        return

    # --- 3. Extract contents into the target directory ---
    try:
        # Extract the files
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            print(f"Extracting contents into {target_dir}/...")
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
        
    except zipfile.BadZipFile:
        print(f"ERROR: {zip_filename} is not a valid ZIP file.")
        
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        
    # --- 4. Delete the ZIP file ---
    try:
        os.remove(zip_filepath)
        print(f"Successfully deleted the ZIP file: {zip_filename}")
    except OSError as e:
        print(f"ERROR: Could not delete file {zip_filename}: {e}")

# URL of the pretrained weights ZIP file
DOWNLOAD_URL = "https://zenodo.org/records/17399201/files/pretrained.zip?download=1" 

if __name__ == "__main__":
    download_and_extract_pretrained_zip(DOWNLOAD_URL)
