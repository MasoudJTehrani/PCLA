import os
import requests
import zipfile
from tqdm import tqdm

def download_and_extract_pretrained_zip(url):
    """
    Downloads a pretrained weights ZIP file from Hugging Face, extracts contents 
    directly into the pcla_agents directory, and deletes the ZIP file.

    Args:
        url (str): The Hugging Face URL of the .zip file to download.
    """
    
    # 1. Define file paths
    # Get the directory where this script is located (pcla_functions)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one directory up (to PCLA root), then into pcla_agents
    target_dir = os.path.join(os.path.dirname(script_dir), "pcla_agents")
    # Save zip file one directory up from script location
    zip_filepath = os.path.join(os.path.dirname(script_dir), "pretrained.zip")
    
    print(f"Download location: {zip_filepath}")
    print(f"Target extraction directory: {target_dir}")

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
                desc="pretrained.zip",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    bar.update(size) # Update the progress bar
                    
        print(f"Successfully downloaded pretrained.zip")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download the file. Check the URL and connection: {e}")
        return

    # --- 3. Extract contents directly into the target directory ---
    try:
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Extract the files directly into target_dir
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            print(f"Extracting contents into {target_dir}/")
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        
    except zipfile.BadZipFile:
        print(f"ERROR: pretrained.zip is not a valid ZIP file.")
        return
        
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return
        
    # --- 4. Delete the ZIP file ---
    try:
        os.remove(zip_filepath)
        print(f"Successfully deleted the ZIP file: pretrained.zip")
    except OSError as e:
        print(f"ERROR: Could not delete file pretrained.zip: {e}")

# URL of the pretrained weights ZIP file from Hugging Face
# TODO: Replace with actual Hugging Face URL
DOWNLOAD_URL = "https://huggingface.co/MasoudJTehrani/PCLA/blob/main/pretrained.zip"

if __name__ == "__main__":
    download_and_extract_pretrained_zip(DOWNLOAD_URL)
