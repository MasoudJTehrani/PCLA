import os
import requests
import zipfile
from tqdm import tqdm

def download_and_extract_assets_zip(url):
    """
    Downloads the PCLA runtime-assets ZIP (per-town BEV maps, speed limits, town
    pickles, car_data — the large binaries kept OUT of git), and extracts it so
    every file lands back in its original place under the repo.

    The archive stores full repo-relative paths (e.g. pcla_agents/carl/.../Town11.h5),
    so it is extracted into the PCLA root and each file is restored to its folder.
    Mirrors download_weights.download_and_extract_pretrained_zip.

    Args:
        url (str): URL of pcla_assets.zip (a GitHub Release asset for this repo).
    """

    # 1. Paths: extract into the PCLA root (one level up from pcla_functions/).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.dirname(script_dir)                      # PCLA repo root
    zip_filepath = os.path.join(target_dir, "pcla_assets.zip")

    print(f"Download location: {zip_filepath}")
    print(f"Target extraction directory: {target_dir}")
    print(f"Starting download from: {url}")

    # 2. Download the ZIP (streamed, with a progress bar).
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 8192
            with open(zip_filepath, 'wb') as f, tqdm(
                desc="pcla_assets.zip",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    bar.update(size)
        print("Successfully downloaded pcla_assets.zip")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download the file. Check the URL and connection: {e}")
        return

    # 3. Extract into the repo root (paths inside are already repo-relative).
    try:
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            print(f"Extracting contents into {target_dir}/")
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("ERROR: pcla_assets.zip is not a valid ZIP file.")
        return
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return

    # 4. Delete the ZIP.
    try:
        os.remove(zip_filepath)
        print("Successfully deleted the ZIP file: pcla_assets.zip")
    except OSError as e:
        print(f"ERROR: Could not delete file pcla_assets.zip: {e}")


# URL of the runtime-assets ZIP, hosted as a GitHub Release asset for this repo.
DOWNLOAD_URL = "https://github.com/MasoudJTehrani/PCLA/releases/download/assets-v1/pcla_assets.zip"

if __name__ == "__main__":
    download_and_extract_assets_zip(DOWNLOAD_URL)
