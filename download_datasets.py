import os
import requests
from datasets import DATASETS


def download_file(url, directory):
    """Download an individual file and save it to a directory."""
    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(directory, local_filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def download_dataset(dataset_name, dataset_info):
    """Download all files for a given dataset."""
    data_dir = dataset_info['data_dir']
    file_names = dataset_info.get('files', [])
    download_dir = os.path.join('data', dataset_name)
    os.makedirs(download_dir, exist_ok=True)

    for file_name in file_names:
        file_url = data_dir + file_name
        download_file(file_url, download_dir)
        print(f"Downloaded {file_name} to {dataset_name}/")

# Download each dataset
for dataset_name, dataset_info in DATASETS.items():
    print(f"Downloading {dataset_name} dataset...")
    download_dataset(dataset_name, dataset_info)
    print(f"Finished downloading {dataset_name}.\n")

print("All datasets downloaded.")
