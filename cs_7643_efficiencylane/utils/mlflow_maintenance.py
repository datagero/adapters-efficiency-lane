"""
Simple functions to maintain repo cleanliness by deleting old MLflow logs.
"""

import os

def list_folders(directory):
    """
    List all the folders in the specified directory.
    """
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folders.append(entry.name)
    return folders

def find_mlflow_ids(directory):
    """
    Find mlflow_ids.txt files in the specified directory and its subdirectories.
    """
    mlflow_ids = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "mlflow_id.txt":
                mlflow_ids.append(root)
    return mlflow_ids

def main():
    directory = "mlruns/0"
    folders = list_folders(directory)
    mlflow_ids = find_mlflow_ids("training_output")

    print("Folders in mlruns/0:")
    print(folders)
    print("\nFolders with mlflow_id.txt files:")
    print(mlflow_ids)

    keep_mlflow_ids = []
    # For every folder in trainer results with mlflow_id.txt
    for folder in mlflow_ids:
        with open(os.path.join(folder, "mlflow_id.txt"), "r") as file:
            keep_mlflow_ids.append(file.read())

    # Delete all other mlflow logs
    delete_mlflow_ids = [folder for folder in folders if folder not in keep_mlflow_ids]
    for id in delete_mlflow_ids:
        os.system(f"rm -r {os.path.join(directory, id)}")

if __name__ == "__main__":
    main()
