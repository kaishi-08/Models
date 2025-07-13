# scripts/download_crossdock.py
import os
import wget
import tarfile
import zipfile
from pathlib import Path
import argparse

def download_crossdock_data(data_dir: str = "data/raw/"):
    """Download CrossDock2020 dataset"""
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # CrossDock2020 URLs (these are example URLs - adjust based on actual source)
    urls = {
        "crossdock2020-v1.1-other-pl.tar.gz": "https://bits.csb.pitt.edu/files/crossdock2020/crossdock2020-v1.1-other-pl.tar.gz",
        "index.pkl": "https://bits.csb.pitt.edu/files/crossdock2020/index.pkl"
    }
    
    crossdock_dir = data_path / "crossdock2020"
    crossdock_dir.mkdir(exist_ok=True)
    
    print("Downloading CrossDock2020 dataset...")
    
    for filename, url in urls.items():
        file_path = crossdock_dir / filename
        
        if file_path.exists():
            print(f"{filename} already exists, skipping...")
            continue
        
        print(f"Downloading {filename}...")
        try:
            wget.download(url, str(file_path))
            print(f"\nDownloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            continue
    
    # Extract tar files
    for file_path in crossdock_dir.glob("*.tar.gz"):
        print(f"Extracting {file_path.name}...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(crossdock_dir)
        print(f"Extracted {file_path.name}")
    
    print("CrossDock2020 dataset download completed!")
    return str(crossdock_dir)

def setup_directory_structure():
    """Setup project directory structure"""
    directories = [
        "data/raw/crossdock2020",
        "data/processed",
        "models",
        "logs",
        "generated_molecules",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description='Download and setup CrossDock dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/', 
                       help='Directory to download data')
    parser.add_argument('--setup_dirs', action='store_true',
                       help='Setup project directory structure')
    
    args = parser.parse_args()
    
    if args.setup_dirs:
        setup_directory_structure()
    
    download_crossdock_data(args.data_dir)

if __name__ == "__main__":
    main()