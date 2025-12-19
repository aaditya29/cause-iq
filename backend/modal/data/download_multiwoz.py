import os
import json
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Dict, Any


# class for multiwoz data downloader
class MultiWOZDownloader:
    MULTIWOZ_URL = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2.zip"

    # function for initializing the downloader
    def __init__(self, data_dir: str = "../data/raw"):
        self.data_dir = Path(data_dir)
        # ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, destination: Path) -> None:
        logger.info(f"Downloading MultiWOZ dataset from {url}...")
        # Download with streaming i.e. in chunks
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        # Get total size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f:
            with tqdm(total_=total_size, unit='B', unit_scale=True) as pbar:
                # Download in 8KB chunks
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        logger.info(f"Downloaded to {destination}")

        def extract_zip(self, zip_path: Path, extract_dir: Path) -> None:
            logger.info(f"Extracting {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to specified directory
                zip_ref.extractall(extract_dir)

            # Download MultiWOZ dataset
            logger.info(f"Extracted to {extract_dir}")

    def download_multiwoz(self) -> Path:
        """Download MultiWOZ 2.2 dataset"""
        zip_path = self.data_dir / "MultiWOZ_2.2.zip"
        extract_dir = self.data_dir / "multiwoz_2.2"

        # Check if already downloaded
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"MultiWOZ dataset already exists at {extract_dir}")
            return extract_dir

        # Download
        if not zip_path.exists():
            try:
                self.download_file(self.MULTIWOZ_URL, zip_path)
            except Exception as e:
                logger.error(f"Failed to download from official URL: {e}")
                logger.info("Attempting alternative download method...")
                # Alternative: download from HuggingFace datasets
                self.download_from_huggingface()
                return extract_dir

        # Extract
        self.extract_zip(zip_path, extract_dir)

        # Clean up zip
        zip_path.unlink()

        return extract_dir

    def download_from_huggingface(self) -> None:
        try:
            from datasets import load_dataset

            logger.info("Downloading MultiWOZ from HuggingFace datasets...")

            # Load MultiWOZ dataset
            dataset = load_dataset("multi_woz_v22")

            # Save to local directory
            output_dir = self.data_dir / "multiwoz_2.2"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save train/val/test splits as JSON
            for split in ['train', 'validation', 'test']:
                split_data = dataset[split]

                # Convert to list of dicts
                conversations = []
                for item in split_data:
                    conversations.append(dict(item))

                # Save as JSON
                output_file = output_dir / f"{split}.json"
                with open(output_file, 'w') as f:
                    json.dump(conversations, f, indent=2)

                logger.info(
                    f"Saved {len(conversations)} conversations to {output_file}")

            logger.info(
                "MultiWOZ dataset downloaded successfully from HuggingFace")

        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            raise

    def verify_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        logger.info("Verifying dataset...")

        stats = {
            "total_conversations": 0,
            "total_turns": 0,
            "domains": set(),
            "splits": {}
        }

        # Check for common file patterns
        json_files = list(dataset_dir.glob("**/*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {dataset_dir}")
            return stats

        logger.info(f"Found {len(json_files)} JSON files")

        # Try to load and count
        for json_file in json_files[:5]:  # Sample first 5 files
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    stats["total_conversations"] += len(data)
                    if data:
                        # Count turns in first conversation
                        if "turns" in data[0]:
                            stats["total_turns"] += sum(len(c.get("turns", []))
                                                        for c in data)
                        elif "dialogue" in data[0]:
                            stats["total_turns"] += sum(len(c.get("dialogue", []))
                                                        for c in data)

            except Exception as e:
                logger.warning(f"Could not parse {json_file}: {e}")
                continue

        logger.info(f"Dataset stats: {stats}")
        return stats


def main():
    downloader = MultiWOZDownloader()

    # Download dataset
    dataset_dir = downloader.download_multiwoz()

    # Verify dataset
    stats = downloader.verify_dataset(dataset_dir)

    logger.info("✓ MultiWOZ dataset ready")
    logger.info(f"Dataset location: {dataset_dir}")
    logger.info(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
