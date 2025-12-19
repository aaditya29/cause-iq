import os
import json
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Dict, Any


class MultiWOZDownloader:
    MULTIWOZ_URL = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2.zip"

    def __init__(self, data_dir: str = "../data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, destination: Path) -> None:
        logger.info(f"Downloading MultiWOZ dataset from {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded to {destination}")

    def extract_zip(self, zip_path: Path, extract_dir: Path) -> None:
        logger.info(f"Extracting {zip_path}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        logger.info(f"Extracted to {extract_dir}")

        # Handle nested "data/" directory structure
        data_subdir = extract_dir / "data"
        if data_subdir.exists():
            logger.info("Restructuring nested 'data/' directory...")
            for item in data_subdir.iterdir():
                target = extract_dir / item.name
                if target.exists():
                    logger.warning(f"Skipping {item.name} - already exists")
                else:
                    item.rename(target)
            data_subdir.rmdir()
            logger.info(" Restructured to flat directory")

    def download_multiwoz(self) -> Path:
        zip_path = self.data_dir / "MultiWOZ_2.2.zip"
        extract_dir = self.data_dir / "multiwoz_2.2"

        # Check if already downloaded and valid
        if extract_dir.exists() and self.is_valid_dataset(extract_dir):
            logger.info(f"MultiWOZ dataset already exists at {extract_dir}")
            return extract_dir

        # Download
        if not zip_path.exists():
            try:
                self.download_file(self.MULTIWOZ_URL, zip_path)
            except Exception as e:
                logger.error(f"Failed to download from official URL: {e}")
                logger.info("Attempting alternative download method...")
                self.download_from_huggingface()
                return extract_dir

        # Extract
        self.extract_zip(zip_path, extract_dir)

        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Cleaned up zip file")

        return extract_dir

    def is_valid_dataset(self, dataset_dir: Path) -> bool:
        expected_dirs = ["train", "dev", "test"]
        for split_dir in expected_dirs:
            if not (dataset_dir / split_dir).exists():
                return False
        return True

    def download_from_huggingface(self) -> None:
        try:
            from datasets import load_dataset

            logger.info("Downloading MultiWOZ from HuggingFace datasets...")

            dataset = load_dataset("multi_woz_v22")

            output_dir = self.data_dir / "multiwoz_2.2"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Map validation -> dev
            split_mapping = {
                'train': 'train',
                'validation': 'dev',
                'test': 'test'
            }

            for hf_split, local_split in split_mapping.items():
                split_data = dataset[hf_split]

                # Create split directory
                split_dir = output_dir / local_split
                split_dir.mkdir(exist_ok=True)

                # Convert to list of dicts
                conversations = []
                for item in split_data:
                    conversations.append(dict(item))

                # Save as dialogues_001.json
                output_file = split_dir / "dialogues_001.json"
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
        """Verify downloaded dataset structure and contents"""
        logger.info("Verifying dataset...")

        stats = {
            "total_conversations": 0,
            "total_dialogue_files": 0,
            "splits_found": [],
            "has_dialog_acts": False,
            "has_schema": False,
            "sample_services": set()
        }

        # Check for expected splits
        expected_splits = ["train", "dev", "test"]
        for split in expected_splits:
            split_dir = dataset_dir / split
            if split_dir.exists() and split_dir.is_dir():
                stats["splits_found"].append(split)

                # Count dialogue files
                dialogue_files = list(split_dir.glob("dialogues_*.json"))
                stats["total_dialogue_files"] += len(dialogue_files)

                # Sample first file to count conversations
                if dialogue_files:
                    try:
                        with open(dialogue_files[0], 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                stats["total_conversations"] += len(data)
                                # Sample services from first conversation
                                if data and "services" in data[0]:
                                    stats["sample_services"].update(
                                        data[0]["services"])
                    except (json.JSONDecodeError, IOError, KeyError) as e:
                        logger.warning(
                            f"Could not read {dialogue_files[0]}: {e}")

                logger.info(
                    f"Found {len(dialogue_files)} dialogue file(s) in {split}/")

        # Check for additional files
        stats["has_dialog_acts"] = (dataset_dir / "dialog_acts.json").exists()
        stats["has_schema"] = (dataset_dir / "schema.json").exists()

        # Convert set to list for JSON serialization
        stats["sample_services"] = list(stats["sample_services"])

        logger.info(f"Dataset verification complete:")
        logger.info(f"  Splits found: {stats['splits_found']}")
        logger.info(f"  Total dialogue files: {stats['total_dialogue_files']}")
        logger.info(
            f"  Sample conversation count: {stats['total_conversations']}")
        logger.info(f"  Has dialog_acts.json: {stats['has_dialog_acts']}")
        logger.info(f"  Has schema.json: {stats['has_schema']}")
        logger.info(f"  Sample services: {stats['sample_services']}")

        return stats


def main():
    logger.info("MultiWOZ 2.2 Dataset Downloader")

    downloader = MultiWOZDownloader()

    # Download dataset
    dataset_dir = downloader.download_multiwoz()

    # Verify dataset
    stats = downloader.verify_dataset(dataset_dir)
    logger.info(" MultiWOZ dataset ready!")
    logger.info(f"Dataset location: {dataset_dir}")


if __name__ == "__main__":
    main()
