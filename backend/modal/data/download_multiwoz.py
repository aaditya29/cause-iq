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
