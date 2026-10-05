"""Load project settings for local runs; deployment environment takes priority."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env():
    root = Path(os.environ.get('PDFSCAN_ROOT', Path(__file__).resolve().parents[2]))
    load_dotenv(root / '.env', override=False)
