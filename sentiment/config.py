from pathlib import Path

RESOURCE_PATH = Path("/Users/vanvt/simple-sentiment/resources")
if not RESOURCE_PATH.exists():
    RESOURCE_PATH.mkdir()