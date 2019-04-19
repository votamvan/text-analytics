from pathlib import Path
import os

_path = os.getenv("RESOURCE_PATH", "/Users/vanvt/2019/text-analytics/resources/")
RESOURCE_PATH = Path(_path)