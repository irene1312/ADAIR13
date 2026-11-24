"""Allow scripts to run directly on Windows/Linux, including a local dependency target."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / '.local_deps').is_dir():
    sys.path.insert(0, str(ROOT / '.local_deps'))
