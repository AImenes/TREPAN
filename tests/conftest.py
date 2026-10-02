import sys
from pathlib import Path

# Make the package importable when the tests run from a plain checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
