import sys
import uuid
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.local_deps'))


@pytest.fixture(scope='session')
def retained_workspace():
    """Do not use pytest temp cleanup: the user's no-deletion rule includes fixtures."""
    p = ROOT/'local_validation'/('pytest_'+uuid.uuid4().hex[:8])
    p.mkdir(parents=True, exist_ok=False)
    return p


@pytest.fixture(scope='session')
def synthetic(retained_workspace):
    from isar.synthetic import create_synthetic
    return create_synthetic(retained_workspace)
