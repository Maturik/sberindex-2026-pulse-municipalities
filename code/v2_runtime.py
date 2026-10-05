"""Portable runtime setup. Call before importing numerical/native libraries."""
from pathlib import Path
import os
import sys
import tempfile


def configure(temp_dir, project=None):
    """Use an explicit writable temp path; local runs use only drive G."""
    temp = Path(temp_dir).resolve()
    temp.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(temp)
    tempfile.tempdir = str(temp)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    if project:
        root = Path(project)
        for name in reversed(("v2_packages", "foundation_packages", "pilot_packages",
                              "python_packages", "helper_packages")):
            sys.path.insert(0, str(root / "work" / name))
        os.environ["HF_HOME"] = str(root / "work/hf_cache")
        os.environ["MPLCONFIGDIR"] = str(root / "work/mpl_config")

