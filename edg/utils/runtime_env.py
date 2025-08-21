"""Runtime environment helpers for smoother execution on clusters/NFS.

This module centralizes small environment tweaks to avoid common issues like:
- NFS '.nfsXXXX' temp-file cleanup errors from multiprocessing/DataLoader
- Matplotlib writing config under slow or read-only home directories

Call `configure_local_tmpdir()` as early as possible in the process (before
modules that may first call `tempfile.gettempdir()` are imported) so that
Python and child processes use a local, non-NFS temporary directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".__edg_write_test__"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def configure_local_tmpdir() -> Path:
    """Ensure TMPDIR points to a local, non-NFS path to avoid '.nfs*' errors.

    Preference order:
    1) EDG_LOCAL_TMPDIR if set
    2) /dev/shm/edg-${USER} (tmpfs, if available)
    3) /tmp/edg-${USER}

    Also sets TEMP and TMP for broader compatibility and points Matplotlib's
    config directory to a subfolder to avoid home-directory writes.
    """
    user = os.environ.get("USER") or str(os.getuid())

    candidates = []
    if os.environ.get("EDG_LOCAL_TMPDIR"):
        candidates.append(Path(os.environ["EDG_LOCAL_TMPDIR"]))

    # Prefer tmpfs if present
    candidates.append(Path(f"/dev/shm/edg-{user}"))
    # Fallback to local /tmp
    candidates.append(Path(f"/tmp/edg-{user}"))

    chosen: Path | None = None
    for c in candidates:
        if _is_writable_dir(c):
            chosen = c
            break

    if chosen is None:
        # Last resort: keep current default tempdir
        return Path(tempfile.gettempdir())

    # Restrict permissions a bit (ignore failures)
    try:
        os.chmod(chosen, 0o700)
    except Exception:
        pass

    # Set all common temp env vars
    os.environ.setdefault("TMPDIR", str(chosen))
    os.environ.setdefault("TEMP", str(chosen))
    os.environ.setdefault("TMP", str(chosen))

    # Update tempfile module cache in case it was not computed yet
    tempfile.tempdir = str(chosen)

    # Hint multiprocessing (best-effort; API varies across Python versions)
    try:
        import multiprocessing.util as mp_util  # type: ignore

        # Some CPython versions expose a setter; guard just in case
        if hasattr(mp_util, "set_temp_dir"):
            mp_util.set_temp_dir(str(chosen))  # type: ignore[attr-defined]
        elif hasattr(mp_util, "_set_temp_dir"):
            mp_util._set_temp_dir(str(chosen))  # type: ignore[attr-defined]
    except Exception:
        # Non-fatal; environment vars usually suffice
        pass

    # Direct Matplotlib config/cache to local tmp to avoid HOME writes on NFS
    mpl_cfg = chosen / "mplconfig"
    if _is_writable_dir(mpl_cfg):
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

    return chosen
