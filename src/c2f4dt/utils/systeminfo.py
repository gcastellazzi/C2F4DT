from __future__ import annotations

import shutil

def disk_usage_percent(path: str = "/") -> tuple[float, float, float]:
    """Return disk usage as (used_GB, free_GB, used_percent).

    Args:
        path: Filesystem path to query. Defaults to root.

    Returns:
        Tuple containing used space (GB), free space (GB), and used percentage.
    """
    total, used, free = shutil.disk_usage(path)
    to_gb = lambda b: b / (1024**3)
    percent = (used / total) * 100 if total else 0.0
    return to_gb(used), to_gb(free), percent
