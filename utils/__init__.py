# utils/__init__.py
"""Utility modules for FrameGen."""
import sys
from datetime import time
from typing import Optional

from .config_loader import load_config, validate_config, create_paths_from_config
from .progress_tracker import ProgressTracker, BatchProgressTracker, logger

__all__ = ['load_config', 'validate_config', 'create_paths_from_config',
           'ProgressTracker', 'BatchProgressTracker']
def __init__(self, enable_progress_bar: bool = True):
    self.enable_progress_bar = enable_progress_bar
    self.start_time = time()
    self.last_update = 0
    self.last_progress = 0


def update(self, progress: float, current: int, total: int,
           message: Optional[str] = None):
    """Update progress display."""
    current_time = time()

    # Throttle updates (max once per second)
    if current_time - self.last_update < 1.0 and progress < 1.0:
        return

    self.last_update = current_time
    elapsed = current_time - self.start_time

    # Calculate ETA
    if progress > 0:
        eta = (elapsed / progress) * (1 - progress)
        eta_str = self._format_time(eta)
    else:
        eta_str = "Unknown"

    # Calculate processing speed
    if elapsed > 0:
        items_per_sec = current / elapsed
        speed_str = f"{items_per_sec:.1f} items/s"
    else:
        speed_str = "Calculating..."

    # Format progress message
    progress_msg = (
        f"Progress: {progress * 100:.1f}% ({current}/{total}) | "
        f"Elapsed: {self._format_time(elapsed)} | "
        f"ETA: {eta_str} | "
        f"Speed: {speed_str}"
    )

    if message:
        progress_msg += f" | {message}"

    if self.enable_progress_bar:
        # Clear line and print progress
        sys.stdout.write(f"\r{progress_msg}")
        sys.stdout.flush()

        if progress >= 1.0:
            sys.stdout.write("\n")
    else:
        # Log progress at intervals
        progress_percent = int(progress * 100)
        last_percent = int(self.last_progress * 100)

        if progress_percent >= last_percent + 10 or progress >= 1.0:
            logger.info(progress_msg)

    self.last_progress = progress
