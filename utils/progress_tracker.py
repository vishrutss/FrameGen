# utils/progress_tracker.py
"""Progress tracking and reporting utilities."""

import time
import sys
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and display processing progress."""

    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def finish(self, message: str = "Processing completed"):
        """Mark progress as finished."""
        total_time = time.time() - self.start_time
        final_message = f"{message} (Total time: {self._format_time(total_time)})"

        if self.enable_progress_bar:
            sys.stdout.write(f"\r{final_message}\n")
            sys.stdout.flush()
        else:
            logger.info(final_message)


class BatchProgressTracker:
    """Track progress across multiple batch operations."""

    def __init__(self, total_batches: int, enable_progress_bar: bool = True):
        self.total_batches = total_batches
        self.current_batch = 0
        self.enable_progress_bar = enable_progress_bar
        self.start_time = time.time()
        self.batch_times = []

    def start_batch(self, batch_name: str):
        """Start processing a new batch."""
        self.current_batch += 1
        self.batch_start_time = time.time()

        message = f"Starting batch {self.current_batch}/{self.total_batches}: {batch_name}"
        if self.enable_progress_bar:
            print(message)
        else:
            logger.info(message)

    def finish_batch(self, batch_name: str):
        """Finish current batch."""
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)

        # Calculate average batch time and ETA
        avg_batch_time = sum(self.batch_times) / len(self.batch_times)
        remaining_batches = self.total_batches - self.current_batch
        eta = avg_batch_time * remaining_batches

        progress = self.current_batch / self.total_batches

        message = (
            f"Finished batch {self.current_batch}/{self.total_batches}: {batch_name} "
            f"(took {self._format_time(batch_time)}) | "
            f"Progress: {progress * 100:.1f}% | "
            f"ETA: {self._format_time(eta)}"
        )

        if self.enable_progress_bar:
            print(message)
        else:
            logger.info(message)

    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def get_summary(self) -> dict[str, float]:
        """Get processing summary statistics."""
        total_time = time.time() - self.start_time
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0

        return {
            'total_batches': self.total_batches,
            'total_time': total_time,
            'average_batch_time': avg_batch_time,
            'fastest_batch': min(self.batch_times) if self.batch_times else 0,
            'slowest_batch': max(self.batch_times) if self.batch_times else 0
        }