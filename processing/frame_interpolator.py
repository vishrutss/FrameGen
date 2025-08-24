import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import yaml

from models.rife_model import RIFEModel
from .video_io import VideoReader, VideoWriter, VideoInfo

logger = logging.getLogger(__name__)


@dataclass
class InterpolationConfig:
    """Configuration for frame interpolation."""
    target_fps: float = 60.0
    interpolation_factor: int = 2  # 2x, 4x, etc.
    batch_size: int = 4
    max_resolution: Tuple[int, int] = (1920, 1080)
    model_path: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    quality: int = 23  # Video quality (lower = better)
    temporal_consistency: bool = True
    memory_efficient: bool = True


class FrameInterpolator:
    """Main frame interpolation engine using RIFE model."""

    def __init__(self, config: InterpolationConfig):
        self.config = config
        self.device = config.device

        # Initialize model
        self.model = RIFEModel(device=self.device)
        if config.model_path and Path(config.model_path).exists():
            self.model.load_pretrained(config.model_path)
        else:
            logger.warning("No pretrained model loaded. Using random weights.")

        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'processing_time': 0.0,
            'average_fps': 0.0
        }

        logger.info(f"Frame interpolator initialized on {self.device}")

    def interpolate_video(self, input_path: str, output_path: str,
                          progress_callback: Optional[Callable] = None) -> dict:
        """
        Interpolate frames in a video to increase frame rate.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()

        try:
            # Load video
            reader = VideoReader(input_path,
                                 target_size=self._get_optimal_resolution(input_path))

            # Calculate output fps
            output_fps = self._calculate_output_fps(reader.info.fps)

            # Create writer
            with VideoWriter(output_path, output_fps,
                             reader.info.width, reader.info.height,
                             quality=self.config.quality) as writer:

                if self.config.memory_efficient:
                    self._interpolate_streaming(reader, writer, progress_callback)
                else:
                    self._interpolate_batch(reader, writer, progress_callback)

            # Calculate final stats
            total_time = time.time() - start_time
            self.stats['processing_time'] = total_time
            self.stats['average_fps'] = self.stats['frames_processed'] / total_time

            logger.info(f"Interpolation completed: {output_path}")
            logger.info(f"Processed {self.stats['frames_processed']} frames in {total_time:.2f}s")
            logger.info(f"Average processing FPS: {self.stats['average_fps']:.2f}")

            return self.stats.copy()

        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            raise

    def _interpolate_streaming(self, reader: VideoReader, writer: VideoWriter,
                               progress_callback: Optional[Callable] = None):
        """Memory-efficient streaming interpolation."""
        frame_pairs = list(reader.read_frame_pairs())
        total_pairs = len(frame_pairs)

        for i, (frame1, frame2) in enumerate(frame_pairs):
            # Write original frame1
            writer.write_frame(frame1)
            self.stats['frames_processed'] += 1

            # Generate interpolated frames
            interpolated_frames = self._generate_interpolated_frames(frame1, frame2)

            for interp_frame in interpolated_frames:
                writer.write_frame(interp_frame)
                self.stats['frames_processed'] += 1

            # Progress callback
            if progress_callback:
                progress = (i + 1) / total_pairs
                progress_callback(progress, i + 1, total_pairs)

            # Memory cleanup
            del frame1, frame2, interpolated_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Write final frame
        if frame_pairs:
            writer.write_frame(frame_pairs[-1][1])
            self.stats['frames_processed'] += 1

    def _interpolate_batch(self, reader: VideoReader, writer: VideoWriter,
                           progress_callback: Optional[Callable] = None):
        """Batch processing for faster interpolation (more memory intensive)."""
        frames = list(reader.read_frames())
        total_frames = len(frames)

        # Process in batches
        for i in range(0, total_frames - 1, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, total_frames - 1)
            batch_frames1 = frames[i:batch_end]
            batch_frames2 = frames[i + 1:batch_end + 1]

            # Stack for batch processing
            frames1_batch = torch.cat(batch_frames1, dim=0)
            frames2_batch = torch.cat(batch_frames2, dim=0)

            # Generate interpolated frames
            with torch.no_grad():
                interpolated_batch = self._generate_interpolated_batch(frames1_batch, frames2_batch)

            # Write frames
            for j, (f1, interp, f2) in enumerate(zip(batch_frames1, interpolated_batch, batch_frames2)):
                writer.write_frame(f1)
                for interp_frame in interp:
                    writer.write_frame(interp_frame)
                self.stats['frames_processed'] += len(interp) + 1

            # Progress callback
            if progress_callback:
                progress = (batch_end) / (total_frames - 1)
                progress_callback(progress, batch_end, total_frames - 1)

        # Write final frame
        if frames:
            writer.write_frame(frames[-1])
            self.stats['frames_processed'] += 1

    def _generate_interpolated_frames(self, frame1: torch.Tensor,
                                      frame2: torch.Tensor) -> List[torch.Tensor]:
        """Generate interpolated frames between two frames."""
        interpolated = []

        # Calculate timesteps based on interpolation factor
        timesteps = np.linspace(0, 1, self.config.interpolation_factor + 2)[1:-1]

        with torch.no_grad():
            self.model.eval()
            for t in timesteps:
                interp_frame = self.model.forward(frame1, frame2, float(t))

                # Apply temporal consistency if enabled
                if self.config.temporal_consistency and len(interpolated) > 0:
                    interp_frame = self._apply_temporal_smoothing(
                        interp_frame, interpolated[-1], alpha=0.1
                    )

                interpolated.append(interp_frame)

        return interpolated

    def _generate_interpolated_batch(self, frames1_batch: torch.Tensor,
                                     frames2_batch: torch.Tensor) -> List[List[torch.Tensor]]:
        """Generate interpolated frames for a batch of frame pairs."""
        batch_size = frames1_batch.shape[0]
        timesteps = np.linspace(0, 1, self.config.interpolation_factor + 2)[1:-1]

        batch_results = [[] for _ in range(batch_size)]

        with torch.no_grad():
            self.model.eval()
            for t in timesteps:
                # Process entire batch at once
                interp_batch = self.model.interpolate_batch(frames1_batch, frames2_batch, t)

                # Split batch results
                for i in range(batch_size):
                    frame = interp_batch[i:i + 1]

                    # Apply temporal consistency
                    if self.config.temporal_consistency and len(batch_results[i]) > 0:
                        frame = self._apply_temporal_smoothing(
                            frame, batch_results[i][-1], alpha=0.1
                        )

                    batch_results[i].append(frame)

        return batch_results

    def _apply_temporal_smoothing(self, current_frame: torch.Tensor,
                                  previous_frame: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """Apply temporal smoothing to reduce flickering."""
        return alpha * previous_frame + (1 - alpha) * current_frame

    def _calculate_output_fps(self, input_fps: float) -> float:
        """Calculate output FPS based on interpolation factor."""
        if self.config.target_fps:
            return self.config.target_fps
        return input_fps * self.config.interpolation_factor

    def _get_optimal_resolution(self, video_path: str) -> Optional[Tuple[int, int]]:
        """Determine optimal processing resolution."""
        # Quick probe to get video dimensions
        temp_reader = VideoReader(video_path)
        width, height = temp_reader.info.width, temp_reader.info.height
        temp_reader.__del__()

        max_w, max_h = self.config.max_resolution

        if width <= max_w and height <= max_h:
            return None  # Use original resolution

        # Scale down maintaining aspect ratio
        scale = min(max_w / width, max_h / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Ensure dimensions are divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        logger.info(f"Scaling resolution: {width}x{height} -> {new_width}x{new_height}")
        return new_width, new_height

    def benchmark(self, test_video_path: str, num_frames: int = 100) -> dict:
        """Benchmark interpolation performance."""
        logger.info(f"Starting benchmark with {num_frames} frames")

        reader = VideoReader(test_video_path)
        frames = list(reader.read_frames(0, min(num_frames, len(reader))))

        if len(frames) < 2:
            raise ValueError("Not enough frames for benchmarking")

        # Warm-up
        _ = self._generate_interpolated_frames(frames[0], frames[1])

        # Benchmark
        start_time = time.time()
        total_interpolated = 0

        for i in range(len(frames) - 1):
            interpolated = self._generate_interpolated_frames(frames[i], frames[i + 1])
            total_interpolated += len(interpolated)

        end_time = time.time()

        benchmark_stats = {
            'input_frames': len(frames) - 1,
            'interpolated_frames': total_interpolated,
            'total_time': end_time - start_time,
            'fps': total_interpolated / (end_time - start_time),
            'device': str(self.device),
            'model_memory': self._get_model_memory_usage()
        }

        logger.info(f"Benchmark completed: {benchmark_stats}")
        return benchmark_stats

    def _get_model_memory_usage(self) -> dict:
        """Get model memory usage statistics."""
        if torch.cuda.is_available() and self.device != 'cpu':
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 ** 2,
                'cached_mb': torch.cuda.memory_reserved() / 1024 ** 2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2
            }
        return {'cpu_model': 'Memory usage not tracked on CPU'}

    def save_config(self, config_path: str):
        """Save current configuration to file."""
        config_dict = {
            'target_fps': self.config.target_fps,
            'interpolation_factor': self.config.interpolation_factor,
            'batch_size': self.config.batch_size,
            'max_resolution': list(self.config.max_resolution),
            'model_path': self.config.model_path,
            'device': self.config.device,
            'quality': self.config.quality,
            'temporal_consistency': self.config.temporal_consistency,
            'memory_efficient': self.config.memory_efficient
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Configuration saved to {config_path}")

    @classmethod
    def load_config(cls, config_path: str) -> 'FrameInterpolator':
        """Load configuration from file and create interpolator."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert dict to dataclass
        config = InterpolationConfig(**config_dict)
        return cls(config)


class MultiScaleInterpolator(FrameInterpolator):
    """Advanced interpolator with multi-scale processing for better quality."""

    def __init__(self, config: InterpolationConfig, scales: List[float] = [0.5, 1.0]):
        super().__init__(config)
        self.scales = scales
        logger.info(f"Multi-scale interpolator with scales: {scales}")

    def _generate_interpolated_frames(self, frame1: torch.Tensor,
                                      frame2: torch.Tensor) -> List[torch.Tensor]:
        """Generate interpolated frames using multi-scale approach."""
        interpolated = []
        timesteps = np.linspace(0, 1, self.config.interpolation_factor + 2)[1:-1]

        with torch.no_grad():
            self.model.eval()

            for t in timesteps:
                # Multi-scale processing
                scale_results = []

                for scale in self.scales:
                    if scale != 1.0:
                        # Resize frames for this scale
                        h, w = frame1.shape[-2:]
                        new_h, new_w = int(h * scale), int(w * scale)

                        frame1_scaled = torch.nn.functional.interpolate(
                            frame1, size=(new_h, new_w), mode='bilinear', align_corners=False
                        )
                        frame2_scaled = torch.nn.functional.interpolate(
                            frame2, size=(new_h, new_w), mode='bilinear', align_corners=False
                        )

                        # Interpolate at this scale
                        interp_scaled = self.model.forward(frame1_scaled, frame2_scaled, float(t))

                        # Resize back to original resolution
                        interp_scaled = torch.nn.functional.interpolate(
                            interp_scaled, size=(h, w), mode='bilinear', align_corners=False
                        )

                        scale_results.append(interp_scaled)
                    else:
                        # Full resolution
                        interp_full = self.model.forward(frame1, frame2, float(t))
                        scale_results.append(interp_full)

                # Combine multi-scale results (weighted average)
                if len(scale_results) > 1:
                    weights = [0.3, 0.7]  # Give more weight to full resolution
                    combined = sum(w * result for w, result in zip(weights, scale_results))
                    interpolated.append(combined)
                else:
                    interpolated.append(scale_results[0])

        return interpolated


def create_interpolation_pipeline(config_path: str) -> FrameInterpolator:
    """Factory function to create interpolation pipeline from config."""
    if Path(config_path).exists():
        return FrameInterpolator.load_config(config_path)
    else:
        # Create default config
        config = InterpolationConfig()
        interpolator = FrameInterpolator(config)
        interpolator.save_config(config_path)
        return interpolator