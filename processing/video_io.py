import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Generator, Tuple, Optional, List, Union
import logging
from dataclasses import dataclass
import ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata container."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    format: str


class VideoReader:
    """Efficient video reader with frame buffering and preprocessing."""

    def __init__(self, video_path: str, target_size: Optional[Tuple[int, int]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.video_path = Path(video_path)
        self.target_size = target_size
        self.device = device

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video info
        self.info = self._get_video_info()
        logger.info(f"Loaded video: {self.info}")

    def _get_video_info(self) -> VideoInfo:
        """Extract video metadata."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Get codec info using ffprobe
        try:
            probe = ffmpeg.probe(str(self.video_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            codec = video_stream.get('codec_name', 'unknown')
            format_name = probe['format'].get('format_name', 'unknown')
        except:
            codec = 'unknown'
            format_name = 'unknown'

        return VideoInfo(width, height, fps, frame_count, duration, codec, format_name)

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.

        Args:
            frame: BGR frame from OpenCV [H, W, 3]

        Returns:
            Preprocessed tensor [1, 3, H, W] in range [0, 1]
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and convert to tensor
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_tensor(frame).permute(2, 0, 1).unsqueeze(0)

        return frame_tensor.to(self.device)

    def read_frames(self, start_frame: int = 0, num_frames: Optional[int] = None) -> Generator[
        torch.Tensor, None, None]:
        """
        Read frames as preprocessed tensors.

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames to read (None for all)

        Yields:
            Preprocessed frame tensors [1, 3, H, W]
        """
        # Seek to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_read = 0
        while True:
            if num_frames and frames_read >= num_frames:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            yield self.preprocess_frame(frame)
            frames_read += 1

    def read_frame_pairs(self, start_frame: int = 0, step: int = 1) -> Generator[
        Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Read consecutive frame pairs for interpolation.

        Args:
            start_frame: Starting frame index
            step: Step between frame pairs

        Yields:
            Tuple of (frame1, frame2) tensors
        """
        frames = list(self.read_frames(start_frame))

        for i in range(0, len(frames) - step, step):
            yield frames[i], frames[i + step]

    def get_frame_at(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Get specific frame by index."""
        if frame_idx >= self.info.frame_count or frame_idx < 0:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            return self.preprocess_frame(frame)
        return None

    def __len__(self) -> int:
        return self.info.frame_count

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Efficient video writer with batch processing support."""

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 codec: str = 'mp4v', quality: int = 23):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.quality = quality

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize OpenCV writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer for: {output_path}")

        self.frame_count = 0
        logger.info(f"Video writer initialized: {output_path} ({width}x{height} @ {fps}fps)")

    def postprocess_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output tensor to OpenCV frame.

        Args:
            frame_tensor: Model output [1, 3, H, W] or [3, H, W] in range [0, 1]

        Returns:
            BGR frame for OpenCV [H, W, 3] in range [0, 255]
        """
        # Handle batch dimension
        if frame_tensor.dim() == 4:
            frame_tensor = frame_tensor.squeeze(0)

        # Move to CPU and convert to numpy
        frame = frame_tensor.detach().cpu().numpy()

        # Permute from CHW to HWC
        frame = np.transpose(frame, (1, 2, 0))

        # Clamp values and convert to uint8
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize if needed
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def write_frame(self, frame: Union[torch.Tensor, np.ndarray]):
        """Write a single frame."""
        if isinstance(frame, torch.Tensor):
            frame = self.postprocess_frame(frame)
        elif isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        self.writer.write(frame)
        self.frame_count += 1

    def write_frames(self, frames: List[Union[torch.Tensor, np.ndarray]]):
        """Write multiple frames efficiently."""
        for frame in frames:
            self.write_frame(frame)

    def write_from_generator(self, frame_generator: Generator):
        """Write frames from a generator."""
        for frame in frame_generator:
            self.write_frame(frame)

    def finalize(self):
        """Finalize video writing and cleanup."""
        if hasattr(self, 'writer') and self.writer:
            self.writer.release()
            logger.info(f"Video written: {self.output_path} ({self.frame_count} frames)")

            # Optional: Use ffmpeg for better compression
            if self.codec == 'mp4v':
                self._compress_with_ffmpeg()

    def _compress_with_ffmpeg(self):
        """Compress video using ffmpeg for better quality/size ratio."""
        try:
            temp_path = self.output_path.with_suffix('.temp.mp4')

            (
                ffmpeg
                .input(str(self.output_path))
                .output(str(temp_path),
                        vcodec='libx264',
                        crf=self.quality,
                        preset='medium',
                        pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Replace original with compressed version
            self.output_path.unlink()
            temp_path.rename(self.output_path)
            logger.info(f"Video compressed with ffmpeg: {self.output_path}")

        except Exception as e:
            logger.warning(f"FFmpeg compression failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def __del__(self):
        self.finalize()


class BatchVideoProcessor:
    """Process multiple videos or large videos in batches."""

    def __init__(self, batch_size: int = 8, max_resolution: Tuple[int, int] = (1920, 1080)):
        self.batch_size = batch_size
        self.max_resolution = max_resolution

    def process_video_list(self, video_paths: List[str], output_dir: str,
                           processing_func, **kwargs):
        """Process multiple videos in batches."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in video_paths:
            try:
                input_path = Path(video_path)
                output_path = output_dir / f"processed_{input_path.name}"

                self.process_single_video(
                    str(input_path), str(output_path), processing_func, **kwargs
                )

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")

    def process_single_video(self, input_path: str, output_path: str,
                             processing_func, **kwargs):
        """Process a single video with memory management."""
        # Determine optimal resolution
        with VideoReader(input_path) as reader:
            target_size = self._get_optimal_size(reader.info.width, reader.info.height)

            with VideoWriter(output_path,
                             reader.info.fps * 2,  # Assume 2x frame rate for interpolation
                             target_size[0], target_size[1]) as writer:
                # Process in chunks to manage memory
                for chunk_start in range(0, len(reader), self.batch_size):
                    chunk_end = min(chunk_start + self.batch_size, len(reader))
                    frames = list(reader.read_frames(chunk_start, chunk_end - chunk_start))

                    # Process chunk
                    processed_frames = processing_func(frames, **kwargs)
                    writer.write_frames(processed_frames)

                    # Clear memory
                    del frames, processed_frames
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _get_optimal_size(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate optimal processing size."""
        max_w, max_h = self.max_resolution

        if width <= max_w and height <= max_h:
            return width, height

        # Scale down maintaining aspect ratio
        scale = min(max_w / width, max_h / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Ensure dimensions are divisible by 8 (common requirement for video codecs)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        return new_width, new_height