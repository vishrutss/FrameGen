#!/usr/bin/env python3
"""
FrameGen - AI-Powered Video Frame Interpolation

Main entry point for the frame generation application.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import time
from typing import Dict, Any, Tuple

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from processing.frame_interpolator import (
        FrameInterpolator,
        InterpolationConfig,
        MultiScaleInterpolator,
        create_interpolation_pipeline
    )
    from processing.video_io import VideoReader, BatchVideoProcessor
    from utils.config_loader import load_config, validate_config
    from utils.progress_tracker import ProgressTracker
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


def load_configuration(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load and validate configuration."""
    try:
        if not Path(config_path).exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return create_default_config()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate configuration
        validate_config(config)
        logger.info(f"Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return create_default_config()


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'model': {
            'device': 'auto',
            'model_path': 'models/rife_weights.pth'
        },
        'interpolation': {
            'target_fps': 60.0,
            'interpolation_factor': 2,
            'temporal_consistency': True
        },
        'processing': {
            'batch_size': 4,
            'max_resolution': [1920, 1080],
            'memory_efficient': True
        },
        'output': {
            'quality': 23,
            'codec': 'mp4v'
        }
    }


def create_interpolator(config: Dict[str, Any]) -> FrameInterpolator:
    """Create frame interpolator from configuration."""
    # Extract interpolation config
    max_res = config['processing'].get('max_resolution', [1920, 1080])
    max_resolution: Tuple[int, int] = (int(max_res[0]), int(max_res[1]))

    interp_config = InterpolationConfig(
        target_fps=config['interpolation'].get('target_fps', 60.0),
        interpolation_factor=config['interpolation'].get('interpolation_factor', 2),
        batch_size=config['processing'].get('batch_size', 4),
        max_resolution=max_resolution,
        model_path=config['model'].get('model_path'),
        device=config['model'].get('device', 'auto'),
        quality=config['output'].get('quality', 23),
        temporal_consistency=config['interpolation'].get('temporal_consistency', True),
        memory_efficient=config['processing'].get('memory_efficient', True)
    )

    # Choose interpolator type
    if config.get('interpolation', {}).get('multi_scale', False):
        return MultiScaleInterpolator(interp_config)
    else:
        return FrameInterpolator(interp_config)


def interpolate_single_video(input_path: str, output_path: str, config: Dict[str, Any]):
    """Interpolate a single video file."""
    logger.info(f"Processing: {input_path} -> {output_path}")

    # Create interpolator
    interpolator = create_interpolator(config)

    # Setup progress tracking
    progress_tracker = ProgressTracker()

    def progress_callback(progress: float, current: int, total: int):
        progress_tracker.update(progress, current, total)

    try:
        # Start interpolation
        start_time = time.time()
        stats = interpolator.interpolate_video(
            input_path, output_path, progress_callback
        )

        processing_time = time.time() - start_time

        # Print results
        logger.info("=" * 50)
        logger.info("INTERPOLATION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Frames processed: {stats['frames_processed']}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Average FPS: {stats['average_fps']:.2f}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise


def interpolate_batch(input_dir: str, output_dir: str, config: Dict[str, Any]):
    """Interpolate multiple videos in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find video files
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []

    for ext in supported_formats:
        video_files.extend(input_path.glob(f"*{ext}"))

    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    logger.info(f"Found {len(video_files)} video files to process")

    # Process each video
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"Processing video {i}/{len(video_files)}: {video_file.name}")

        output_file = output_path / f"interpolated_{video_file.name}"

        try:
            interpolate_single_video(str(video_file), str(output_file), config)
        except Exception as e:
            logger.error(f"Failed to process {video_file.name}: {e}")
            continue


def benchmark_performance(test_video: str, config: Dict[str, Any]):
    """Run performance benchmark."""
    logger.info("Starting performance benchmark...")

    interpolator = create_interpolator(config)

    try:
        stats = interpolator.benchmark(test_video, num_frames=100)

        logger.info("=" * 50)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="FrameGen - AI-Powered Video Frame Interpolation"
    )

    parser.add_argument(
        'mode',
        choices=['single', 'batch', 'benchmark'],
        help='Processing mode'
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video file or directory'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output video file or directory'
    )

    parser.add_argument(
        '--config', '-c',
        default='config/settings.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--target-fps',
        type=float,
        help='Target output frame rate'
    )

    parser.add_argument(
        '--factor',
        type=int,
        help='Interpolation factor (2x, 3x, 4x)'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        help='Processing device'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_configuration(args.config)

    # Override config with command line arguments
    if args.target_fps:
        config['interpolation']['target_fps'] = args.target_fps
    if args.factor:
        config['interpolation']['interpolation_factor'] = args.factor
    if args.device:
        config['model']['device'] = args.device

    try:
        if args.mode == 'single':
            if not args.output:
                # Generate output filename
                input_path = Path(args.input)
                args.output = str(input_path.parent / f"interpolated_{input_path.name}")

            interpolate_single_video(args.input, args.output, config)

        elif args.mode == 'batch':
            if not args.output:
                args.output = str(Path(args.input).parent / "interpolated")

            interpolate_batch(args.input, args.output, config)

        elif args.mode == 'benchmark':
            benchmark_performance(args.input, config)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()