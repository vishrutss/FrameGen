import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class FlowEstimator(nn.Module):
    """Optical flow estimation network."""

    def __init__(self, input_channels: int = 6):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            ConvBlock(input_channels, 32),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
        ])

        # Flow output
        self.flow_head = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, x):
        # Encode
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        # Decode with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(features) - 1:
                x = x + features[-(i + 2)]

        flow = self.flow_head(x)
        return flow


class ContextNet(nn.Module):
    """Context extraction network for refinement."""

    def __init__(self, input_channels: int = 3):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvBlock(input_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        ])

        self.output = nn.Conv2d(256, 128, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class RIFEModel(nn.Module):
    """RIFE (Real-Time Intermediate Flow Estimation) model for frame interpolation."""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

        # Flow estimation networks
        self.flow_net = FlowEstimator(input_channels=6)  # 2 RGB frames
        self.context_net = ContextNet(input_channels=3)

        # Refinement network
        self.refine_net = nn.Sequential(
            ConvBlock(9, 64),  # interpolated frame + 2 warped frames
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.to(device)
        logger.info(f"RIFE model initialized on {device}")

    def warp_frame(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp frame using optical flow."""
        B, C, H, W = frame.shape

        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # Apply flow
        flow_normalized = flow / torch.tensor([W, H], device=self.device).view(1, 2, 1, 1) * 2
        warped_grid = grid + flow_normalized
        warped_grid = warped_grid.permute(0, 2, 3, 1)

        # Sample from frame
        warped_frame = F.grid_sample(frame, warped_grid, align_corners=True)
        return warped_frame

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor,
                timestep: float = 0.5) -> torch.Tensor:
        """
        Generate intermediate frame between frame1 and frame2.

        Args:
            frame1: First frame [B, 3, H, W]
            frame2: Second frame [B, 3, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)

        Returns:
            Interpolated frame [B, 3, H, W]
        """
        B, C, H, W = frame1.shape

        # Ensure frames are on correct device
        frame1 = frame1.to(self.device)
        frame2 = frame2.to(self.device)

        # Concatenate frames for flow estimation
        flow_input = torch.cat([frame1, frame2], dim=1)

        # Estimate bidirectional flow
        flow_01 = self.flow_net(flow_input)  # Frame0 -> Frame1
        flow_10 = self.flow_net(torch.cat([frame2, frame1], dim=1))  # Frame1 -> Frame0

        # Scale flows by timestep
        flow_t0 = flow_01 * (1 - timestep)
        flow_t1 = flow_10 * timestep

        # Warp frames
        warped_frame1 = self.warp_frame(frame1, flow_t0)
        warped_frame2 = self.warp_frame(frame2, flow_t1)

        # Initial interpolation (simple blending)
        interpolated = (1 - timestep) * warped_frame1 + timestep * warped_frame2

        # Refinement
        refine_input = torch.cat([interpolated, warped_frame1, warped_frame2], dim=1)
        refined_frame = self.refine_net(refine_input)

        return refined_frame

    def load_pretrained(self, model_path: str):
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def save_model(self, save_path: str, epoch: int = None, optimizer_state: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer_state
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    @torch.no_grad()
    def interpolate_batch(self, frames1: torch.Tensor, frames2: torch.Tensor,
                          timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Batch interpolation for efficiency.

        Args:
            frames1: Batch of first frames [B, 3, H, W]
            frames2: Batch of second frames [B, 3, H, W]
            timesteps: Custom timesteps for each pair [B] or single float

        Returns:
            Interpolated frames [B, 3, H, W]
        """
        self.eval()

        if timesteps is None:
            timesteps = 0.5

        if isinstance(timesteps, (int, float)):
            return self.forward(frames1, frames2, timesteps)
        else:
            # Handle different timesteps per frame pair
            results = []
            for i, t in enumerate(timesteps):
                result = self.forward(frames1[i:i + 1], frames2[i:i + 1], float(t))
                results.append(result)
            return torch.cat(results, dim=0)