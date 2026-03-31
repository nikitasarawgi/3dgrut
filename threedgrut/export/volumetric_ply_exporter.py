import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

from threedgrut.export.base import ModelExporter
from threedgrut.utils.logger import logger

class VolumetricPointCloudExporter(ModelExporter):
    def __init__(self, resolution=128, density_threshold=0.1, max_points=1_000_000, device="cuda"):
        self.resolution = resolution
        self.density_threshold = density_threshold
        self.max_points = max_points
        self.device = device

    @torch.no_grad()
    def export(self, model, output_path: Path, dataset=None, conf=None, **kwargs):
        logger.info(f"Exporting volumetric point cloud to {output_path}...")

        # 1. Get bounding box
        positions = model.get_positions().to(self.device)
        bbox_min = positions.min(dim=0).values
        bbox_max = positions.max(dim=0).values
        padding = 0.05 * (bbox_max - bbox_min)
        bbox_min -= padding
        bbox_max += padding

        # 2. Create grid
        xs = torch.linspace(bbox_min[0], bbox_max[0], self.resolution, device=self.device)
        ys = torch.linspace(bbox_min[1], bbox_max[1], self.resolution, device=self.device)
        zs = torch.linspace(bbox_min[2], bbox_max[2], self.resolution, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)

        # 3. Evaluate density at grid points (batched)
        batch_size = 32_768
        densities_list = []
        for start in range(0, grid.shape[0], batch_size):
            end = min(start + batch_size, grid.shape[0])
            pts_batch = grid[start:end]
            dens_batch = model.compute_density_at(pts_batch)  # <-- implement this in your model
            if dens_batch.ndim > 1:
                dens_batch = dens_batch.squeeze(-1)
            densities_list.append(dens_batch.cpu())
        densities = torch.cat(densities_list, dim=0)

        # 4. Threshold
        mask = densities > self.density_threshold
        kept_points = grid[mask.to(grid.device)]
        kept_dens = densities[mask.cpu()]
        num_kept = kept_points.shape[0]
        if num_kept == 0:
            logger.warning("No points above threshold — nothing to export.")
            return

        # 5. Downsample if needed
        if num_kept > self.max_points:
            idx = torch.randperm(num_kept)[: self.max_points]
            kept_points = kept_points[idx]
            kept_dens = kept_dens[idx]
            num_kept = self.max_points

        kept_points = kept_points.cpu().numpy()
        kept_dens = kept_dens.cpu().numpy()

        # 6. Color (simple: density to grayscale)
        intensities = np.clip(kept_dens / kept_dens.max(), 0.0, 1.0)
        rgb = (intensities[:, None] * 255.0).astype(np.uint8)
        r = rgb[:, 0]
        g = rgb[:, 0]
        b = rgb[:, 0]

        # 7. Write PLY
        dtype_full = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("density", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        elements = np.empty(num_kept, dtype=dtype_full)
        elements["x"] = kept_points[:, 0].astype(np.float32)
        elements["y"] = kept_points[:, 1].astype(np.float32)
        elements["z"] = kept_points[:, 2].astype(np.float32)
        elements["density"] = kept_dens.astype(np.float32)
        elements["red"] = r
        elements["green"] = g
        elements["blue"] = b

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(str(output_path))
        logger.info(f"Volumetric point cloud successfully written to {output_path}")