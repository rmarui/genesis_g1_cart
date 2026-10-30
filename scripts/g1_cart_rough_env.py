import math

# genesis must be imported before rsl_rl / tensorboard (see g1_env.py)
import genesis as gs
import numpy as np
import torch

from g1_cart_env import G1CartEnv


def make_height_field(t):
    """Smooth random bumps as a Genesis height field (rows: x, columns: y), in units of vertical_scale.

    The bumps are bilinearly interpolated random heights in [-bump_height, bump_height] with a length scale of
    bump_wavelength. The ground is flat up to x = flat_until (start area of the robot and the cart) and the bump
    amplitude grows linearly to its full size over `ramp` metres.
    """
    hs, vs = t["horizontal_scale"], t["vertical_scale"]
    nx = int(round((t["x_range"][1] - t["x_range"][0]) / hs)) + 1
    ny = int(round((t["y_range"][1] - t["y_range"][0]) / hs)) + 1
    wl = t["bump_wavelength"]
    rng = np.random.default_rng(t["seed"])
    coarse = rng.uniform(-1.0, 1.0, (int(math.ceil((nx - 1) * hs / wl)) + 2, int(math.ceil((ny - 1) * hs / wl)) + 2))

    gx, gy = np.arange(nx) * hs / wl, np.arange(ny) * hs / wl
    i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = (gx - i0)[:, None], (gy - j0)[None, :]
    h = (
        coarse[i0][:, j0] * (1 - fx) * (1 - fy)
        + coarse[i0 + 1][:, j0] * fx * (1 - fy)
        + coarse[i0][:, j0 + 1] * (1 - fx) * fy
        + coarse[i0 + 1][:, j0 + 1] * fx * fy
    )
    x = t["x_range"][0] + np.arange(nx) * hs
    amp = np.clip((x - t["flat_until"]) / t["ramp"], 0.0, 1.0)
    h = h * amp[:, None] * t["bump_height"]
    return np.round(h / vs).astype(np.float32)


class G1CartRoughEnv(G1CartEnv):
    """G1CartEnv on uneven ground. All envs share one height field; the robot and the cart start on the flat area at
    the origin and push the cart forward (+x) onto the bumps. Heights used by the rewards / termination are measured
    relative to the terrain below the robot (base height, feet height)."""

    def _add_ground(self):
        t = self.env_cfg["terrain"]
        hf = make_height_field(t)
        self.terrain_x0, self.terrain_y0 = float(t["x_range"][0]), float(t["y_range"][0])
        self.terrain_hs = float(t["horizontal_scale"])
        self.terrain_h = torch.tensor(hf * t["vertical_scale"], dtype=gs.tc_float, device=gs.device)  # [m]
        # the height field starts at its corner, so place that corner at (x0, y0)
        self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(self.terrain_x0, self.terrain_y0, 0.0),
                height_field=hf,
                horizontal_scale=t["horizontal_scale"],
                vertical_scale=t["vertical_scale"],
            )
        )
        print(
            f"[G1CartRoughEnv] terrain {hf.shape[0] * self.terrain_hs:.1f} x {hf.shape[1] * self.terrain_hs:.1f} m, "
            f"x = {t['x_range']}, y = {t['y_range']}, height {float(self.terrain_h.min()):.3f} .. "
            f"{float(self.terrain_h.max()):.3f} m"
        )

    def _ground_height(self, xy):
        nx, ny = self.terrain_h.shape
        u = ((xy[:, 0] - self.terrain_x0) / self.terrain_hs).clamp(0.0, nx - 1.001)
        v = ((xy[:, 1] - self.terrain_y0) / self.terrain_hs).clamp(0.0, ny - 1.001)
        i0, j0 = u.floor().long(), v.floor().long()
        fx, fy = u - i0, v - j0
        h = self.terrain_h
        return (
            h[i0, j0] * (1 - fx) * (1 - fy)
            + h[i0 + 1, j0] * fx * (1 - fy)
            + h[i0, j0 + 1] * (1 - fx) * fy
            + h[i0 + 1, j0 + 1] * fx * fy
        )
