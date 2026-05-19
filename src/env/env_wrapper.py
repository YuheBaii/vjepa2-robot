"""Thin wrapper around robosuite environments for V-JEPA 2-AC pipeline."""

from __future__ import annotations

import numpy as np


class RobosuiteEnv:
    """Wraps a robosuite environment and exposes render / step / reset."""

    def __init__(self, env_name: str = "Lift", render_size: int = 256):
        import robosuite as suite

        self._env = suite.make(
            env_name=env_name,
            robots="Panda",
            controller_configs=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            camera_widths=render_size,
            camera_heights=render_size,
            camera_depths=False,
            horizon=500,
            control_freq=20,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            lite_physics=False,
        )
        self.render_size = render_size

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        return self._render()

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        return self._render(), reward, done, info

    def _render(self) -> np.ndarray:
        img = self._env.sim.render(
            width=self.render_size,
            height=self.render_size,
            camera_name="frontview",
        )
        return img  # (H, W, 3) uint8

    @property
    def action_dim(self) -> int:
        return self._env.action_spec[0].shape[0]

    def close(self):
        self._env.close()
