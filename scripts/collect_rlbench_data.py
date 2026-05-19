"""Collect RLBench PickAndLift demonstrations and save to HDF5.

Records front_rgb, end-effector position, and gripper contact state at each
timestep. Uses live_demos=True to generate demonstrations on the fly.

Usage:
    python scripts/collect_rlbench_data.py --num_demos 50 --output data/rlbench_pick_and_lift.h5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import h5py

logging.basicConfig(level=logging.INFO)

# Ensure CoppeliaSim libs are found before importing RLBench
os.environ.setdefault(
    "COPPELIASIM_ROOT", "/home/yuhe/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
)
os.environ["LD_LIBRARY_PATH"] = (
    os.environ["COPPELIASIM_ROOT"]
    + ":"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ.setdefault("QT_PLUGIN_PATH", os.environ["COPPELIASIM_ROOT"])

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, GripperJointPosition
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.tasks import PickAndLift


def collect_demos(
    num_demos: int, output_path: str, headless: bool = False
) -> None:
    """Collect demonstrations and save to HDF5."""
    action_mode = ActionMode(JointPosition(), GripperJointPosition())
    env = Environment(action_mode, headless=False)
    env.launch()

    task = env.get_task(PickAndLift)
    n_variations = task.variation_count()
    print(f"Task: PickAndLift, variations: {n_variations}")

    all_front_rgb: list[np.ndarray] = []
    all_ee_pos: list[np.ndarray] = []
    all_contact: list[np.ndarray] = []

    for demo_idx in range(num_demos):
        variation = np.random.randint(0, n_variations)
        task.set_variation(variation)

        def _step_callback(obs):
            ee_pos = np.array(obs.gripper_pose[:3], dtype=np.float32)
            contact = np.float32(1.0 - obs.gripper_open)
            all_front_rgb.append(obs.front_rgb.copy())
            all_ee_pos.append(ee_pos)
            all_contact.append(contact)

        try:
            demos = task.get_demos(
                amount=1,
                live_demos=True,
                callable_each_step=_step_callback,
                max_attempts=1,
            )
        except RuntimeError as e:
            print(f"  Demo {demo_idx + 1}: FAILED - {e}")
            print(f"  Skipping variation {variation}, continuing...")
            continue

        n_steps = len(demos[0]._observations)
        print(
            f"  Demo {demo_idx + 1}/{num_demos}: {n_steps} steps, "
            f"variation={variation}"
        )

    env.shutdown()

    front_rgb = np.stack(all_front_rgb, axis=0)
    ee_pos = np.stack(all_ee_pos, axis=0)
    contact = np.stack(all_contact, axis=0)

    print(f"\nTotal frames: {len(front_rgb)}")
    print(f"front_rgb: {front_rgb.shape}, {front_rgb.dtype}")
    print(f"ee_pos:    {ee_pos.shape}")
    print(f"contact:   {contact.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("front_rgb", data=front_rgb, compression="gzip")
        f.create_dataset("ee_pos", data=ee_pos)
        f.create_dataset("contact", data=contact)

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect RLBench PickAndLift demos")
    parser.add_argument(
        "--num_demos", type=int, default=50, help="Number of demos to collect"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rlbench_pick_and_lift.h5",
        help="Output HDF5 path",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run with GUI (requires display)",
    )
    args = parser.parse_args()

    collect_demos(
        num_demos=args.num_demos,
        output_path=args.output,
        headless=not args.headful,
    )


if __name__ == "__main__":
    main()
