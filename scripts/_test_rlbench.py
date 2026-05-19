"""Quick test script for RLBench setup."""
import os
import sys

os.environ["COPPELIASIM_ROOT"] = "/home/yuhe/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["LD_LIBRARY_PATH"] = (
    os.environ["COPPELIASIM_ROOT"] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["QT_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
os.environ["DISPLAY"] = ":99"

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, GripperJointPosition
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PickAndLift

print("Starting...", flush=True)

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.rgb = True
obs_config.front_camera.image_size = (256, 256)

action_mode = ActionMode(JointPosition(), GripperJointPosition())
env = Environment(action_mode, headless=False)
print("Launching CoppeliaSim...", flush=True)
env.launch()

task = env.get_task(PickAndLift)
print(f"Task: PickAndLift, variations: {task.variation_count()}", flush=True)

print("Collecting 1 demo (live_demos=True)...", flush=True)
demos = task.get_demos(amount=1, live_demos=True, max_attempts=1)
print(f"Got {len(demos)} demo(s)", flush=True)

obs = demos[0][0]
print(f"front_rgb shape: {obs.front_rgb.shape}", flush=True)
print(f"gripper_pose: {obs.gripper_pose[:3]}", flush=True)
print(f"gripper_open: {obs.gripper_open}", flush=True)

env.shutdown()
print("SUCCESS!", flush=True)
