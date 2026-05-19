import h5py
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(script_dir, 'rlbench_pick_and_lift.h5')

with h5py.File(h5_path, 'r') as f:
    print(list(f.keys()))                                        # ✅ 在 with 块内
    f.visititems(lambda name, obj: print(name, ':', obj))        # ✅ 在 with 块内