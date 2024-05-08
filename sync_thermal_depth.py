import os
import rosbag
import sys
from utils import bag_processor
import numpy as np

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree"
    
    bag_processor.setup_root_dir(root_dir)  
    
    src_folder = "velodyne_points2thermal_left"

    src_timestamps_file = os.path.join(root_dir, src_folder, "timestamps.txt")

    target_folder = "devansh"

    target_timestamps_file = os.path.join(root_dir, target_folder, "timestamps.txt")
    # in devansh's format the timestamp is in ms and he removes first several and last several frames

    target_msgs = list(os.listdir(os.path.join(root_dir, target_folder, 'depth_filtered')))
    target_msgs = sorted(target_msgs)

    first_target_idx = target_msgs[0].split(".")[0]
    last_target_idx = target_msgs[-1].split(".")[0]

    target_t = []
    src_t = []
    with open(target_timestamps_file, "r") as file:
        target_t = [int(line.strip()) * 1e6 for line in file]
    print("first_target_idx: ", first_target_idx)
    print("last_target_idx: ", last_target_idx)
    target_t = target_t[int(first_target_idx) : int(last_target_idx) + 1]
    target_t = np.array(target_t)

    with open(src_timestamps_file, "r") as file:
        src_t = [int(line.strip()) for line in file]
    src_t = np.array(src_t)

    print((target_t[0] - src_t[1895]) * 1e-9)
    print((target_t[-1] - src_t[6929]) * 1e-9)
    src_folder = os.path.join(src_folder, 'src')
    target_folder = os.path.join(target_folder, 'depth_filtered')


    sync_pairs = bag_processor.sync_message(src_folder, target_folder, src_t, target_t, window_size=20, tolerance=250, time_shift=0)

    print("found {} pairs".format(len(sync_pairs)))

    # bag_processor.generate_sync_folder()