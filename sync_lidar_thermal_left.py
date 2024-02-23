import os
import rosbag
import sys
from utils import bag_processor

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree"
    
    bag_processor.setup_root_dir(root_dir)  

    src_folder = "_velodyne_points"

    target_folder = "_thermal_left_image"

    sync_pairs = bag_processor.sync_message(src_folder, target_folder, window_size=20, tolerance=250, time_shift=-0.38)

    print(len(sync_pairs))
    
    bag_processor.generate_sync_folder(src_folder, target_folder, "velodyne_points", "thermal_left", sync_pairs)
