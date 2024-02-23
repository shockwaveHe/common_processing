import os
import rosbag
import sys
from utils import bag_processor

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree"
    
    bag_processor.setup_root_dir(root_dir)  

    bag_file = "lidar_points.bag"

    bag_processor.extract_lidar(bag_file, topics=["/velodyne_points"])