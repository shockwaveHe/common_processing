import os
import rosbag
import sys
from utils import bag_processor

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run3_pole"
    
    bag_processor.setup_root_dir(root_dir)  

    bag_file = "thermal_left.bag"

    bag_processor.extract_img(bag_file, topics=["/thermal_left/image"])