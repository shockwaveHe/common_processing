from utils import bag_processor
from utils.measurement_preprocess import *
import os

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree/velodyne_points2thermal_left"

    bag_processor.setup_root_dir(root_dir)

    lidar_scan_folder = "src"

    registrated_folder = "registrated"

    pcl_files = []

    idxes = range(3397, 3417, 1)

    orig_file = None

    orig_idx = 10

    for idx in idxes:
        ply_file = os.path.join(root_dir, lidar_scan_folder, f"{str(idx).zfill(5)}.ply")
        pcl_files.append(ply_file)
        if len(pcl_files) == orig_idx:
            orig_file = f"{str(idx).zfill(5)}.ply"
            print(orig_file)

    

    out_folder = os.path.join(root_dir, registrated_folder)

    print("output folder:", out_folder)

    out_file = os.path.join(out_folder, orig_file)

    print("output file:", out_file)

    stitch_lidar_scan(pcl_files, out_file, origin_index=10)