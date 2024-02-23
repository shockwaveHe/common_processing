from utils import bag_processor
from utils.measurement_preprocess import *
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree/velodyne_points2thermal_left"

    bag_processor.setup_root_dir(root_dir)

    lidar_scan_folder = "src"
    lidar_scan_folder = os.path.join(root_dir, lidar_scan_folder)
    registrated_folder = "registrated"
    registrated_folder = os.path.join(root_dir, registrated_folder)

    orig_idx = 11
    # skip_initial = 1896
    skip_initial = 3494
    skip_end = 6931
    lidar_scan_files = os.listdir(lidar_scan_folder)
    lidar_scan_files = sorted(lidar_scan_files)

    with tqdm(total=len(range(max(skip_initial, orig_idx - 1), min(skip_end, len(lidar_scan_files) - orig_idx), 1)), desc="stitching lidar scans") as pbar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for index in range(max(skip_initial, orig_idx - 1), min(skip_end, len(lidar_scan_files) - orig_idx), 1):
                # generate a subset of the lidar scans to process
                consecutive_lidar_scan_files = lidar_scan_files[index - orig_idx + 1: index + orig_idx + 1]
                # abs path
                for i, file in enumerate(consecutive_lidar_scan_files):
                    consecutive_lidar_scan_files[i] = os.path.join(lidar_scan_folder, file)

                executor.submit(stitch_lidar_scan, consecutive_lidar_scan_files, os.path.join(registrated_folder, f"{str(index).zfill(5)}.ply"), origin_index=orig_idx, visualize=False)
                # stitch_lidar_scan(consecutive_lidar_scan_files, os.path.join(registrated_folder, f"{str(index).zfill(5)}.ply"), origin_index=orig_idx, visualize=False)
                
                pbar.update(1)