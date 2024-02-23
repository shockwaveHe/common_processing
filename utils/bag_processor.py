import rosbag
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

############################################
                # Global
############################################
root_dir = "/home"

def setup_root_dir(root_dir_="/home"):
    global root_dir
    root_dir = root_dir_
    print("successfully set root dir: ", root_dir)

def setup_output_directory(output_dir):
    global root_dir
    os.makedirs(os.path.join(root_dir,output_dir), exist_ok=True)
    print("Write to output directory: ", os.path.join(root_dir,output_dir))
    return os.path.join(root_dir,output_dir)

def create_metadata(bag_name, topic_name, raw=True):
    global root_dir
    metadata_filename = os.path.join(root_dir, topic_name.replace("/","_") + "_desps.txt")
    print(f"Saving metadata to: {metadata_filename}")
    with open(metadata_filename, "w") as f:
        f.write(f"Bag file: {bag_name}\n")
        f.write(f"Topic: {topic_name}\n")
        f.write(f"Raw: {raw}\n")

############################################
        #Lidar extraction functions
############################################

def pcl2ply(pcl_msg, name):
    """
    Convert a pcl message to a pcd file.
    """
    cloud = o3d.geometry.PointCloud()
    points_list = list(pc2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=True))
    points_np = np.array(points_list)
    cloud.points = o3d.utility.Vector3dVector(points_np)
    o3d.io.write_point_cloud(name, cloud)
    # pcl.save(cloud, name)

def extract_lidar(lidar_bag_path, topics=["/velodyne_points"]):
    """
    Extracts the lidar point cloud from the rosbag file and save to pcd.
    """
    global root_dir
    lidar_bag_path = os.path.join(root_dir, lidar_bag_path)
    lidar_bag = rosbag.Bag(lidar_bag_path, "r")
    total_messages = 0
    for topic in topics:
        total_messages += lidar_bag.get_message_count(topic)
    with tqdm(total=total_messages, desc="Processing Lidar Messages") as pbar:
        for topic in topics:
            output_dir = setup_output_directory(topic.replace("/","_"))
            timestamp_txt = os.path.join(root_dir, topic.replace("/","_") + "timestamps.txt")
            f = open(timestamp_txt, "w")
            for topic_, msg_, t_ in lidar_bag.read_messages(topics=topic):
                if pbar.n == 0:
                    create_metadata(lidar_bag_path, topic)
                    print(f"Extracting lidar from topic: {topic}")
                # print(os.path.join(output_dir, f"{t_.to_nsec()}.ply"))
                pcl2ply(msg_, os.path.join(output_dir, f"{t_.to_nsec()}.ply"))
                f.write(f"{t_.to_nsec()}\n")
                pbar.update(1)
            f.close()
            

############################################
        #Image extraction functions
############################################
def extract_img(img_bag_path, topics):
    """
    Extracts images from the rosbag file and save to png.
    """
    global root_dir
    img_bag_path = os.path.join(root_dir, img_bag_path)
    img_bag = rosbag.Bag(img_bag_path, "r")
    total_messages = 0
    if (len(topics) == 0):
        print("No topics to extract!")
        return
    for topic in topics:
        total_messages += img_bag.get_message_count(topic)
    with tqdm(total=total_messages, desc="Processing Image Messages") as pbar:
        for topic in topics:
            output_dir = setup_output_directory(topic.replace("/","_"))
            timestamp_txt = os.path.join(root_dir, topic.replace("/","_") + "timestamps.txt")
            f = open(timestamp_txt, "w")
            for topic_, msg_, t_ in img_bag.read_messages(topics=topic):
                if pbar.n == 0:
                    create_metadata(img_bag_path, topic)
                    print(f"Extracting images from topic: {topic}")
                cv_image = CvBridge().imgmsg_to_cv2(msg_, desired_encoding="passthrough")
                img = cv_image
                image_timestamp = int(t_.to_nsec())
                image_filename = os.path.join(output_dir, f"{image_timestamp}.png")
                cv2.imwrite(image_filename, img)
                f.write(f"{t_.to_nsec()}\n")
                pbar.update(1)
            f.close()

############################################
            # Sync functions
############################################
                
def names_to_timestamp(names):
    timestamps = []
    for name in names:
        timestamps.append(int(name.split(".")[0]))
    timestamps = np.array(timestamps)
    return timestamps


def sync_message(src_folder, target_folder, window_size=20, tolerance=250, time_shift=0):
    
    """
    Syncs the source message to the target message.
    @param src_folders: list of source folders (with timestamps as the file names)
    @param target_folder: target folder (with timestamps as the file names)
    @param window_size: search for the next closest candidate within the window size after previous closest index
    @param tolerance: tolerance in milliseconds
    @param time_shift: time offset between the source and target messages (target_stamp = src_stamp + time_shift)
    return: A synced list with the source messages synced to the target messages
    """
    global root_dir

    target_folder = os.path.join(root_dir, target_folder)

    src_folder = os.listdir(os.path.join(root_dir, src_folder))

    target_msgs = list(os.listdir(target_folder))

    print(f"Found {len(target_msgs)} target messages")
    
    target_timestamps = names_to_timestamp(target_msgs)

    print(f"Syncing source messages to target messages")

    sync_pairs = []

    closest_idx = 0

    # read timestamp
    with tqdm(total=len(src_folder), desc="Synchronizing data") as pbar:
        for src_msg in src_folder:
            src_stamp = int(src_msg.split(".")[0])
            # find the closest target timestamp within the window
            closest_idx = np.argmin(
                np.abs(target_timestamps[max(0, closest_idx - window_size) : 
                                            min(closest_idx + window_size, len(target_msgs))] - src_stamp))

            if (np.abs((float(src_stamp) * 1e-6 + time_shift) - float(target_timestamps[closest_idx]) * 1e-6) < tolerance):
                sync_pairs.append((src_msg, target_msgs[closest_idx]))
            else:
                closest_idx = np.argmin(np.abs(target_timestamps - src_stamp))
                closest_target_stamp = target_timestamps[closest_idx]
                # check if the closest timestamp is within the tolerance
                if np.abs((float(src_stamp) * 1e-6 + time_shift) - float(target_timestamps[closest_idx]) * 1e-6) < tolerance:
                    sync_pairs.append((src_msg, target_msgs[closest_idx]))
                else:
                    print(f"No target message found within tolerance for {src_msg}")
            pbar.update(1) 
    return sync_pairs

def sync_message(src_folder, target_folder, src_timestamps, target_timestamps, window_size=20, tolerance=250, time_shift=0):
    
    """
    Syncs the source message to the target message.
    @param src_stamps: list of source timestamps
    @param src_folders: list of source folders (formated into 00001, 00002, etc.)
    @param target_stamps: target timestamp
    @param target_folder: target folder (formated into 00001, 00002, etc.)
    @param window_size: search for the next closest candidate within the window size after previous closest index
    @param tolerance: tolerance in milliseconds
    @param time_shift: time offset between the source and target messages (target_stamp = src_stamp + time_shift)
    return: A synced list with the source messages synced to the target messages
    """
    global root_dir

    target_folder = os.path.join(root_dir, target_folder)

    src_folder = sorted(os.listdir(os.path.join(root_dir, src_folder)))
    
    target_msgs = sorted(list(os.listdir(target_folder)))

    print(f"Found {len(src_folder)} src messages")

    print(f"Found {len(target_msgs)} target messages")

    assert(len(src_timestamps) == len(src_folder))
    assert(len(target_timestamps) == len(target_msgs))

    print(f"Syncing source messages to target messages")

    sync_pairs = []

    closest_idx = 0

    # read timestamp
    with tqdm(total=len(src_folder), desc="Synchronizing data") as pbar:
        for src_msg, src_stamp in zip(src_folder, src_timestamps):
            # find the closest target timestamp within the window
            closest_idx = np.argmin(
                np.abs(target_timestamps[max(0, closest_idx - window_size) : 
                                            min(closest_idx + window_size, len(target_msgs))] - src_stamp))
            if (np.abs((float(src_stamp) * 1e-6 + time_shift) - float(target_timestamps[closest_idx]) * 1e-6) < tolerance):
                sync_pairs.append((src_msg, target_msgs[closest_idx]))
            else:
                closest_idx = np.argmin(np.abs(target_timestamps - src_stamp))
                closest_target_stamp = target_timestamps[closest_idx]
                # check if the closest timestamp is within the tolerance
                if np.abs((float(src_stamp) * 1e-6 + time_shift) - float(target_timestamps[closest_idx]) * 1e-6) < tolerance:
                    sync_pairs.append((src_msg, target_msgs[closest_idx]))
                else:
                    continue
                    print(f"No target message found within tolerance for {src_msg}")
            pbar.update(1) 
    return sync_pairs

def generate_sync_folder(src_folder, target_folder, src_topic, target_topic, sync_pairs):
    """
    Generate a sync folder with the synced messages.
    """
    global root_dir
    ## create the sync folder
    sync_folder = src_topic + '2' + target_topic
    abs_sync_folder = os.makedirs(os.path.join(root_dir, sync_folder), exist_ok=True)
    os.makedirs(os.path.join(root_dir, sync_folder, "src"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, sync_folder, "target"), exist_ok=True)
    stamp_file = os.path.join(root_dir, sync_folder, "timestamps.txt")
    f = open(stamp_file, "w")
    count = 0
    with tqdm(total=len(sync_pairs), desc="Generating sync results") as pbar:
        for src_msg, target_msg in sync_pairs:
            target_stamp = int(target_msg.split(".")[0])
            src_path = os.path.join(root_dir, src_folder, src_msg)
            target_path = os.path.join(root_dir, target_folder, target_msg)
            index = f"{count:05d}"
            sync_src_path = os.path.join(root_dir, sync_folder, "src", index + '.' + src_msg.split(".")[1])
            sync_target_path = os.path.join(root_dir, sync_folder, "target", index + '.' + target_msg.split(".")[1])
            count += 1
            # copy
            shutil.copy(src_path, sync_src_path)
            shutil.copy(target_path, sync_target_path)
            f.write(f"{target_stamp}\n")
            pbar.update(1)
        f.close()
