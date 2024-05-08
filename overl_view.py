from utils import bag_processor
from utils.measurement_preprocess import *
import os
import open3d as o3d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/hawkins_run1_tree"

    thermal_img_path = os.path.join(root_dir, "img_left/09809.png") 
    depth_img_corrected = os.path.join(root_dir, "depth_icp_aligned/09808.png") 
    depth_img_before_corrected = os.path.join(root_dir, "depth_filtered/09808.png")
    print(root_dir)
    # extrinsic parameters

    # reading camera config
    config_file_path = "/home/airlab3090ti/ReferenceCode/point_cloud_to_depth/config/fake_thermal_left.yaml"

   
    # project depth back to thermal image frame
    image_undistorter = UndistortedImage(config_file_path)
    img = cv2.imread(thermal_img_path, cv2.IMREAD_UNCHANGED)
    # processed_img = process_thermal_image(img)
    # undistorted_img = image_undistorter(processed_img)
    # plt.imshow(depth_img)
    # plt.show()
    depth_img = cv2.imread(depth_img_corrected)
    depth_img = depth_img.astype(np.uint16)

    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)


    depth_img_uncorrected = cv2.imread(depth_img_before_corrected)
    depth_img_uncorrected = depth_img_uncorrected.astype(np.uint16)

    depth_img_uncorrected = cv2.cvtColor(depth_img_uncorrected, cv2.COLOR_BGR2GRAY)
    # overlay_image = depth_overlay(undistorted_img, depth_img)
    # plt.imshow(overlay_image)
    # plt.show()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    plt.subplot(2,2,1)
    plt.title("raw img")
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.title("corrected depth img")
    plt.imshow(depth_img)
    plt.subplot(2,2,3)
    plt.title("uncorrected depth img")
    plt.imshow(depth_img_uncorrected)
    plt.subplot(2,2,4)
    plt.title("overlay")
    plt.imshow(img)
    plt.imshow(depth_img, alpha=0.5)
    # plt.subplot(2,1,2)
    # plt.imshow(undistorted_img)
    plt.show()
