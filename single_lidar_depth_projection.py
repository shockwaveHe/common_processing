from utils.measurement_preprocess import *
import numpy as np
import os

if __name__ == "__main__":
    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree/velodyne_points2thermal_left"

    # sample_lidar_file = os.path.join(root_dir, "src/01521.ply")
    # sample_img_file = os.path.join(root_dir, "target/01521.png")
    # sample_lidar_file = os.path.join(root_dir, "src/03407.ply")
    # sample_img_file = os.path.join(root_dir, "target/03407.png")
    sample_lidar_file = os.path.join(root_dir, "registrated/02305.ply")
    sample_img_file = os.path.join(root_dir, "target/02305.png")

    # reading 
    config_file_path = "/home/airlab3090ti/ReferenceCode/point_cloud_to_depth/config/fake_thermal_left.yaml"

    cam = CameraPropertiesLoader(config_file_path)

    print("camera intrinsic parameters: \n", cam.K)
    print("camera distortion parameters: \n", cam.D)
    print("camera extrinsic parameters: \n", cam.T)

    T_imu_lidar = np.eye(4)
    rotation = np.radians(180)
    T_imu_lidar[:3, :3] = np.array(
        [[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]
    )
    # T_imu_lidar[:3, 3] = np.array([-0.116655, 0.0, 0.082 + 0.0377])
    T_imu_lidar[:3, 3] = np.array([-0.116655, 0.0, 0.082 + 0.0377])

    print("lidar to imu transformation: \n", T_imu_lidar)

    T_lidar_imu = np.linalg.inv(T_imu_lidar)
    print("imu to lidar transformation: \n", T_lidar_imu)

    cam.T = T_lidar_imu @ cam.T # T_imu_cam  
    # cam.T = cam.T @ T_imu_lidar
    print("camera extrinsic parameters after transformation: \n", cam.T)
    # cam.T[:3, :3] = np.array(
    #     [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    # )
    cam.T = np.linalg.inv(cam.T)
    print("camera extrinsic parameters after transformation: \n", cam.T)

    image_undistorter = UndistortedImage(config_file_path)

    img = cv2.imread(sample_img_file)

    processed_img = process_thermal_image(img)

    undistorted_img = image_undistorter(processed_img)

    l2c_projector = LidarToImgProjection(cam)

    point_cloud = o3d.io.read_point_cloud(sample_lidar_file)
    np_pcd = np.asarray(point_cloud.points)

    depth_img = l2c_projector(np_pcd)
    # plt.imshow(depth_img)
    # plt.show()
    depth_img = depth_img.astype(np.uint16)

    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

    # overlay_image = depth_overlay(undistorted_img, depth_img)
    # plt.imshow(overlay_image)
    # plt.show()

    # plt.subplot(2,1,1)
    plt.imshow(undistorted_img)
    plt.imshow(depth_img, alpha=0.5)
    # plt.subplot(2,1,2)
    # plt.imshow(undistorted_img)
    plt.show()


