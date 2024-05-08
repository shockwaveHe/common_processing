from utils import bag_processor
from utils.measurement_preprocess import *
import os
import open3d as o3d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/run1_handheld_tree/"

    # timestamp = 1679940089541803258 # in ns
    # depth_from_SLAM = os.path.join(root_dir, "devansh/depth_filtered/09818.png") # in camera, project to lidar frame
    # depth_from_lidar = os.path.join(root_dir, "velodyne_points2thermal_left/single_registrated/03406.ply") # in lidar frame
    # thermal_img_path = os.path.join(root_dir, "velodyne_points2thermal_left/target/03406.png") # in camera frame

    root_dir = "/media/airlab3090ti/Extreme Pro3/wildfire/2024-01-26-Hawkins/hawkins_run1_tree"

    depth_from_SLAM = os.path.join(root_dir, "depth_filtered/09809.png") # in camera, project to lidar frame
    depth_from_lidar = os.path.join(root_dir, "velodyne_points2thermal_left/single_registrated/03406.ply") # in lidar frame
    thermal_img_path = os.path.join(root_dir, "velodyne_points2thermal_left/target/03406.png") # in camera frame
    # extrinsic parameters

    # reading camera config
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

    depth_img_from_SLAM = np.array(cv2.imread(depth_from_SLAM, cv2.IMREAD_UNCHANGED).astype(np.float32)) / 256.0

    # plt.imshow(depth_img_from_SLAM)
    # plt.show()

    pcl_from_lidar = o3d.io.read_point_cloud(depth_from_lidar)

    l2c_projector = LidarToImgProjection(cam)
    
    # Unproject depth
    
    pcl_from_SLAM = l2c_projector.unproject(depth_img_from_SLAM, extrinsic=cam.T, depth_scale=1)

    draw_registration_result(pcl_from_SLAM, pcl_from_lidar, np.eye(4))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcl_from_SLAM, pcl_from_lidar, 2, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000, relative_rmse=1e-9)
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    draw_registration_result(pcl_from_SLAM, pcl_from_lidar, reg_p2p.transformation)

    # project depth back to thermal image frame
    image_undistorter = UndistortedImage(config_file_path)
    img = cv2.imread(thermal_img_path)
    processed_img = process_thermal_image(img)
    undistorted_img = image_undistorter(processed_img)


    point_cloud = copy.deepcopy(pcl_from_SLAM).transform(reg_p2p.transformation)
    
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

