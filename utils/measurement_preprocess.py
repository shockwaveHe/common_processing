import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy 

class CameraPropertiesLoader:
    def __init__(self, calibration_file):
        self.K = None #intrinsics
        self.D = None #distortion
        self.T = None #extrinsics translation
        self.f = None
        self.timeshift_imu_cam = 0.0
        self.width = 0
        self.height = 0

        calibration_data = self.load_calibration_data(calibration_file)
        self.process_kalibr(calibration_data)

    def get_properties(self):
        return self.K, self.D, self.T

    def get_timeshift(self):
        return self.timeshift_imu_cam

    def get_resolution(self):
        return self.width, self.height

    def load_calibration_data(self, calibration_file):
        with open(calibration_file, "r") as f:
            calibration_data = yaml.load(f, Loader=yaml.FullLoader)
        return calibration_data

    def process_calibration(self, calibration_data):
        # Intrinsic parameters
        K = [
            calibration_data["intrinsic"]["projection_parameters"]["fx"],
            0,
            calibration_data["intrinsic"]["projection_parameters"]["cx"],
            0,
            calibration_data["intrinsic"]["projection_parameters"]["fy"],
            calibration_data["intrinsic"]["projection_parameters"]["cy"],
            0,
            0,
            1,
        ]

        # Distortion parameters (for Pinhole model)
        D = [
            calibration_data["intrinsic"]["distortion_parameters"]["k1"],
            calibration_data["intrinsic"]["distortion_parameters"]["k2"],
            calibration_data["intrinsic"]["distortion_parameters"]["p1"],
            calibration_data["intrinsic"]["distortion_parameters"]["p2"],
            0,
        ]  # k3 is 0 for a Pinhole camera

        # Extrinsic parameters (rotation and translation)
        rotation_matrix = np.array(calibration_data["extrinsic"]["rotation_to_body"]["data"]).reshape(3, 3)
        translation_vector = np.array(calibration_data["extrinsic"]["translation_to_body"]["data"]).reshape(3, 1)

        # The rotation matrix and translation vector are already in the correct format for ROS CameraInfo
        R = list(rotation_matrix.flatten())
        P = list((rotation_matrix @ translation_vector).flatten()) + [0, 0, 0, 1]

        # Convert K, D, rotation_matrix, translation_vector to cv2 matrices
        K = np.array(K).reshape(3, 3)
        D = np.array(D).reshape(1, 5)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        translation_vector = np.array(translation_vector).reshape(3, 1)

        self.K = K
        self.D = D
        self.R = rotation_matrix
        self.t = translation_vector

    def process_kalibr(self, calibration_data):
        cam_data = calibration_data["cam0"]

        # Intrinsic parameters
        K = [
            cam_data["intrinsics"][0],  # fx
            0,
            cam_data["intrinsics"][2],  # cx
            0,
            cam_data["intrinsics"][1],  # fy
            cam_data["intrinsics"][3],  # cy
            0,
            0,
            1,
        ]

        # Distortion parameters
        D = cam_data["distortion_coeffs"] + [0]

        T_imu_cam = np.array(cam_data["T_imu_cam"])
        rotation_matrix = T_imu_cam[:3, :3]
        translation_vector = T_imu_cam[:3, 3].reshape(3, 1)

        self.K = np.array(K).reshape(3, 3)
        self.D = np.array(D).reshape(1, 5)

        self.T = np.eye(4)
        self.T[:3, :3] = rotation_matrix
        self.T[:3, 3] = translation_vector.reshape(3)

        self.timeshift_imu_cam = cam_data.get("timeshift_imu_cam", 0.0)

        # horizontal focal length fx
        self.f = cam_data["intrinsics"][0]

        # resolution
        self.width = cam_data["resolution"][0]
        self.height = cam_data["resolution"][1]


# camP = CameraPropertiesLoader("config/fake_thermal_left.yaml")
# print(camP.R)


class UndistortedImage(object):
    def __init__(self, calib_file):
        self.property = CameraPropertiesLoader(calib_file)

    def __call__(self, raw_img):
        K, D, t = self.property.get_properties()
        undistorted_img = cv2.undistort(raw_img, K, D)
        return undistorted_img


class LidarToImgProjection(object):
    def __init__(self, camera_property, max_depth=200):
        self.extrinsic = camera_property.T
        self.K = camera_property.K
        self.max_depth = max_depth
        self.img_width = camera_property.width
        self.img_height = camera_property.height
        
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 1.0
        
        self.cam = o3d.camera.PinholeCameraParameters()
        self.cam.extrinsic = self.extrinsic
        self.cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, self.K)

    def __call__(self, point_cloud):
        """
        Project the point cloud to the image plane.
        @param point_cloud: 3D point cloud (in o3d format)
        """
        point_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))

        render = o3d.cuda.pybind.visualization.rendering.OffscreenRenderer(self.img_width, self.img_height)

        render.scene.set_background([0.0, 0.0, 0.0, 0.0])
        render.scene.add_geometry("point_cloud", point_cloud, self.material)
        render.setup_camera(self.cam.intrinsic, self.cam.extrinsic)
        
        depth_img = np.asarray(render.render_to_depth_image(z_in_view_space=True))

        depth_img[depth_img < 0] = 0
        depth_img[depth_img > self.max_depth] = 0
        return depth_img

    def unproject(self, depth_img, extrinsic=np.eye(4), depth_scale=1.0, depth_trunc=1000.0, stride=1):
        """
        Unproject the depth image to the point cloud at camera frame
        @param depth_img: depth image
        """
        depth_img = depth_img.astype(np.float32)

        print("depth_img shape: ", depth_img.shape)
        print("depth_img dtype: ", depth_img.dtype)

        o3d_image = o3d.geometry.Image(np.random.rand(640,480).astype(np.float32))

        o3d_image = o3d.geometry.Image(depth_img)
        pcl = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_image,
            self.cam.intrinsic,
            extrinsic,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            stride=stride
        )
        
        return pcl


def depth_overlay(img_raw, depth_img):
    img = cv2.addWeighted(depth_img, 0.8, img_raw, 0.2, 0)
    return img

def process_thermal_image(img_16, type="minmax"):
    if type == "minmax":
        img_out = (img_16 - np.min(img_16)) / (np.max(img_16) - np.min(img_16)) * 255
    elif type == "hist_99":
        if np.max(img_16) < 9000:
            img_out = (img_16 - np.min(img_16)) / (np.max(img_16) - np.min(img_16)) * 255
            return img_out.astype(np.uint8)
        intensity, bin_edges = np.histogram(img_16, bins=3)
        upper_bound = bin_edges[1]
        lower_bound = bin_edges[0]

        img_out = np.zeros_like(img_16, dtype=np.uint8)
        mask = (img_16 >= lower_bound) & (img_16 <= upper_bound)
        img_out[mask] = ((img_16[mask] - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
        img_out[img_16 > upper_bound] = np.mean(img_out[img_16 <= upper_bound])
        img_out = (img_out - np.min(img_out)) / (np.max(img_out) - np.min(img_out)) * 255
    else:
        img_out = img_16 / 255
    return img_out.astype(np.uint8)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0.8])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
def stitch_lidar_scan(pcl_files, output_file, origin_index=0, p2p_max_dist=3, icp_max_iter=2000, icp_rmse=1e-9, visualize=True):
    """
    Stitch multiple lidar scans into one point cloud. Assuming that the lidar scans are continuous.
    @param pcl_files: list of point cloud files 
    @param output_file: output file name
    @param origin_index: index of the point cloud frame considered as the base frame
    """
    registrated_pcl = o3d.geometry.PointCloud()

    # registrated_pcl += pcl
    pcl_origin = o3d.io.read_point_cloud(pcl_files[origin_index])

    initial_transformation = np.eye(4)

    last_transformation = initial_transformation

    last_pcl = copy.deepcopy(pcl_origin)
    # start from origin_index, inter to the left and right
    for i in range(origin_index-1, -1, -1):
        pcl_ = o3d.io.read_point_cloud(pcl_files[i])
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl_, last_pcl, p2p_max_dist, initial_transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter, relative_rmse=icp_rmse)
        )
        if visualize:
            print(reg_p2p)
            print(f"Transformation is: \n{reg_p2p.transformation}")
        transform = reg_p2p.transformation @ last_transformation

        last_transformation = transform
        last_pcl = pcl_
        transformed_pcl = copy.deepcopy(pcl_).transform(transform)

        # draw_registration_result(transformed_pcl, pcl_origin, np.eye(4))
        registrated_pcl += transformed_pcl
    
    last_pcl = copy.deepcopy(pcl_origin)
    last_transformation = initial_transformation

    for i in range(origin_index+1, len(pcl_files)):
        pcl_ = o3d.io.read_point_cloud(pcl_files[i])
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl_, last_pcl, p2p_max_dist, initial_transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter, relative_rmse=icp_rmse)
        )
        if visualize:
            print(reg_p2p)
            print(f"Transformation is: \n{reg_p2p.transformation}")

        transform = reg_p2p.transformation @ last_transformation
        last_transformation = transform
        last_pcl = pcl_
        transformed_pcl = copy.deepcopy(pcl_).transform(transform)

        # draw_registration_result(transformed_pcl, pcl_origin, np.eye(4))
        registrated_pcl += transformed_pcl

    if visualize:
        draw_registration_result(registrated_pcl, pcl_origin, np.eye(4))
    registrated_pcl += pcl_origin
    if visualize:    
        draw_registration_result(registrated_pcl, pcl_origin, np.eye(4))
    o3d.io.write_point_cloud(output_file, registrated_pcl)