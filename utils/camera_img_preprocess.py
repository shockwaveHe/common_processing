import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R

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
