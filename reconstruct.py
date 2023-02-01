import cv2
from enum import Enum
import numpy as np
import open3d as o3d
import os
import sys


intrinsics = np.array([[1083.503747, 0.000000, 966.672214],
	[0.000000, 1082.863379, 541.878867],
	[0.000000, 0.000000, 1.000000]])
baseline = 120.0

general_path = "Y:/cmes_data/cmes_sensors_box/20230131_prevzed/"
file_list = os.listdir(general_path)

file_list_img = [left for left in file_list if left.endswith("color.png")]
file_list_disp = [disp for disp in file_list if disp.endswith("_crestereo.png")]
filename_pcl_tail = "_pointcloud.ply"

disp_levels = 1.0
disp_scale = 0.01

for filename_img, filename_disp in zip(file_list_img, file_list_disp):
  input_image = cv2.imread(general_path + filename_img)
  input_disparity = cv2.imread(general_path + filename_disp, cv2.IMREAD_ANYDEPTH)

  input_width, input_height, _ = input_image.shape

  focal_length = (intrinsics[0][0] + intrinsics[1][1]) * 0.5
  depth = (disp_levels * baseline * focal_length / ((input_disparity * disp_scale) + sys.float_info.epsilon)).astype(np.uint16)

  o3d_image = o3d.geometry.Image(input_image)
  o3d_depth = o3d.geometry.Image(depth)
  o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth)

  pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    input_width,
    input_height,
    intrinsics[0][0],
    intrinsics[1][1],
    intrinsics[0][2],
    intrinsics[1][2])
  pcl = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, pinhole_intrinsics, project_valid_depth_only=False)

  # add rotate
  pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  pcl.orient_normals_towards_camera_location([0.0, 0.0, -1.0])
  
  splitted = filename_img.split("_")
  filename_pcl = general_path + splitted[0] + "_" + splitted[1] + filename_pcl_tail
  o3d.io.write_point_cloud(filename_pcl, pcl)

  print("Point cloud has saved to " + filename_pcl)