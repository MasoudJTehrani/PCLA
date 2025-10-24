
import numpy as np
import math
import cv2
import torch


def project_points(points2D_list, K, tvec=None, rvec=None):

  all_points_2d = []
  if rvec is None:
    rvec_new = np.zeros((3, 1), np.float32) 
  else:
    rvec_new = np.array([[-rvec[1], rvec[2], rvec[0]]], np.float32)
  if tvec is None:
    tvec = np.array([[0.0, 2.0, 1.5]], np.float32)

  # print(f"rvec_new: {rvec_new}")
  for point in  points2D_list:
    pos_3d = np.array([point[1], 0, point[0]+tvec[0][2]])
    # Define the distortion coefficients 
    dist_coeffs = np.zeros((5, 1), np.float32) 
    points_2d, _ = cv2.projectPoints(pos_3d, 
                        rvec=rvec_new, tvec=tvec, 
                        cameraMatrix=K, 
                        distCoeffs=dist_coeffs)
    all_points_2d.append(points_2d[0][0])
        
  return all_points_2d

def get_rotation_matrix(roll, pitch, yaw):
    roll = roll * np.pi / 180.0
    pitch = pitch * np.pi / 180.0
    yaw = yaw * np.pi / 180.0

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = pitchMatrix * yawMatrix * rollMatrix

    #inverse rotation
    R = R.T

    return R

def get_camera_intrinsics(w, h, fov):
  """
  Get camera intrinsics matrix from width, height and fov.
  Returns:
    K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
      the carla camera.
  """
  focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
  K = np.identity(3)
  K[0, 0] = K[1, 1] = focal
  K[0, 2] = w / 2.0
  K[1, 2] = h / 2.0

  K = torch.tensor(K, dtype=torch.float32)
  return K

def get_camera_extrinsics():
    """
    Get camera extrinsics matrix for the carla camera.
    extrinsics: A float32 tensor of shape ``[4, 4]`` containing the extrinic calibration matrix for
      the carla camera. The extriniscs are specified as homogeneous matrices of the form ``[R t; 0 1]``
    """
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    extrinsics[3, 3] = 1.0
    extrinsics[:3, :3] = np.eye(3)
    extrinsics[:3, 3] = [-1.5, 0.0, 2.0]
    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

    return extrinsics

