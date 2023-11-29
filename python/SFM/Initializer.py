import os
import sys
sys.path.append('.')
sys.path.append('..')

import logging
import cv2
import numpy as np

from .View import View

def get_pose_from_EssentialMatrix(E):
    """Calculates rotation and translation component from essential matrix"""

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)

    R1 = u @ W @ vt
    R2 = u @ W_t @ vt
    t1 = u[:, -1].reshape((3, 1))
    t2 = - t1
    return R1, R2, t1, t2

def check_determinant(R):
    """Validates using the determinant of the rotation matrix"""

    if np.linalg.det(R) + 1.0 < 1e-9:
        return False
    else:
        return True

def check_triangulation(points, P):
    """Checks whether reconstructed points lie in front of the camera"""

    P = np.vstack((P, np.array([0, 0, 0, 1])))
    reprojected_points = cv2.perspectiveTransform(src=points[np.newaxis], m=P)
    z = reprojected_points[0, :, -1]
    if (np.sum(z > 0)/z.shape[0]) < 0.75:
        return False
    else:
        return True

def get_3D_point(u1, P1, u2, P2):
    """DLT triangulation method"""

    x1_p3_p1 = u1[0] * P1[2, :3] - P1[0, :3]
    y1_p3_p2 = u1[1] * P1[2, :3] - P1[1, :3]
    x2_p3_p1 = u2[0] * P2[2, :3] - P2[0, :3]
    y2_p3_p2 = u2[1] * P2[2, :3] - P2[1, :3]
    A = np.array([x1_p3_p1,
                  y1_p3_p2,
                  x2_p3_p1,
                  y2_p3_p2])
    
    B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                  -(u1[1] * P1[2, 3] - P1[1, 3]),
                  -(u2[0] * P2[2, 3] - P2[0, 3]),
                  -(u2[1] * P2[2, 3] - P2[1, 3])])

    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    # print(f'X: {X}')
    return X[1]

def calculate_reprojection_error(point_3D, point_2D, K, R, t):
    """Calculates the reprojection error for a 3D point by projecting it back into the image plane"""

    reprojected_point = K.dot(R.dot(point_3D) + t)
    reprojected_point = cv2.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T
    error = np.linalg.norm(point_2D.reshape((2, 1)) - reprojected_point)
    return error,reprojected_point

class Initializer:
    """Represents the functions that compute the baseline pose from the initial images of a reconstruction"""

    def __init__(self, view1:View, view2:View,  
                #  K:np.ndarray,
                 ):
        """_summary_

        Args:
            view1 (View): _description_
            view2 (View): _description_
            K (np.ndarray):  camera intrinsic matrix
        """

        self.view1 = view1  # first view
        self.view1.R = np.eye(3, 3)  # identity rotation since the first view is said to be at the origin
        self.view2 = view2  # second view
        # self.K = K  # camera intrinsic matrix

    def get_pose(self, K):
        """Computes and returns the rotation and translation components for the second view"""

        # geometric verification
        F, mask = cv2.findFundamentalMat(self.view1.keypoints[self.view1.corresp_indices], 
                                         self.view2.keypoints[self.view2.corresp_indices], 
                                         method=cv2.FM_RANSAC,
                                         ransacReprojThreshold=0.9, 
                                         confidence=0.99)
        mask = mask.astype(bool).flatten()
        # filter the inlier points
        self.view1.inlier_indices = self.view1.corresp_indices[mask]
        self.view2.inlier_indices = self.view2.corresp_indices[mask]
        logging.info(f"fundamental matrix inliers: {len(self.view1.inlier_indices)}")

        E = K.T @ F @ K  # compute the essential matrix from the fundamental matrix
        logging.info("Computed essential matrix")
        logging.info("Choosing correct pose out of 4 solutions")

        valid_kpts1 = self.view1.keypoints[self.view1.inlier_indices]
        valid_kpts2 = self.view2.keypoints[self.view2.inlier_indices]
        logging.debug(f'valid_kpts1: {valid_kpts1.shape}')
        logging.debug(f'valid_kpts2: {valid_kpts2.shape}')
        # retval, R, t, mask = cv2.recoverPose(E, valid_kpts1, valid_kpts2, K)
        # logging.info(f"retval: \n{retval} \nR: \n{R} \nt: \n{t} \nmask: \n{mask}")
        return self.check_pose(E, K)

    def check_pose(self, E, K):
        """Retrieves the rotation and translation components from the essential matrix by decomposing it and verifying the validity of the 4 possible solutions"""

        R1, R2, t1, t2 = get_pose_from_EssentialMatrix(E)  # decompose E
        if not check_determinant(R1):
            R1, R2, t1, t2 = get_pose_from_EssentialMatrix(-E)  # change sign of E if R1 fails the determinant test

        # solution 1
        reprojection_error, points_3D = self.triangulate(K, R1, t1)
        # check if reprojection is not faulty and if the points are correctly triangulated in the front of the camera
        if reprojection_error > 100.0 or not check_triangulation(points_3D, np.hstack((R1, t1))):

            # solution 2
            reprojection_error, points_3D = self.triangulate(K, R1, t2)
            if reprojection_error > 100.0 or not check_triangulation(points_3D, np.hstack((R1, t2))):

                # solution 3
                reprojection_error, points_3D = self.triangulate(K, R2, t1)
                if reprojection_error > 100.0 or not check_triangulation(points_3D, np.hstack((R2, t1))):

                    # solution 4
                    return R2, t2

                else:
                    return R2, t1

            else:
                return R1, t2

        else:
            return R1, t1

    def triangulate(self, K, R, t):
        """Triangulate points between the baseline views and calculates the mean reprojection error of the triangulation"""

        K_inv = np.linalg.inv(K)
        P1 = np.hstack((self.view1.R, self.view1.t))
        P2 = np.hstack((R, t))

        # only reconstructs the inlier points filtered using the fundamental matrix
        pixel_points1, pixel_points2 = self.view1.keypoints[self.view1.inlier_indices], self.view2.keypoints[self.view2.inlier_indices]

        # convert 2D pixel points to homogeneous coordinates
        pixel_points1 = np.concatenate((pixel_points1, np.ones((pixel_points1.shape[0], 1))), axis=1)
        pixel_points2 = np.concatenate((pixel_points2, np.ones((pixel_points2.shape[0], 1))), axis=1)

        reprojection_error = []

        points_3D = np.zeros((0, 3))  # stores the triangulated points

        # # triangulate points
        # points_4D = cv2.triangulatePoints(P1, P2, pixel_points1.T, pixel_points2.T)

        for i in range(len(pixel_points1)):
            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            # convert homogeneous 2D points to normalized device coordinates
            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            # calculate 3D point
            point_3D = get_3D_point(u1_normalized, P1, u2_normalized, P2)

            # calculate reprojection error
            error, _ = calculate_reprojection_error(point_3D[:3], u2[0:2], K, R, t)
            reprojection_error.append(error)

            # append point
            points_3D = np.concatenate((points_3D, point_3D[:3].T), axis=0)

        return np.mean(reprojection_error), points_3D