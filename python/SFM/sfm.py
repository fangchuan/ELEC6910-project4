import os
import sys
sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np
import logging

import open3d as o3d
from utils.PinholeCamera import PinholeCamera
from utils.tools import Tcw2qvec

from .Initializer import Initializer, get_3D_point, calculate_reprojection_error
from .View import View
from .nn_matcher import KNNMatcherWithGeometricVerification
from .MapPoint import MapPoint
from .BundleAdjustment import BundleAdjustment

class SFM:
    """Represents the main reconstruction loop"""

    def __init__(self, dataloader, detector, matcher, camera: PinholeCamera, config:dict, output_dir:str='./output'):
        """_summary_

        Args:
            detector (_type_): keypooint detector
            matcher (_type_):  keypoint matcher
            camera (PinholeCamera): camera intrinsic parameters
            output_dir (str, optional): _description_. Defaults to './output'.
        """
        self.dataloader = dataloader
        self.detector = detector
        self.matcher = matcher

        self.names = dataloader.img_path_lst
        self.prev_frames_lst = []  # list of views that have been reconstructed
        self.K = np.array([[camera.fx, 0., camera.cx],
                            [0., camera.fy, camera.cy],
                            [0., 0., 1.]
                           ])  # intrinsic matrix
        # self.points_3D = np.zeros((0, 3))  # reconstructed 3D points
        self.map = [] # list of 3D map points
        self.point_counter = 0  # keeps track of the reconstructed points
        self.point_map = {}  # a dictionary of the 2D points that contributed to a given 3D point
        self.errors = []  # list of mean reprojection errors taken at the end of every new view being added

        self.kptdescs = {}
        self.prev_view = None
        self.global_feat_matcher = KNNMatcherWithGeometricVerification({'method': 'mutual_nn_ratio', 'ratio':0.8})

        # store results in a root_path/points
        self.ply_path = os.path.join(output_dir, 'points')
        if not os.path.exists(self.ply_path):
            os.makedirs(self.ply_path)
        self.image_path = os.path.join(output_dir, 'images')
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)

        # number of views to skip when reconstructing points
        self.KEYFRAME_INTVAL = int(config["kfrm_interval"])
        self.RECON_PREV_KEYFRAME_NUM = int(config["recon_kfrm_num"])
        self.PNP_PREV_KEYFRAME_NUM = int(config["pnp_kfrm_num"])
        self.PNP_REPROJECTION_THRESH = float(config["pnp_reprojection_thresh"])

    def get_index_of_view(self, view:View):
        """Extracts the position of a view in the list of views"""

        return self.names.index(view.name)
    
    def skip_mapped_points(self, view1_inliers, view2_inliers, image_idx):
        """Removes points that have already been reconstructed in the completed views"""

        inliers1 = []
        inliers2 = []

        for i in range(len(view1_inliers)):
            if (image_idx, view1_inliers[i]) not in self.point_map:
                inliers1.append(view1_inliers[i])
                inliers2.append(view2_inliers[i])

        return inliers1, inliers2

    def compute_pose(self, img1:np.ndarray, img1_name:str, img2:np.ndarray=None, img2_name:str=None, is_init_views:bool=False):
        """Computes the pose of the new view"""

        # procedure for initial pose estimation
        if is_init_views and img2 is not None:
            
            self.kptdescs["ref"] = self.detector(img1)
            self.kptdescs["cur"] = self.detector(img2)
            matches_dict = self.matcher(self.kptdescs)
            logging.info(f'init_view_matches: {len(matches_dict["ref_indices"])}')

            view1_kpts, view1_desc, view1_scores = self.kptdescs["ref"]['keypoints'], self.kptdescs["ref"]['descriptors'], self.kptdescs["ref"]['scores']
            view1 = View(color_image=img1, 
                         keypoints_lst=view1_kpts, 
                         descriptors_lst=view1_desc, 
                         feature_type=self.detector.config["name"], 
                         image_name=img1_name,
                         correspondences=matches_dict['ref_indices'],
                         keypoint_scores=view1_scores)

            view2_kpts, view2_desc, view2_scores = self.kptdescs["cur"]['keypoints'], self.kptdescs["cur"]['descriptors'], self.kptdescs["cur"]['scores']
            view2 = View(color_image=img2, 
                         keypoints_lst=view2_kpts, 
                         descriptors_lst=view2_desc, 
                         feature_type=self.detector.config["name"], 
                         image_name=img2_name,
                         correspondences=matches_dict['cur_indices'],
                         keypoint_scores=view2_scores)
            initializer = Initializer(view1, view2)
            view2.R, view2.t = initializer.get_pose(self.K)

            rpe1, rpe2, vis_repro_img = self.triangulate(view1, view2)
            save_img_filepath = os.path.join(self.image_path, str(self.get_index_of_view(view1)) + '_' + str(self.get_index_of_view(view2)) + '_reprojection.png')
            cv2.imwrite(save_img_filepath, vis_repro_img)
            self.errors.append(np.mean(rpe1))
            self.errors.append(np.mean(rpe2))

            self.prev_frames_lst.append(view1)
            self.prev_frames_lst.append(view2)
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.prev_view = view2


        # procedure for estimating the pose of all other views
        else:
            self.kptdescs["cur"] = self.detector(img1)
            cur_view_kpts, cur_view_desc, cur_view_scores = self.kptdescs["cur"]['keypoints'], self.kptdescs["cur"]['descriptors'], self.kptdescs["cur"]['scores']
            cur_view = View(color_image=img1, 
                            keypoints_lst=cur_view_kpts, 
                            descriptors_lst=cur_view_desc, 
                            feature_type=self.detector.config["name"], 
                            image_name=img1_name,
                            keypoint_scores=cur_view_scores)
            cur_view.R, cur_view.t = self.compute_pose_PNP(cur_view)
            errors = []

            # reconstruct unreconstructed points from all of the previous keyframes
            for i in range(len(self.prev_frames_lst)-1, max(len(self.prev_frames_lst)-self.RECON_PREV_KEYFRAME_NUM, -1), -1):
                old_view = self.prev_frames_lst[i]
                if i % self.KEYFRAME_INTVAL != 0:
                    continue

                logging.debug(f"Reconstructing points from keyframe view {self.get_index_of_view(old_view)} and {self.get_index_of_view(cur_view)}")
                # match old view keypoints against the new view keypoints
                self.kptdescs["ref"]["keypoints"] = old_view.keypoints
                self.kptdescs["ref"]["descriptors"] = old_view.descriptors
                self.kptdescs["ref"]["scores"] = old_view.scores
                sg_matches = self.matcher(self.kptdescs)
                logging.debug(f'cur_view matches view {self.get_index_of_view(old_view)}: {len(sg_matches["ref_indices"])}')
                # geometric verification using fundamental matrix
                F, mask = cv2.findFundamentalMat(sg_matches["ref_keypoints"], sg_matches["cur_keypoints"], method=cv2.FM_RANSAC,
                                            ransacReprojThreshold=0.9, confidence=0.99)
                mask = mask.astype(bool).flatten()
                old_view_inlier_idxs = sg_matches["ref_indices"][mask]
                cur_view_inlier_idxs = sg_matches["cur_indices"][mask]
                # logging.info(f"inliers after geometric verification: {len(old_view_inlier_idxs)}")

                # Verify if the current point has an associated map point
                old_view_inlier_idxs_filtered = []
                cur_view_inlier_idxs_filtered = []
                for i in range(len(old_view_inlier_idxs)):
                    if (self.get_index_of_view(old_view), old_view_inlier_idxs[i]) in self.point_map:
                        # add observation to the map point
                        map_point_idx = self.point_map[(self.get_index_of_view(old_view), old_view_inlier_idxs[i])]
                        self.map[map_point_idx].add_observation(self.get_index_of_view(cur_view), cur_view_inlier_idxs[i])
                    else:
                        # add new map point using the observation pair
                        old_view_inlier_idxs_filtered.append(old_view_inlier_idxs[i])
                        cur_view_inlier_idxs_filtered.append(cur_view_inlier_idxs[i])

                if len(old_view_inlier_idxs_filtered) < 10:
                    logging.debug(f"No new points to reconstruct from view {self.get_index_of_view(old_view) and self.get_index_of_view(cur_view)}")
                    continue
                # triangulate the filtered points
                _, rpe, vis_reprojection_img = self.triangulate(old_view, cur_view, old_view_inlier_idxs_filtered, cur_view_inlier_idxs_filtered)
                errors += rpe
                # save the reprojection image
                save_img_filepath = os.path.join(self.image_path, str(self.get_index_of_view(old_view)) + '_' + str(self.get_index_of_view(cur_view)) + '_reprojection.png')
                cv2.imwrite(save_img_filepath, vis_reprojection_img)

            self.prev_frames_lst.append(cur_view)
            if len(errors) > 0:
                self.errors.append(np.mean(errors))

    def check_point_angle(self, point_3D:np.ndarray, R1cw:np.ndarray, t1cw:np.ndarray, R2cw:np.ndarray, t2cw:np.ndarray):
        """ check if 3D map point cos angle w.r.t two view

        Args:
            point_3D (np.ndarray): 3d point by triangulation
            R1cw (np.ndarray): view1 rotation matrix 3x3
            t1cw (np.ndarray): view1 translation vector 1x3
            R2cw (np.ndarray): view2 rotation matrix 3x3
            t2cw (np.ndarray): view2 translation vector 1x3
        """
        v1_cam_center = -R1cw.T @ t1cw
        v2_cam_center = -R2cw.T @ t2cw
        n1 = (v1_cam_center - point_3D)
        n1 = n1 / np.linalg.norm(n1)
        n2 = (v2_cam_center - point_3D)
        n2 = n2 / np.linalg.norm(n2)
        cos_angle = np.dot(n1.T, n2)
        return cos_angle < 0.999

    def triangulate(self, view1:View, view2:View, view1_inlier_idxs:np.array=None, view2_inlier_idxs:np.array=None):
        """Triangulates 3D points from two views whose poses have been recovered. Also updates the point_map dictionary"""

        K_inv = np.linalg.inv(self.K)
        P1 = np.hstack((view1.R, view1.t))
        P2 = np.hstack((view2.R, view2.t))

        # only reconstructs the filtered inlier points 
        if view1_inlier_idxs is not None and view2_inlier_idxs is not None:
            pixel_points1, pixel_points2 = view1.keypoints[view1_inlier_idxs], view2.keypoints[view2_inlier_idxs]
        else:
            pixel_points1, pixel_points2 = view1.keypoints[view1.inlier_indices], view2.keypoints[view2.inlier_indices]

        # convert 2D pixel points to homogeneous coordinates
        pixel_points1 = np.concatenate((pixel_points1, np.ones((pixel_points1.shape[0], 1))), axis=1)
        pixel_points2 = np.concatenate((pixel_points2, np.ones((pixel_points2.shape[0], 1))), axis=1)
        logging.debug(f'pixel_points: {pixel_points1.shape}')

        reprojection_error1 = []
        reprojection_error2 = []

        vis_repro_image1 = view1.image.copy()
        vis_repro_image2 = view2.image.copy()

        # cv2.triangulatePoints(P1, P2, pixel_points1.T, pixel_points2.T)
        for i in range(len(pixel_points1)):

            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            point_3D = get_3D_point(u1_normalized, P1, u2_normalized, P2)
            # check if the point is in front of both cameras
            # if point_3D[2,0] < 0 and self.check_point_angle(point_3D, view1.R, view1.t, view2.R, view2.t):
            if point_3D[2,0] < 0:
                continue

            if view1_inlier_idxs is not None and view2_inlier_idxs is not None:
                map_point = MapPoint(position=point_3D.T,  feature=view1.descriptors[view1_inlier_idxs[i]], idx=self.point_counter)
            else:
                map_point = MapPoint(position=point_3D.T,  feature=view1.descriptors[view1.inlier_indices[i]], idx=self.point_counter)

            # calculate and visualize reprojection error
            error1, reprojected1 = calculate_reprojection_error(point_3D, u1[0:2], self.K, view1.R, view1.t)
            # observed point
            cv2.circle(vis_repro_image1, (int(u1[0]), int(u1[1])), 2, (0, 255, 0), -1)
            # projected point
            cv2.circle(vis_repro_image1, (int(reprojected1[0]), int(reprojected1[1])), 2, (0, 0, 255), -1)
            reprojection_error1.append(error1)
            error2, reprojected2 = calculate_reprojection_error(point_3D, u2[0:2], self.K, view2.R, view2.t)
            cv2.circle(vis_repro_image2, (int(u2[0]), int(u2[1])), 2, (0, 255, 0), -1)
            cv2.circle(vis_repro_image2, (int(reprojected2[0]), int(reprojected2[1])), 2, (0, 0, 255), -1)
            reprojection_error2.append(error2)

            # updates point_map with the key (index of view, index of point in the view) and value point_counter
            # multiple keys can have the same value because a 3D point is reconstructed using 2 points
            inliers1 = view1_inlier_idxs if view1_inlier_idxs is not None else view1.inlier_indices
            inliers2 = view2_inlier_idxs if view2_inlier_idxs is not None else view2.inlier_indices
            self.point_map[(self.get_index_of_view(view1), inliers1[i])] = self.point_counter
            self.point_map[(self.get_index_of_view(view2), inliers2[i])] = self.point_counter
            map_point.add_observation(self.get_index_of_view(view1), inliers1[i])
            map_point.add_observation(self.get_index_of_view(view2), inliers2[i])
            self.map.append(map_point)
            self.point_counter += 1

        # logging.debug(f"Number of new points reconstructed: {len(reprojection_error2)}")
        # concatenate the reprojection error images
        vis_repro_image = np.concatenate((vis_repro_image1, vis_repro_image2), axis=1)
        return reprojection_error1, reprojection_error2, vis_repro_image

    def compute_pose_PNP(self, new_view:View):
        """Computes pose of new view using perspective n-point"""

        points_3D_lst = np.zeros((0, 3))
        points_2D_lst = np.zeros((0, 2))
        # find the keypoint matches over latest reconstructed views
        for old_view in self.prev_frames_lst[-self.PNP_PREV_KEYFRAME_NUM:]:
            prev_view_descriptors = old_view.descriptors
            pre_view_keypoints = old_view.keypoints
            matches = self.global_feat_matcher.match(prev_view_descriptors, new_view.descriptors, kpt_uv_a=pre_view_keypoints, kpt_uv_b=new_view.keypoints)

            img_idx = self.get_index_of_view(old_view)
            logging.debug(f'global_feat_matches: {matches.shape} on view {img_idx}')

            for match in matches:
                old_image_idx, old_image_kp_idx, new_image_kp_idx = img_idx, match[0], match[1]
                if (old_image_idx, old_image_kp_idx) in self.point_map:
                    # obtain the 2D point from match
                    point_2D = np.array(new_view.keypoints[new_image_kp_idx]).T.reshape((1, 2))
                    points_2D_lst = np.concatenate((points_2D_lst, point_2D), axis=0)

                    # obtain the 3D point from the point_map
                    point_3D = self.map[self.point_map[(old_image_idx, old_image_kp_idx)]].position
                    points_3D_lst = np.concatenate((points_3D_lst, point_3D), axis=0)

        logging.debug(f'PnP 2d-3d pairs: {points_3D_lst.shape}')
        # compute new view pose 
        _, R, t, _ = cv2.solvePnPRansac(points_3D_lst[:, np.newaxis], points_2D_lst[:, np.newaxis], self.K, None,
                                        confidence=0.99, reprojectionError=self.PNP_REPROJECTION_THRESH, flags=cv2.SOLVEPNP_DLS)
        R, _ = cv2.Rodrigues(R)
        return R, t


    def save_map_points(self):
        """Saves the reconstructed 3D points to ply files using Open3D"""

        number = len(self.prev_frames_lst)
        filename = os.path.join(self.ply_path, str(number) + '_images.ply')
        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.points_3D)
        map_points = np.array([p.position for p in self.map]).reshape((-1, 3))
        pcd.points = o3d.utility.Vector3dVector(map_points)
        o3d.io.write_point_cloud(filename, pcd)

    def bundle_adjustment(self, max_iter=20, verbose=True) -> None:
        ba = BundleAdjustment(self.K)

        # Add cameras
        for kf_idx, kf in enumerate(self.prev_frames_lst):
            ba.add_pose(pose_id=kf_idx, Rcw=kf.R, tcw=kf.t, fixed=(kf_idx < 2))

        # Add points
        for mp_id, mp_3d in enumerate(self.map):
            ba.add_point(point_id=mp_3d.idx, point=mp_3d.position.T)

        # Add observation
        for mp_id, mp_3d in enumerate(self.map):
            mp_id = mp_3d.idx
            for kf_idx, kf_kpt_idx in mp_3d.observations():
                uv_measurement = self.prev_frames_lst[kf_idx].keypoints[kf_kpt_idx]
                _ = ba.add_edge(mp_id, kf_idx, uv_measurement)

        # Optimize
        if verbose:
            before_err = ba.get_edge_costs()
            ba.optimize(max_iterations=max_iter, verbose=True)
            after_err = ba.get_edge_costs()

            print(
                "[IncrementalReconstructor] Bundle Adjustment, before_err: %.2f, after_err: %.2f"
                % (before_err, after_err)
            )
        else:
            ba.optimize(max_iterations=max_iter, verbose=False)

        # Update keyframe pose and map points
        for kf_idx, kf in enumerate(self.prev_frames_lst):
            self.prev_frames_lst[kf_idx].Tcw = ba.get_pose(kf_idx).matrix()

        for mp_id, map_pt in enumerate(self.map):
            mp_id = map_pt.idx
            map_pt.position = ba.get_point(mp_id)

        return ba
    
    def reconstruct(self):
        """Starts the main reconstruction loop for a given set of views and matches"""

        # compute initial pose
        init_img1_dict, init_img2_dict = self.dataloader[0], self.dataloader[1]
        logging.info("Computing baseline pose and reconstructing points")
        self.compute_pose(img1=init_img1_dict['image'],
                          img1_name=init_img1_dict['image_name'], 
                          img2=init_img2_dict['image'], 
                          img2_name=init_img2_dict['image_name'],
                          is_init_views=True)
        logging.info("Mean reprojection error for 0 view is %f", self.errors[0])
        logging.info("Mean reprojection error for 1 view is %f", self.errors[1])
        self.save_map_points()
        logging.debug("Points plotted for %d views", len(self.prev_frames_lst))

        for i in range(2, len(self.dataloader)):

            logging.debug("Computing pose and reconstructing points for view %d", i)
            img_dict = self.dataloader[i]
            self.compute_pose(img1=img_dict['image'],
                              img1_name=img_dict['image_name'])
            logging.info("Mean reprojection error for %d view is %f", i, self.errors[i])
            self.save_map_points()
            logging.debug("Points plotted for %d views", i)

    @property
    def reconstructed_views(self):
        return self.prev_frames_lst
    
    @property
    def reconstructed_map_points(self):
        return np.array([p.position for p in self.map]).reshape((-1, 3))
    
    def save_trajectory(self, output_filepath:str=None):
        """save camera trajectory in the format of TUM RGB-D dataset"""
        
        if output_filepath is None:
            output_filepath = os.path.join(self.image_path, 'est_trajectory.txt')
        with open(output_filepath, 'w') as f:
            for i in range(len(self.prev_frames_lst)):
                Tcw = self.prev_frames_lst[i].Tcw
                Twc = np.linalg.inv(Tcw)
                f.write(f'{i} {" ".join([str(t) for t in Twc[:3, 3]])} {" ".join([str(q) for q in Tcw2qvec(Twc)])}\n')
        