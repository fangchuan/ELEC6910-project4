import numpy as np
import ipyvolume as ipv
# from core_3dv.camera_operator import camera_center_from_Tcw, translation_from_center

def ipv_prepare(ipv, dark_background=True):
    """
        Prepare the ipv instance

    Args:
        ipv (object): ipv instance
        dark_background (bool): toggle to show dark background

    """
    ipv.clear()
    if dark_background is True:
        ipv.style.set_style_dark()


def ipv_draw_point_cloud(ipv, pts, colors, pt_size=10):
    """
        Draw point clouds

    Args:
        ipv (object): ipv instance
        pts (array): point cloud array, dim (num_points, 3)
        colors (array): color of points, dim (num_points, 3)
        pt_size (float): size of point

    """
    pts = pts.reshape((-1, 3))
    colors = colors.reshape((-1, 3))
    assert colors.shape[0] == pts.shape[0]
    ipv.scatter(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], color=colors.reshape(-1, 3), marker='sphere', size=pt_size)


def ipv_draw_pose_3d(ipv, R: np.ndarray, t: np.ndarray, color='red', camera_scale=0.15):
    """
        Draw camera poses in ipv

    Args:
        ipv (object): ipv instance
        R (array): rotation matrix, dim (3x4)
        t (array): translation vec, dim (3, )
        color (str): the color of camera pose
        camera_scale (float): the scale of camera pose

    """
    # camera obj
    cam_points = np.array([[0, 0, 0],
                           [-1, -1, 1.5],
                           [1, -1, 1.5],
                           [1, 1, 1.5],
                           [-1, 1, 1.5]])
    # axis indicators
    axis_points = np.array([[-0.5, 1, 1.5],
                            [0.5, 1, 1.5],
                            [0, 1.2, 1.5],
                            [1, -0.5, 1.5],
                            [1, 0.5, 1.5],
                            [1.2, 0, 1.5]])
    # transform camera objs...
    cam_points = (camera_scale * cam_points - t).dot(R)
    axis_points = (camera_scale * axis_points - t).dot(R)

    x = cam_points[:, 0]
    y = cam_points[:, 1]
    z = cam_points[:, 2]
    cam_wire_draw_order = np.asarray([(0, 1, 4, 0),  # left
                                      (0, 3, 2, 0),  # right
                                      (0, 4, 3, 0),  # top
                                      (0, 2, 1, 0)])  # bottom
    x = np.take(x, cam_wire_draw_order)
    y = np.take(y, cam_wire_draw_order)
    z = np.take(z, cam_wire_draw_order)

    axis_triangles = np.asarray([(3, 5, 4,),  # x-axis indicator
                                 (0, 1, 2)])  # y-axis indicator

    ipv.plot_wireframe(x, y, z, color=color, wrapx=True)
    ipv.plot_trisurf(x=axis_points[:, 0], y=axis_points[:, 1], z=axis_points[:, 2],
                     triangles=axis_triangles,
                     color=color)

def camera_center_from_Tcw(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    """
        Compute the camera center from extrinsic matrix (world -> camera)

    Args:
        Rcw (array): Rotation matrix with dimension of (3x3), world -> camera
        tcw (array): translation of the camera, world -> camera

    Returns:
        camera center in world coordinate system.

    """
    # C = -Rcw' * t
    C = -np.dot(Rcw.transpose(), tcw)
    return C


def translation_from_center(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
        Convert center to translation vector, C = -R^T * t -> t = -RC

    Args:
        R (array): Rotation matrix with dimension of (3x3)
        C (array): center of the camera

    Returns:
        translation vector

    """
    t = -np.dot(R, C)
    return t

def move_traj_2_origin(Tcws: list, anchor_center=None):
    """ Move the trajectory to origin defined by `anchor_center`
    """
    if anchor_center is None:
        anchor_Tcw = Tcws[len(Tcws) // 2]
        anchor_center = camera_center_from_Tcw(anchor_Tcw[:3, :3], anchor_Tcw[:3, 3])

    new_aligned_Tcws = []
    for tcw in Tcws:
        new_center = camera_center_from_Tcw(tcw[:3, :3], tcw[:3, 3]) - anchor_center
        new_translation = translation_from_center(tcw[:3, :3], new_center)
        new_Tcw = np.eye(4)[:3, :]
        new_Tcw[:3, :3] = tcw[:3, :3]
        new_Tcw[:3, 3] = new_translation[:3]
        new_aligned_Tcws.append(new_Tcw)
    return new_aligned_Tcws, anchor_center


def point_max_dim(pts: np.ndarray):
    """ Return the maximum dimension of point cloud, used for visualize 3D points
    """
    pt_dim = pts.max(axis=0) - pts.min(axis=0)
    return pt_dim.max()

class iPV3DVisualizer:
    """ Visualize the 3D space with ipvolume
    """

    def __init__(self, dim_hw=(1024, 1024), dark_bg=True):
        """
        Args:
            dark_bg (bool): use the dark background for 3D visualization
        """
        ipv.clear()

        # set dimension of WebGL canvas
        self.fig = ipv.figure(width=dim_hw[1], height=dim_hw[0])

        if dark_bg is True:
            ipv.style.set_style_dark()

        self.trajs = []
        self.pts = []

    def add_trajectory(self, traj_Tcws: list, color='blue', size=1.0, anchor=False):
        """
            Add trajectory

        Args:
            traj_Tcws (list): camera poses of trajectory
            color (str): the color of camera
            size (float): the size of camera
            anchor (bool): use as anchor to align other elements in 3D space and move to origin.

        """
        trj_dict = dict()
        trj_dict['Tcws'] = traj_Tcws
        trj_dict['color'] = color
        trj_dict['camera_size'] = size
        trj_dict['anchor'] = anchor
        self.trajs.append(trj_dict)

    def add_point_cloud(self, pts3d_pos, pts3d_color=None, pt_size=1.0):
        """
            Add point cloud

        Args:
            pts3d_pos (array): the position of point in 3D space
            pts3d_color (array): the color of point in 3D space (RGB, 0-1)
            pt_size (float): point size

        """
        pts_dict = dict()
        pts_dict['pts3d_pos'] = pts3d_pos
        pts_dict['pts3d_color'] = pts3d_color if pts3d_color is not None else np.ones_like(pts3d_pos)
        pts_dict['pts3d_size'] = pt_size
        self.pts.append(pts_dict)

    def show(self):
        ipv.style.use(['minimal'])
        anchor_Tcw = self.trajs[0]['Tcws'][len(self.trajs[0]['Tcws']) // 2]
        anchor_trj = self.trajs[0]['Tcws']
        for trj in self.trajs:
            if trj['anchor']:
                anchor_trj = trj['Tcws']
                anchor_Tcw = anchor_trj[len(anchor_trj) // 2] if len(anchor_trj) > 2 else anchor_trj[0]
                break
        anchor_center = camera_center_from_Tcw(anchor_Tcw[:3, :3], anchor_Tcw[:3, 3])
        print(anchor_center)
        
        # visualize trajectory
        for i, trj in enumerate(self.trajs):
            tcws, _ = move_traj_2_origin(trj['Tcws'], anchor_center=anchor_center)
            color = trj['color']

            camera_size = trj['camera_size']
            for j, tcw in enumerate(tcws):
                if isinstance(color, list) and len(color) == len(tcws):
                    c = color[j]
                else:
                    c = color
                if j == len(tcws) - 1:
                    c = 'red'

                ipv_draw_pose_3d(ipv, tcw[:3, :3], tcw[:3, 3], color=c, camera_scale=camera_size)

        # draw point cloud
        for pts in self.pts:
            pts3d_pos = pts['pts3d_pos'] - anchor_center
            pts3d_color = pts['pts3d_color']
            pt_size = pts['pts3d_size']
            ipv_draw_point_cloud(ipv, pts3d_pos, colors=pts3d_color, pt_size=pt_size)

        scale = point_max_dim(np.asarray([camera_center_from_Tcw(tcw[:3, :3], tcw[:3, 3]) for tcw in anchor_trj]))
        ipv.xyzlim(scale)
        ipv.show()
    
    def get_view(self):
        return ipv.view()
    
    def set_view(self, view):
        ipv.view(view[0], view[1], view[2])

    def save_fig(self, path):
        ipv.savefig(path)