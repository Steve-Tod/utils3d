"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-12
 @desc Pyrender utils
"""
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import pyrender
import trimesh
from PIL import Image

from utils3d.utils.transform import Rotation, Transform


def get_pose(distance, center=np.zeros(3), ax=0, ay=0, az=0):
    """generate camera pose from distance, center and euler angles

    Args:
        distance (float): distance from camera to center
        center (np.ndarray, optional): the look at center. Defaults to np.zeros(3).
        ax (float, optional): euler angle x. Defaults to 0.
        ay (float, optional): euler angle around y axis. Defaults to 0.
        az (float, optional): euler angle around z axis. Defaults to 0.

    Returns:
        np.ndarray: camera pose of 4*4 numpy array
    """
    rotation = Rotation.from_euler("xyz", (ax, ay, az))
    vec = np.array([0, 0, distance])
    translation = rotation.as_matrix().dot(vec) + center
    camera_pose = Transform(rotation, translation).as_matrix()
    return camera_pose


class PyRenderer:
    def __init__(
        self,
        resolution=(640, 480),
        camera_kwargs={"yfov": np.pi / 3.0},
        light_kwargs={"color": np.ones(3), "intensity": 3},
    ):
        """Renderer init function

        Args:
            resolution (tuple, optional): render resolution, . Defaults to (640, 480).
            camera_kwargs (dict, optional): camera properties. Defaults to {'yfov': np.pi/ 3.0}.
            light_kwargs (dict, optional): light properties. Defaults to {'color': np.ones(3), 'intensity': 3}.
        """
        self.renderer = pyrender.OffscreenRenderer(*resolution)
        self.camera = pyrender.PerspectiveCamera(**camera_kwargs)
        self.light = pyrender.SpotLight(**light_kwargs)

    def render_mesh(self, scene_or_mesh, camera_pose, light_pose):
        """render a mesh or a scene

        Args:
            scene_or_mesh (trimesh.Trimesh or trimesh.Scene): the scene or mesh to render
            camera_pose (np.ndarray): camera transformation
            light_pose (np.ndarray): light transformation

        Returns:
            PIL.Image: rendered rgb image
            np.ndarray: rendered depth image
        """
        if isinstance(scene_or_mesh, trimesh.Trimesh):
            scene = trimesh.Scene(scene_or_mesh)
        else:
            scene = scene_or_mesh

        r_scene = pyrender.Scene()
        for mesh in scene.geometry.values():
            o_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            r_scene.add(o_mesh)
        r_scene.add(self.camera, name="camera", pose=camera_pose)
        r_scene.add(self.light, name="light", pose=light_pose)
        color_img, depth = self.renderer.render(r_scene)
        return Image.fromarray(color_img), depth

    def render_pointcloud(
        self, xyz, camera_pose, light_pose, radius=0.005, colors=[102, 102, 102, 102]
    ):
        """render a point cloud

        Args:
            xyz (np.ndarray): point cloud to render.
            camera_pose (np.ndarray): camera
            light_pose ([type]): [description]
            radius (float, optional): radius of each ball representing the point. Defaults to 0.005.
            colors (list or np.ndarray, optional): colors of the points. Defaults to [102, 102, 102, 102].

        Returns:
            PIL.Image: rendered rgb image
            np.ndarray: rendered depth image
        """
        r_scene = pyrender.Scene()
        sm = trimesh.creation.uv_sphere(radius=radius)
        if colors is None:
            sm.visual.vertex_colors = [0.4, 0.4, 0.4, 0.8]
        else:
            sm.visual.vertex_colors = colors
        tfs = np.tile(np.eye(4), (len(xyz), 1, 1))
        tfs[:, :3, 3] = xyz
        o_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        r_scene.add(o_mesh)
        r_scene.add(self.camera, name="camera", pose=camera_pose)
        r_scene.add(self.light, name="light", pose=light_pose)
        color_img, depth = self.renderer.render(r_scene)
        return Image.fromarray(color_img), depth
