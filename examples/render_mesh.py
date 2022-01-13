import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils3d.mesh.io import read_mesh
from utils3d.utils.pyrender import PyRenderer, get_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tool", type=str, choices=["py", "nv"], help="Which tool for rendering."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../data/Squirrel_visual.obj",
        help="Input point cloud path.",
    )
    parser.add_argument("--norm", action="store_true", help="Normalize data.")
    args = parser.parse_args()

    if args.tool == "py":
        renderer = PyRenderer()
    else:
        pass

    mesh = read_mesh(args.input)
    center = mesh.bounds.mean(0)
    scale = np.sqrt(((mesh.bounds[1] - mesh.bounds[0]) ** 2).sum())

    surface_point_cloud = mesh.sample(2048)

    camera_pose = get_pose(scale * 1, center=center, ax=np.pi / 3, az=np.pi / 3)
    light_pose = get_pose(scale * 2, center=center, ax=np.pi / 3, az=np.pi / 3)
    img, depth = renderer.render_mesh(mesh, camera_pose=camera_pose, light_pose=light_pose)

    pc_light_pose = get_pose(scale * 4, center=center, ax=np.pi / 3, az=np.pi / 3)
    pc_img, _ = renderer.render_pointcloud(
        surface_point_cloud,
        camera_pose=camera_pose,
        light_pose=pc_light_pose,
        radius=scale * 0.01,
        colors=[102, 204, 102, 102],
    )

    fig, axs = plt.subplots(1, 3, dpi=150)
    axs[0].imshow(img)
    axs[1].imshow(pc_img)
    axs[2].imshow(depth)
    plt.show()


if __name__ == "__main__":
    main()
