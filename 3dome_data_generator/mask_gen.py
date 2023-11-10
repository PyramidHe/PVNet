import open3d as o3d
import cv2
import yaml
import numpy as np
import argparse
import os


def read_camera_configuration(filepath, width=None, height=None):
    """ Reads yaml file containing camera matrices. Can be downsampled"""
    if height is not None and width is None:
        raise ValueError("Height can't be set if width is None")
    intrinsic_all = []
    extrinsic_all = []

    inv_intrinsic_all = []
    inv_extrinsic_all = []

    width_all = []
    height_all = []
    k0_all = []
    k1_all = []

    intrinsic = np.eye(4, dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    dist_coeffs = np.zeros((2, 1))
    with open(filepath, 'r') as file:
        camera_params = yaml.load(file, Loader=yaml.Loader)
        num_cam = len(camera_params["intrinsics"])

    for i in range(num_cam):
        img_width = camera_params["intrinsics"][i]["img_width"]
        img_height = camera_params["intrinsics"][i]["img_height"]

        if width is None:
            width = img_width
        ratio = img_height / img_width
        if height is None:
            height = int(width * ratio)

        width_factor = width / img_width
        height_factor = height / img_height

        intrinsic[0, 0] = camera_params["intrinsics"][i]["fx"] * width_factor
        intrinsic[1, 1] = camera_params["intrinsics"][i]["fy"] * height_factor
        intrinsic[0, 2] = camera_params["intrinsics"][i]["cx"] * width_factor
        intrinsic[1, 2] = camera_params["intrinsics"][i]["cy"] * height_factor

        k0 = camera_params["intrinsics"][i]["dist_k0"]
        k1 = camera_params["intrinsics"][i]["dist_k1"]
        rotation = np.array(camera_params["extrinsics"][i]["rotation"]["data"], dtype=float)

        if rotation.shape[0] == 3:
            extrinsic[:3, :3], _ = cv2.Rodrigues(rotation)
        else:
            extrinsic[:3, :3] = rotation.reshape(3, 3)
        extrinsic[:3, 3] = np.array(camera_params["extrinsics"][i]["translation"]["data"])

        #  append to output
        intrinsic_all.append(np.copy(intrinsic))
        extrinsic_all.append(np.copy(extrinsic))
        inv_intrinsic_all.append(np.linalg.inv(intrinsic))
        inv_extrinsic_all.append(np.linalg.inv(extrinsic))
        width_all.append(width)
        height_all.append(height)
        k0_all.append(k0)
        k1_all.append(k1)

    intrinsic_all = np.stack(intrinsic_all)
    extrinsic_all = np.stack(extrinsic_all)
    inv_intrinsic_all = np.stack(inv_intrinsic_all)
    inv_extrinsic_all = np.stack(inv_extrinsic_all)
    width_all = np.stack(width_all)
    height_all = np.stack(height_all)

    return {
        "intrinsic": intrinsic_all,
        "extrinsic": extrinsic_all,
        "inv_intrinsic": inv_intrinsic_all,
        "inv_extrinsic": inv_extrinsic_all,
        "width": width_all,
        "height": height_all,
        "k0": k0_all,
        "k1": k1_all
    }


def generate_masks_from_mesh(conf_filepath, mesh_filepath, output_folder="output"):
    camera_opt = read_camera_configuration(conf_filepath)
    mesh = o3d.io.read_triangle_mesh(mesh_filepath)
    os.makedirs(output_folder, exist_ok=True)
    pose_folder = os.path.join(output_folder, 'pose')
    mask_folder = os.path.join(output_folder, 'mask')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(pose_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    base_intrinsic = camera_opt['intrinsic'][0, :3, :3]
    np.savetxt(os.path.join(output_folder,'camera.txt'), base_intrinsic)

    for i in range(len(camera_opt['intrinsic'])):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_opt["width"][i], camera_opt["height"][i],
                                                       camera_opt['intrinsic'][i, :3, :3])
        extrinsic_matrix = camera_opt['extrinsic'][i]
        np.save(os.path.join(pose_folder, 'pose' + str(i)), extrinsic_matrix[:3, :])
        render = o3d.visualization.rendering.OffscreenRenderer(camera_opt["width"][i], camera_opt["height"][i])

        # background color
        render.scene.set_background([0.0, 0.0, 0.0, 1.0])
        # mesh color
        mesh.paint_uniform_color([1.0, 1.0, 1.0])
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]
        mtl.shader = "defaultUnlit"

        render.scene.add_geometry("mesh", mesh, mtl)

        # render the scene with respect to the camera
        render.setup_camera(intrinsic, extrinsic_matrix)
        img_o3d = render.render_to_image()

        o3d.io.write_image(os.path.join(mask_folder, "mask" + str(i)+".png"), img_o3d, 9)
        # must delete to work inside loop
        del render

