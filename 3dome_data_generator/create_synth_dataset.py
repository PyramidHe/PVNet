import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import glob
import os
import shutil
import random
import cv2
import open3d as o3d
from tqdm import tqdm
from mask_gen import generate_masks_from_mesh


def create_naive_dataset(images, masks, bkgs, poses, camera, width, height, out, num_out_images=1500,
                         image_ext='.jpg', mask_ext='.png'):

    to_tensor = transforms.ToTensor()
    transforms_to_apply = torch.nn.Sequential(
        transforms.Grayscale(3),
        transforms.Resize((height, width), antialias=True),
        transforms.GaussianBlur(5, sigma=(0.5, 2.0)),
        transforms.ColorJitter(.7, .7, .7, 0.5),
        transforms.RandomAutocontrast(),
        transforms.RandomAdjustSharpness(5)
    )

    resizer = torch.nn.Sequential(
        transforms.Resize((height, width), antialias=True),
    )
    image_list = []
    mask_list = []
    bkgs_list = []
    poses_list = []

    for filename in glob.glob(os.path.join(images, '*')):
        image_list.append(filename)

    for filename in glob.glob(os.path.join(masks, '*')):
        mask_list.append(filename)

    for filename in glob.glob(os.path.join(bkgs, '*')):
        bkgs_list.append(filename)

    for filename in glob.glob(os.path.join(poses, '*')):
        poses_list.append(filename)

    image_list = sorted(image_list)
    mask_list = sorted(mask_list)
    poses_list = sorted(poses_list)

    out_imgs = os.path.join(out, 'rgb')
    out_masks = os.path.join(out, 'mask')
    out_poses = os.path.join(out, 'pose')

    if not os.path.exists(out_imgs):
        os.makedirs(out_imgs)
    if not os.path.exists(out_masks):
        os.makedirs(out_masks)
    if not os.path.exists(out_poses):
        os.makedirs(out_poses)

    image = Image.open(image_list[0])
    image = to_tensor(image)
    image = image[:3, :, :]
    camera = np.loadtxt(camera)
    camera[0, :] = camera[0, :] * width / image.shape[2]
    camera[1, :] = camera[1, :] * height / image.shape[1]
    np.savetxt(os.path.join(out, 'camera.txt'), camera)

    for index in tqdm(range(num_out_images)):
        image = Image.open(image_list[index % len(image_list)])
        mask = Image.open(mask_list[index % len(image_list)])
        image = to_tensor(image)
        image = image[:3, :, :]
        background = Image.open(random.choice(bkgs_list))
        background = to_tensor(background)
        mask = to_tensor(mask)
        pose = np.load(poses_list[index % len(image_list)])


        image = resizer(image)
        mask = resizer(mask)
        background = resizer(background)

        image = (image * mask + (1 - mask) * background)
        image = transforms_to_apply(image)
        image_to_save = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_to_save = (mask.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_imgs, str(index) + image_ext), image_to_save)
        cv2.imwrite(os.path.join(out_masks,  str(index) + mask_ext), mask_to_save)
        np.save(os.path.join(out_poses, 'pose' + str(index) + '.npy'), pose)

    return


parser = argparse.ArgumentParser()

parser.add_argument('--conf', type=str, required=True, help='camera configuration file')
parser.add_argument('--mesh', type=str, required=True, help='mesh file')
parser.add_argument('--images', type=str, required=True, help='path to folder containing masked training images')
parser.add_argument('--out', type=str, required=True, help='path to folder containing processed dataset')
parser.add_argument('--bkgs', type=str, required=True, help='path to folder containing background images')
parser.add_argument('--resolution', default='640x480', help='resolution')
parser.add_argument('--num_out_images', type=int, default=1500, help='resolution')


args = parser.parse_args()

if __name__ == '__main__':
    width, height = args.resolution.split("x")
    width, height = int(width), int(height)
    generate_masks_from_mesh(args.conf, args.mesh, 'tmp')
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    print('Generating augmented dataset')
    create_naive_dataset(args.images, os.path.join('tmp', 'mask'), args.bkgs, os.path.join('tmp', 'pose'),
                         os.path.join('tmp', 'camera.txt'), width, height, args.out, args.num_out_images)
    o3d.io.write_triangle_mesh(os.path.join(args.out, 'model.ply'), mesh, write_ascii=True)
    shutil.rmtree('tmp')


