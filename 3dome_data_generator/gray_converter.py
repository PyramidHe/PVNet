import cv2
import numpy as np
import argparse
import os



parser = argparse.ArgumentParser(description="Generate masks from mesh and camera configuration file")
parser.add_argument("--folder", help="custom pvnet folder")
parser.add_argument('--resolution', default='640x480', help='resolution')

args = parser.parse_args()

if __name__ == "__main__":
    images = os.listdir(args.folder)
    width, height = args.resolution.split("x")
    width, height = int(width), int(height)
    for image in images:
        img = cv2.imread(os.path.join(args.folder, image))
        img = np.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=-1)
        img = cv2.resize(img, (width, height))
        cv2.imwrite(os.path.join(args.folder, image[:-4]+'.jpg'), img)
