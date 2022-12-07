'''
kitti_360_iterator.py

Description:

Date: 12/07/2022
'''

# Imports
# from project import CameraPerspective
from project import CameraFisheye

import os
import numpy as np

def main():
    pass

# Base Runner
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    # Default Prameters:
    # 1. Set KITTI 360 Path (Follow steps in README.md)
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        raise RuntimeError("Kitti 360 Path Not Set! Please set the kitti 360 path. Follow directions in README.md")
    
    cam_id = 2
    sequence = '2013_05_28_drive_0000_sync'

    # fisheye
    camera = CameraFisheye(kitti360Path, sequence, cam_id)
    print(camera.fi)

    # loop over frames
    # Set frame limit to 100
    frame_limit = 100
    for frame in camera.frames:
        # fisheye
        image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rgb', '%010d.png'%frame)
        if not os.path.isfile(image_file):
            print('Missing %s ...' % image_file)
            continue

        print(image_file)
        image = cv2.imread(image_file)
        plt.imshow(image[:,:,::-1])

        if frame>=frame_limit:
            break

    exit()