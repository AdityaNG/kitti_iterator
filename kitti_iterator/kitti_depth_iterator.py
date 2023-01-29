import sys
import os
import pathlib

from multiprocessing.pool import Pool

import time
import yaml
import numpy as np
import pandas as pd
import scipy

from torch.utils.data import Dataset
from torch.multiprocessing import Process, Queue, set_start_method
import torch

import pickle
import cv2
import itertools
import glob
import tqdm
import open3d as o3d

from .ground_removal import Processor

from .helper import *
from .kitti_raw_iterator import KittiRaw

plot3d = False
plot2d = False
point_cloud_array = None
if __name__ == '__main__':
    if plot3d:
        set_start_method('spawn')
        point_cloud_array = Queue()

class KittiDepth(KittiRaw):

    def __init__(self, 
        kitti_depth_base_path="/home/shared/kitti_depth/",
        kitti_raw_base_path="/home/shared/Kitti",
        date_folder="2011_09_26",
        sub_folder="2011_09_26_drive_0001_sync",
        transform=dict(),
        grid_size = (2.0, 2.0, 2.0),
        scale = 1.0,
        sigma = None,
        gaus_n = 4,
        ground_removal=False,
        compute_trajectory=False,
        invalidate_cache=True,
        scale_factor=1.0, plot_3D_x=250, plot_3D_y=500, num_features=5000
    ) -> None:
        super(KittiDepth, self).__init__(
            kitti_raw_base_path=kitti_raw_base_path,
            date_folder=date_folder,
            sub_folder=sub_folder,
            transform=transform,
            grid_size = grid_size,
            scale = scale,
            sigma = sigma,
            gaus_n = gaus_n,
            ground_removal=ground_removal,
            compute_trajectory=compute_trajectory,
            invalidate_cache=invalidate_cache,
            scale_factor=scale_factor, plot_3D_x=plot_3D_x, plot_3D_y=plot_3D_y, num_features=num_features
        )
        self.kitti_depth_path = os.path.join(kitti_depth_base_path, 'train', sub_folder, 'proj_depth', 'groundtruth')
        self.depth_02_path = os.path.join(self.kitti_depth_path, "image_02")
        self.depth_03_path = os.path.join(self.kitti_depth_path, "image_03")

        self.img_list = sorted(os.listdir(self.depth_02_path))
        self.img_list = list(map(lambda x: x.split(".png")[0], self.img_list))
        self.index = 0

        self.frame_count = len(self)

        self.intrinsics = self.o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            intrinsic_matrix=self.intrinsic_mat
        ) 

    def __getitem__(self, index):
        id = self.img_list[index]
        depth_02 = os.path.join(self.depth_02_path, id + ".png")
        depth_03 = os.path.join(self.depth_03_path, id + ".png")
        
        image_00 = os.path.join(self.image_00_path, 'data', id + ".png")
        image_01 = os.path.join(self.image_01_path, 'data', id + ".png")
        image_02 = os.path.join(self.image_02_path, 'data', id + ".png")
        image_03 = os.path.join(self.image_03_path, 'data', id + ".png")
        velodyine_points = os.path.join(self.velodyne_points_path, 'data', id + ".bin")
        
        assert os.path.exists(depth_02), depth_02
        assert os.path.exists(depth_03), depth_03

        assert os.path.exists(image_00), image_00
        assert os.path.exists(image_01), image_01
        assert os.path.exists(image_02), image_02
        assert os.path.exists(image_03), image_03
        assert os.path.exists(velodyine_points), velodyine_points

        depth_02_raw = cv2.imread(depth_02)
        depth_03_raw = cv2.imread(depth_03)

        image_00_raw = cv2.imread(image_00)
        image_01_raw = cv2.imread(image_01)
        image_02_raw = cv2.imread(image_02)
        image_03_raw = cv2.imread(image_03)
        
        x, y, w, h = self.roi_00
        image_00 = cv2.undistort(image_00_raw, self.K_00, self.D_00, None, self.new_K_00)
        image_00 = image_00[y:y+h, x:x+w]

        x, y, w, h = self.roi_01
        image_01 = cv2.undistort(image_01_raw, self.K_01, self.D_01, None, self.new_K_01)
        image_01 = image_01[y:y+h, x:x+w]

        x, y, w, h = self.roi_02
        image_02 = cv2.undistort(image_02_raw, self.K_02, self.D_02, None, self.new_K_02)
        image_02 = image_02[y:y+h, x:x+w]

        x, y, w, h = self.roi_03
        image_03 = cv2.undistort(image_03_raw, self.K_03, self.D_03, None, self.new_K_03)
        image_03 = image_03[y:y+h, x:x+w]


        # velodyine_points = np.fromfile(velodyine_points, dtype=np.float32)
        # velodyine_points = np.reshape(velodyine_points, (velodyine_points.shape[0]//4, 4))
        velodyine_points = np.fromfile(velodyine_points, dtype=np.float32).reshape(-1, 4)[:,:3]

        if self.ground_removal:
            velodyine_points = velodyine_points * np.array([1.0,1.0,-1.0]) # revert the z axis
            velodyine_points = self.process(velodyine_points)
            velodyine_points = velodyine_points * np.array([1.0,1.0,-1.0]) # revert the z axis
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image_02_raw), o3d.geometry.Image(depth_02_raw),
            convert_rgb_to_intensity=False
        )
        occupancy_grid_pcd = self.o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.intrinsics
        )
        
        occupancy_grid_pcd.remove_non_finite_points()
        occupancy_grid_data = o3d.geometry.VoxelGrid.create_from_point_cloud(occupancy_grid_pcd,
                                                                voxel_size=0.0005)

        # P_rect = self.calib_cam_to_cam['P_rect_00'].reshape(3, 4)[:3,:3]
        # image_points = self.transform_points_to_image_space(velodyine_points, self.roi_00, self.K_00, self.R_00, self.T_00, P_rect, color_fn=depth_color)
        # image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
        # dilatation_size = 3
        # dilation_shape = cv2.MORPH_ELLIPSE
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        #                                 (dilatation_size, dilatation_size))
        # depth_image_00 = cv2.dilate(image_points, element)

        # P_rect = self.calib_cam_to_cam['P_rect_01'].reshape(3, 4)[:3,:3]
        # image_points = self.transform_points_to_image_space(velodyine_points, self.roi_01, self.K_01, self.R_01, self.T_01, P_rect, color_fn=depth_color)
        # image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
        # dilatation_size = 3
        # dilation_shape = cv2.MORPH_ELLIPSE
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        #                                 (dilatation_size, dilatation_size))
        # depth_image_01 = cv2.dilate(image_points, element)

        # P_rect = self.calib_cam_to_cam['P_rect_02'].reshape(3, 4)[:3,:3]
        # image_points = self.transform_points_to_image_space(velodyine_points, self.roi_02, self.K_02, self.R_02, self.T_02, P_rect, color_fn=depth_color)
        # image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
        # dilatation_size = 3
        # dilation_shape = cv2.MORPH_ELLIPSE
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        #                                 (dilatation_size, dilatation_size))
        # depth_image_02 = cv2.dilate(image_points, element)

        # P_rect = self.calib_cam_to_cam['P_rect_03'].reshape(3, 4)[:3,:3]
        # image_points = self.transform_points_to_image_space(velodyine_points, self.roi_03, self.K_03, self.R_03, self.T_03, P_rect, color_fn=depth_color)
        # image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
        # dilatation_size = 3
        # dilation_shape = cv2.MORPH_ELLIPSE
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        #                                 (dilatation_size, dilatation_size))
        # depth_image_03 = cv2.dilate(image_points, element)

        data = {
            'image_00': image_00, 
            'image_01': image_01, 
            'image_02': image_02, 
            'image_03': image_03,
            'image_00_raw': image_00_raw, 
            'image_01_raw': image_01_raw, 
            'image_02_raw': image_02_raw, 
            'image_03_raw': image_03_raw,
            'roi_00': self.roi_00,
            'roi_01': self.roi_01,
            'roi_02': self.roi_02,
            'roi_03': self.roi_03,
            'K_00': self.K_00,
            'K_01': self.K_01,
            'K_02': self.K_02,
            'K_03': self.K_03,
            
            'R_00': self.R_00,
            'R_01': self.R_01,
            'R_02': self.R_02,
            'R_03': self.R_03,

            'T_00': self.T_00,
            'T_01': self.T_01,
            'T_02': self.T_02,
            'T_03': self.T_03,

            'calib_cam_to_cam': self.calib_cam_to_cam,
            'calib_imu_to_velo': self.calib_imu_to_velo,
            'calib_velo_to_cam': self.calib_velo_to_cam,

            'occupancy_grid': occupancy_grid_data,

            'depth_image_02': depth_02_raw,
            'depth_image_03': depth_03_raw,
        }
        for key in self.transform:
            data[key] = self.transform[key](data[key])
        return data

def get_kitti_tree(kitti_raw_base_path):
    date_folder_list = list(filter(os.path.isdir, glob.glob(os.path.join(kitti_raw_base_path, '*'))))
    date_folder_list = list(filter(lambda i: len(i.split('_'))==3, date_folder_list))
    kitti_tree = dict()
    for date_folder in date_folder_list:
        date_id = date_folder.split('/')[-1]
        # print(date_id)
        sub_folder_list = list(filter(os.path.isdir, glob.glob(os.path.join(date_folder, '*'))))
        sub_folder_list = list(filter(lambda i: len(i.split('/')[-1].split('_'))==6, sub_folder_list))
        sub_folder_list = list(map(lambda i: i.split('/')[-1], sub_folder_list))

        kitti_tree[date_id] = sub_folder_list
        # print(sub_folder_list)
    return kitti_tree

def get_kitti_raw(**kwargs):
    kitti_raw_base_path=kwargs['kitti_raw_base_path']
    kitti_tree = get_kitti_tree(kitti_raw_base_path)
    kitti_raw = []
    for date_folder in kitti_tree:
        for sub_folder in kitti_tree[date_folder]:
            kitti_raw.append(
                KittiRaw(
                    # kitti_raw_base_path=kitti_raw_base_path,
                    date_folder=date_folder,
                    sub_folder=sub_folder,
                    **kwargs
                )
            )
    return kitti_raw

def main(point_cloud_array=point_cloud_array):

    # import open3d as o3d
    # k_raw = KittiRaw(
    #     kitti_raw_base_path=os.path.expanduser("~/Datasets/kitti/raw/"),
    #     date_folder="2011_09_26",
    #     sub_folder="2011_09_26_drive_0002_sync",
    #     compute_trajectory=True,
    #     scale_factor=1.0,
    #     num_features=5000,
    #     invalidate_cache=True
    # )

    # return

    if plot3d:
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # pcd = o3d.geometry.PointCloud()
        # vis.add_geometry(pcd)
        pass

    # k_raw = KittiRaw()
    # grid_size = (751/25.0, 1063/25.0, 135/25.0)
    # grid_scale = (2.0, 2.0, 4.0)
    # grid_size = (138/grid_scale[0], 99/grid_scale[1], 22/grid_scale[2])

    grid_scale = (5.0, 5.0, 5.0)
    grid_size = (502/grid_scale[0], 182/grid_scale[1], 38/grid_scale[2])

    k_raw = KittiRaw(
        # kitti_raw_base_path="kitti_raw_mini",
        # date_folder="2011_09_26",
        # sub_folder="2011_09_26_drive_0001_sync",
        grid_size = grid_size,
        scale = grid_scale,
        # sigma = 1.0,
        sigma = None,
        gaus_n=1,
        ground_removal=False
    )
    
    print('Starting timer')
    import time
    start_time = time.time()
    dat = k_raw[0]
    print(dat['occupancy_grid'].shape)
    print(time.time()-start_time)
    
    # k_raw = KittiRaw(
    #     kitti_raw_base_path=os.path.expanduser("~/Datasets/kitti/raw/"),
    #     grid_size = (150.0, 74.0, 17.0),
    #     scale = 1.0,
    #     sigma = 1.0,
    #     gaus_n=1
    # )

    print("Found", len(k_raw), "images ")
    for index in range(len(k_raw)):
        data = k_raw[index]
        image_02 = data['image_02_raw']
        velodyine_points = data['velodyine_points']
        velodyine_points_camera = data['velodyine_points_camera']
        occupancy_mask_2d = data['occupancy_mask_2d']
        occupancy_grid  = data['occupancy_grid']
        img_id = '_00'
        roi = data['roi'+img_id]
        R_cam = data['R'+img_id]
        T_cam = data['T'+img_id]

        calib_cam_to_cam = data['calib_cam_to_cam']
        calib_imu_to_velo = data['calib_imu_to_velo']
        calib_velo_to_cam = data['calib_velo_to_cam']

        P_rect = calib_cam_to_cam['P_rect' + img_id].reshape(3, 4)[:3,:3]
        # P_cam = calib_cam_to_cam['P' + img_id].reshape(3, 4)[:3,:3]
        R_rect = calib_cam_to_cam['R_rect' + img_id]
        K = data['K'+img_id]
        
        if plot2d:
            # x, y, w, h = roi
            w, h = k_raw.width, k_raw.height
            # img_input = data['image'+img_id+'_raw']
            img_input = data['image'+img_id + '_raw']
            print('img_input.shape', img_input.shape)
            img_input = cv2.resize(img_input, (w, h))
            print('img_input.shape', img_input.shape)
            print('img_input.dtype', img_input.dtype)
            
            # image_points = k_raw.transform_points_to_image_space(velodyine_points, roi, data['K'+img_id], R_cam, T_cam, P_rect, color_fn=depth_color)
            image_points = k_raw.transform_points_to_image_space(velodyine_points, roi, K, R_cam, T_cam, P_rect, color_fn=depth_color)
            print('image_points.shape', image_points.shape)
            print('image_points.dtype', image_points.dtype)
            # image_points = k_raw.transform_occupancy_grid_to_image_space(occupancy_grid, roi, data['K'+img_id], R_cam, T_cam, P_rect)
        

            image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


            dilatation_size = 4
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))
            image_points_gt = cv2.dilate(image_points, element)

            image_overlay = cv2.addWeighted(image_points_gt.astype(np.uint8), 0.5, img_input, 0.5, 0.0)

            cv2.imshow('img_input', img_input)
            cv2.imwrite('tmps/' + str(index) + 'img_input.png', img_input)
            cv2.imshow('image_points_gt', cv2.applyColorMap(cv2.normalize(image_points_gt - np.min(image_points_gt.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_VIRIDIS))
            cv2.imwrite('tmps/' + str(index) + 'image_points_gt.png', cv2.applyColorMap(cv2.normalize(image_points_gt - np.min(image_points_gt.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_VIRIDIS))

            image_points = k_raw.transform_occupancy_grid_to_image_space(occupancy_grid, roi, data['K'+img_id], R_cam, T_cam, P_rect)            
            image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))
            image_points_grid = cv2.dilate(image_points, element)

            # print("data['depth_image'].shape", data['depth_image'].shape)
            # print("data['depth_image'].dtype", data['depth_image'].dtype)
            cv2.imshow('depth_image_00', data['depth_image_00'])
            cv2.imshow('depth_image_01', data['depth_image_01'])
            cv2.imshow('depth_image_02', data['depth_image_02'])
            cv2.imshow('depth_image_03', data['depth_image_03'])
            # cv2.imshow('img_input', img_input)
            # cv2.imshow('image_overlay', image_overlay)
            
            # cv2.imshow('image_points_grid', cv2.applyColorMap(cv2.normalize(image_points_grid - np.min(image_points_grid.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_VIRIDIS))
            cv2.imwrite('tmps/' + str(index) + 'image_points_grid.png', cv2.applyColorMap(cv2.normalize(image_points_grid - np.min(image_points_grid.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_VIRIDIS))
            
            print(compute_errors(img_input, image_points_grid))
            print(compute_errors(data['depth_image_00'], data['depth_image_00']))

            key = cv2.waitKey(5000)
            if key == ord('q'):
                return

        if plot3d:
            # o3d.visualization.draw_geometries([voxel_grid])
            # return
            # print("Before transform_occupancy_grid_to_points")

            print('Starting transform_occupancy_grid_to_points timer')
            start_time = time.time()
            final_points = k_raw.transform_occupancy_grid_to_points(occupancy_grid, threshold=0.5, skip=1)
            # final_points = k_raw.transform_occupancy_grid_to_points_world_coords(occupancy_grid, threshold=0.5, skip=1)
            # final_points = k_raw.transform_occupancy_grid_to_grid_points(occupancy_grid, threshold=0.5, skip=1)
            
            # final_points = k_raw.transform_occupancy_grid_to_points_list_comp(occupancy_grid, threshold=0.001, skip=int(3))
            # final_points = velodyine_points_camera
            print(time.time() - start_time)
            
            # final_points = velodyine_points_camera
            # final_points = velodyine_points
            
            # print("k_raw.occupancy_shape", k_raw.occupancy_shape)
            print("occupancy_grid.shape", occupancy_grid.shape)
            print("final_points.shape", final_points.shape)
            print(np.sum(occupancy_grid))

            if type(point_cloud_array)!=type(None):
                MESHES = {
                    'vertexes': np.array([]),
                    'faces': np.array([]), 
                    'faceColors': np.array([])
                }
                point_cloud_array.put({
                    'POINTS': final_points,
                    'MESHES': MESHES
                })

            # vis.remove_geometry(pcd)

            # x, y, z = velodyine_points[:,0].copy(), velodyine_points[:,1].copy(), velodyine_points[:,2].copy()
            # final_points_o3d = velodyine_points.copy()

            x, y, z = final_points[:,0].copy(), final_points[:,1].copy(), final_points[:,2].copy()
            final_points_o3d = final_points.copy()
            
            final_points_o3d[:,0] = y
            final_points_o3d[:,1] = x
            # points[:,2] = (z*10) + 10
            final_points_o3d[:,2] = z

            pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(velodyine_points_camera)
            pcd.points = o3d.utility.Vector3dVector(final_points_o3d)

            # vis.add_geometry(pcd)
            # vis.poll_events()
            # vis.update_renderer()

            o3d.visualization.draw_geometries([pcd, ])

            time.sleep(5)

    if plot2d:
        cv2.destroyAllWindows()
    if plot3d:
        vis.destroy_window()

if __name__ == "__main__":
    main(None)
    exit()
    if plot3d:
        image_loop_proc = Process(target=main, args=(point_cloud_array, ))
        image_loop_proc.start()
        
        from . import plotter
        plotter.start_graph(point_cloud_array)

        image_loop_proc.join()
    else:
        main(None)
