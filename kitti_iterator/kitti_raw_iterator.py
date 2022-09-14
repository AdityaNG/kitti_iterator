import os
import yaml
import numpy as np

from torch.multiprocessing import Process, Queue, set_start_method

import cv2

# Sensor Setup: https://www.cvlibs.net/datasets/kitti/setup.php

plot3d = True
plot2d = True
point_cloud_array = None
if __name__ == '__main__':
    if plot3d:
        set_start_method('spawn')
        point_cloud_array = Queue()

def open_yaml(settings_doc):
    settings_doc = settings_doc
    cam_settings = {}
    with open(settings_doc, 'r') as stream:
        try:
            cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return cam_settings

def open_calib(calib_file):
    data = open_yaml(calib_file)
    for k in data:
        try:
            data[k] = np.array(list(map(float, data[k].split(" "))))
        except:
            pass
    return data

class KittiRaw:

    def __init__(self, 
        kitti_raw_base_path="kitti_raw_mini",
        date_folder="2011_09_26",
        sub_folder="2011_09_26_drive_0001_sync"
    ) -> None:
        self.plot3d = True
        self.plot2d = False
        self.scale = 2.0
        self.grid_size = (400.0, 400.0, 10.0)
        self.occupancy_shape = list(map(lambda i: int(i*self.scale), self.grid_size))
        self.grid_x, self.grid_y, self.grid_z = list(map(lambda i: i//2, self.grid_size))
        self.occ_x, self.occ_y, self.occ_z = self.occupancy_shape

        self.kitti_raw_path = os.path.join(kitti_raw_base_path, date_folder)
        self.raw_data_path = os.path.join(self.kitti_raw_path, sub_folder)
        self.image_00_path = os.path.join(self.raw_data_path, "image_00")
        self.image_01_path = os.path.join(self.raw_data_path, "image_01")
        self.image_02_path = os.path.join(self.raw_data_path, "image_02")
        self.image_03_path = os.path.join(self.raw_data_path, "image_03")
        self.oxts_path = os.path.join(self.raw_data_path, "oxts")
        self.velodyne_points_path = os.path.join(self.raw_data_path, "velodyne_points")
        self.calib_cam_to_cam_txt = os.path.join(self.kitti_raw_path, "calib_cam_to_cam.txt")
        self.calib_imu_to_velo_txt = os.path.join(self.kitti_raw_path, "calib_imu_to_velo.txt")
        self.calib_velo_to_cam_txt = os.path.join(self.kitti_raw_path, "calib_velo_to_cam.txt")

        self.calib_cam_to_cam = open_calib(self.calib_cam_to_cam_txt)
        self.calib_imu_to_velo = open_calib(self.calib_imu_to_velo_txt)
        self.calib_velo_to_cam = open_calib(self.calib_velo_to_cam_txt)

        self.R = np.reshape(self.calib_velo_to_cam['R'], (3,3))
        self.T = np.reshape(self.calib_velo_to_cam['T'], (3,1))

        self.K_00 = np.reshape(self.calib_cam_to_cam['K_00'], (3,3))
        self.S_00 = np.reshape(self.calib_cam_to_cam['S_00'], (1,2))
        self.D_00 = np.reshape(self.calib_cam_to_cam['D_00'], (1,5))
        self.w, self.h = list(map(int, (self.S_00[0][0], self.S_00[0][1])))
        self.new_intrinsic_mat, self.roi = cv2.getOptimalNewCameraMatrix(self.K_00, self.D_00, (self.w, self.h), 1, (self.w, self.h))
        self.x, self.y, self.w, self.h = self.roi

        self.intrinsic_mat = self.new_intrinsic_mat
        self.intrinsic_mat = np.vstack((
            np.hstack((
                self.intrinsic_mat, np.zeros((3,1))
            )), 
            np.zeros((1,4))
        ))

        self.img_list = sorted(os.listdir(os.path.join(self.image_00_path, 'data')))
        self.img_list = list(map(lambda x: x.split(".png")[0], self.img_list))
        self.index = 0

    def __len__(self):
        return len(self.img_list)

    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index>=self.__len__():
            raise StopIteration
        data = self[self.index]
        self.index += 1
        return data

    def __getitem__(self, index):
        id = self.img_list[index]
        image_00 = os.path.join(self.image_00_path, 'data', id + ".png")
        image_01 = os.path.join(self.image_01_path, 'data', id + ".png")
        image_02 = os.path.join(self.image_02_path, 'data', id + ".png")
        image_03 = os.path.join(self.image_03_path, 'data', id + ".png")
        velodyine_points = os.path.join(self.velodyne_points_path, 'data', id + ".bin")
        
        assert os.path.exists(image_00)
        assert os.path.exists(image_01)
        assert os.path.exists(image_02)
        assert os.path.exists(image_03)
        assert os.path.exists(velodyine_points)

        image_00 = cv2.imread(image_00)
        image_01 = cv2.imread(image_01)
        image_02 = cv2.imread(image_02)
        image_03 = cv2.imread(image_03)
        
        image_02 = cv2.undistort(image_02, self.K_00, self.D_00, None, self.new_intrinsic_mat)
        x, y, w, h = self.roi
        image_02 = image_02[y:y+h, x:x+w]

        velodyine_points = np.fromfile(velodyine_points, dtype=np.float32)
        velodyine_points = np.reshape(velodyine_points, (velodyine_points.shape[0]//4, 4))
        
        occupancy_grid = np.zeros(self.occupancy_shape, dtype=np.float32)
        
        
        for p in velodyine_points:
            p3d = np.array([
                p[0], p[1], p[2]
            ]).reshape((3,1))
            p3d = p3d - self.T
            p3d = self.R @ p3d
            p4d = np.ones((4,1))
            p4d[:3,:] = p3d
            p2d = self.intrinsic_mat @ p4d
            if p2d[2][0]!=0:
                img_x, img_y = p2d[0][0]//p2d[2][0], p2d[1][0]//p2d[2][0]
                
                if (0 <= img_x < w and 0 <= img_y < h and p3d[2]>0) and 0<p[0]<self.grid_x and -self.grid_y<p[1]<self.grid_y and -self.grid_z<p[2]<self.grid_z:
                    if plot2d:
                        image_02 = cv2.circle(image_02, (int(img_x), int(img_y)), 1, (0,255,0), -1)
                    i, j, k = [
                        int((p[0]*self.occ_x//2)//self.grid_x),
                        int((p[1]*self.occ_y//2)//self.grid_y + self.occ_y//2),
                        int((p[2]*self.occ_z//2)//self.grid_z + self.occ_z//2)
                    ]
                    occupancy_grid[i,j,k] = 1.0
        return image_00, image_01, image_02, image_03, velodyine_points, occupancy_grid



def main(point_cloud_array=point_cloud_array):
    k_raw = KittiRaw()
    print("Found ", len(k_raw.img_list), "images: ", k_raw.img_list)
    for data in k_raw:
        image_00, image_01, image_02, image_03, velodyine_points, occupancy_grid = data
        
        if plot2d:
            cv2.imshow('image_02', image_02)
            key = cv2.waitKey(1)
            if key == ord('q'):
                return

        final_points = []
        for i in range(occupancy_grid.shape[0]):
            for j in range(occupancy_grid.shape[1]):
                for k in range(occupancy_grid.shape[2]):
                    x,y,z = [
                        # (i - occ_x/2) * grid_x / (occ_x/2),
                        (i) * k_raw.grid_x / (k_raw.occ_x/2),
                        (j - k_raw.occ_y/2) * k_raw.grid_y / (k_raw.occ_y/2),
                        (k - k_raw.occ_z/2) * k_raw.grid_z / (k_raw.occ_z/2)
                    ]
                    if occupancy_grid[i,j,k] == 1.0:
                        final_points.append((x,y,z))
        final_points = np.array(final_points, dtype=np.float32)
        MESHES = {
            'vertexes': np.array([]),
            'faces': np.array([]), 
            'faceColors': np.array([])
        }
        print(final_points)
        if plot3d:
            point_cloud_array.put({
                'POINTS': final_points,
                'MESHES': MESHES
            })

if __name__ == "__main__":
    if plot3d:
        image_loop_proc = Process(target=main, args=(point_cloud_array, ))
        image_loop_proc.start()
        
        from . import plotter
        plotter.start_graph(point_cloud_array)

        image_loop_proc.join()
    else:
        main(None)
