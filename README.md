# KITTI Iterator

![pylint workflow](https://github.com/PW22-SBN-01/kitti_iterator/actions/workflows/pylint.yml/badge.svg)
![pypi workflow](https://github.com/PW22-SBN-01/kitti_iterator/actions/workflows/pypi.yml/badge.svg)
![pytest workflow](https://github.com/PW22-SBN-01/kitti_iterator/actions/workflows/pytest.yml/badge.svg)

Example usage:

```python
from kitti_iterator import kitti_multimodal_iterator

# Load in the mini dataset
k_raw = KittiRaw(
    kitti_raw_base_path="kitti_raw_mini",
    date_folder="2011_09_26",
    sub_folder="2011_09_26_drive_0001_sync"
)

# Iterate
for data in k_raw:
    image_00, image_01, image_02, image_03, velodyine_points, occupancy_grid = data

# Indexing
image_00, image_01, image_02, image_03, velodyine_points, occupancy_grid = k_raw[3]
```

## Install

```bash
pip install kitti_iterator
```

To install the latest (might be unstable):
```bash
pip install git+https://github.com/PW22-SBN-01/kitti_iterator
```

## Dataset Structre

Download the full dataset at <a href="https://www.cvlibs.net/datasets/kitti/raw_data.php">www.cvlibs.net/datasets/kitti/raw_data.php</a>

```bash
kitti_raw_mini/
└── 2011_09_26
    ├── 2011_09_26_drive_0001_sync
    │   ├── image_00
    │   │   ├── data
    │   │   │   ├── 0000000000.png
    │   │   │   ├── 0000000001.png
    │   │   │   ├── 0000000002.png
    │   │   │   ├── 0000000003.png
    │   │   │   └── 0000000004.png
    │   │   └── timestamps.txt
    │   ├── image_01
    │   │   ├── data
    │   │   │   ├── 0000000000.png
    │   │   │   ├── 0000000001.png
    │   │   │   ├── 0000000002.png
    │   │   │   ├── 0000000003.png
    │   │   │   └── 0000000004.png
    │   │   └── timestamps.txt
    │   ├── image_02
    │   │   ├── data
    │   │   │   ├── 0000000000.png
    │   │   │   ├── 0000000001.png
    │   │   │   ├── 0000000002.png
    │   │   │   ├── 0000000003.png
    │   │   │   └── 0000000004.png
    │   │   └── timestamps.txt
    │   ├── image_03
    │   │   ├── data
    │   │   │   ├── 0000000000.png
    │   │   │   ├── 0000000001.png
    │   │   │   ├── 0000000002.png
    │   │   │   ├── 0000000003.png
    │   │   │   └── 0000000004.png
    │   │   └── timestamps.txt
    │   ├── oxts
    │   │   ├── data
    │   │   │   ├── 0000000000.txt
    │   │   │   ├── 0000000001.txt
    │   │   │   ├── 0000000002.txt
    │   │   │   ├── 0000000003.txt
    │   │   │   └── 0000000004.txt
    │   │   ├── dataformat.txt
    │   │   └── timestamps.txt
    │   └── velodyne_points
    │       ├── data
    │       │   ├── 0000000000.bin
    │       │   ├── 0000000001.bin
    │       │   ├── 0000000002.bin
    │       │   ├── 0000000003.bin
    │       │   └── 0000000004.bin
    │       ├── timestamps_end.txt
    │       ├── timestamps_start.txt
    │       └── timestamps.txt
    ├── calib_cam_to_cam.txt
    ├── calib_imu_to_velo.txt
    └── calib_velo_to_cam.txt
```