
def test_kitti_raw():
    from kitti_iterator import kitti_raw_iterator
    import numpy as np
    import pandas as pd
    raw_iter = kitti_raw_iterator.KittiRaw(
        kitti_raw_base_path="kitti_raw_mini",
        date_folder="2011_09_26",
        sub_folder="2011_09_26_drive_0001_sync"
    )
    for row in raw_iter:
        assert len(row) == 6
        image_00 = row['image_00']
        image_01 = row['image_01']
        image_02 = row['image_02']
        image_03 = row['image_03']
        velodyine_points = row['velodyine_points']
        occupancy_grid = row['occupancy_grid']
        assert type(image_00) == np.ndarray
        assert type(image_01) == np.ndarray
        assert type(image_02) == np.ndarray
        assert type(image_03) == np.ndarray
        assert type(velodyine_points) == np.ndarray
        assert type(occupancy_grid) == np.ndarray
