import numpy as np

def read_all_LiDAR_data():
    path = '../data/Lidar/All/'

    input_cache_file = np.load(path + "all_data_lidar_train.npz", allow_pickle=True)
    lidar_train = input_cache_file["lidar_train"].astype(int)


    input_cache_file = np.load(path + "all_data_lidar_test.npz", allow_pickle=True)
    lidar_test = input_cache_file["lidar_test"].astype(int)

    return lidar_train, lidar_test

def read_LOS_LiDAR_data():
    path = '../data/Lidar/LOS/'

    input_cache_file = np.load(path + "lidar_LOS_train.npz", allow_pickle=True)
    lidar_LOS_train = input_cache_file["lidar_train"].astype(int)


    input_cache_file = np.load(path + "lidar_LOS_test.npz", allow_pickle=True)
    lidar_LOS_test = input_cache_file["lidar_test"].astype(int)

    return lidar_LOS_train, lidar_LOS_test

def read_NLOS_LiDAR_data():
    path = '../data/Lidar/NLOS/'

    input_cache_file = np.load(path + "lidar_NLOS_train.npz", allow_pickle=True)
    lidar_NLOS_train = input_cache_file["lidar_train"].astype(int)


    input_cache_file = np.load(path + "lidar_NLOS_test.npz", allow_pickle=True)
    lidar_NLOS_test = input_cache_file["lidar_test"].astype(int)

    return lidar_NLOS_train, lidar_NLOS_test