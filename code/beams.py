
import numpy as np


def read_all_beams(antenna_config):

    path ='../data/beams/'+antenna_config+'/all_index_beam/'

    input_cache_file = np.load(path + "index_beams_rx_test.npz", allow_pickle=True)
    index_beam_rx_test = input_cache_file["all_beams_rx_test"].astype(int)

    input_cache_file = np.load(path + "index_beams_rx_train.npz", allow_pickle=True)
    index_beam_rx_train = input_cache_file["all_beams_rx_train"].astype(int)

    input_cache_file = np.load(path + "index_beams_tx_train.npz", allow_pickle=True)
    index_beam_tx_train = input_cache_file["all_beams_tx_train"].astype(int)

    input_cache_file = np.load(path + "index_beams_tx_test.npz", allow_pickle=True)
    index_beam_tx_test = input_cache_file["all_beams_tx_test"].astype(int)

    return index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test

def read_LOS_beams(antenna_config):

    path = '../data/beams/'+antenna_config+'/LOS_index_beam/'

    input_cache_file = np.load(path + "beam_LOS_rx_train.npz", allow_pickle=True)
    index_beam_rx_LOS_train = input_cache_file["beam_LOS_rx_train"].astype(int)
    label_rx_LOS_train =index_beam_rx_LOS_train.tolist()

    input_cache_file = np.load(path + "beam_LOS_rx_test.npz", allow_pickle=True)
    index_beam_rx_LOS_test = input_cache_file["beam_LOS_rx_test"].astype(int)
    label_rx_LOS_test = index_beam_rx_LOS_test.tolist()

    input_cache_file = np.load(path + "beam_LOS_tx_train.npz", allow_pickle=True)
    index_beam_tx_LOS_train = input_cache_file["beam_LOS_tx_train"].astype(int)
    label_tx_LOS_train = index_beam_tx_LOS_train.tolist()

    input_cache_file = np.load(path + "beam_LOS_tx_test.npz", allow_pickle=True)
    index_beam_tx_LOS_test = input_cache_file["beam_LOS_tx_test"].astype(int)
    label_tx_LOS_test = index_beam_tx_LOS_test.tolist()

    return label_rx_LOS_train, label_rx_LOS_test, label_tx_LOS_train, label_tx_LOS_test


def read_NLOS_beams(antenna_config):

    path = '../data/beams/' + antenna_config + '/NLOS_index_beam/'

    input_cache_file = np.load(path + "beam_NLOS_rx_train.npz", allow_pickle=True)
    index_beam_rx_NLOS_train = input_cache_file["beam_NLOS_rx_train"].astype(int)


    input_cache_file = np.load(path + "beam_NLOS_rx_test.npz", allow_pickle=True)
    index_beam_rx_NLOS_test = input_cache_file["beam_NLOS_rx_test"].astype(int)

    input_cache_file = np.load(path + "beam_NLOS_tx_train.npz", allow_pickle=True)
    index_beam_tx_NLOS_train = input_cache_file["beam_NLOS_tx_train"].astype(int)

    input_cache_file = np.load(path + "beam_NLOS_tx_test.npz", allow_pickle=True)
    index_beam_tx_NLOS_test = input_cache_file["beam_NLOS_tx_test"].astype(int)



    return index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test