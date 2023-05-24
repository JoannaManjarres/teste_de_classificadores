import numpy as np
import csv


def read_coordinates():
    filename = "../data/coordinates/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        number_of_rows = len(list(reader))

    all_info_coord = np.zeros([number_of_rows, 6], dtype=object)

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        for row in reader:
            all_info_coord[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
            cont += 1

    return all_info_coord

def all_valid_LOS_NLOS_coord(all_info_coord):
    '''
        ordem dos parametros de all_info_coord
        EpisodeID, x, y, z, LOS, Val
    '''

    limit_ep_train = 1564
    valid_coord_all_info = all_info_coord[(all_info_coord[:, 5] == 'V')]
    LOS_coord_all_info = valid_coord_all_info[(valid_coord_all_info[:, 4] == 'LOS=1')]
    NLOS_coord_all_info = valid_coord_all_info[(valid_coord_all_info[:, 4] == 'LOS=0')]

    # Separacao do conjunto de dados em treinamento e teste dos dados completos
    valid_coord_all_info_train = valid_coord_all_info[(valid_coord_all_info[:,0] < limit_ep_train+1)]
    valid_coord_all_info_test  = valid_coord_all_info[(valid_coord_all_info[:,0]) > limit_ep_train]

    # coordenadas x,y,z do grupo de train e test
    valid_coord_train = valid_coord_all_info_train[:,[1,2,3]]
    valid_coord_test  = valid_coord_all_info_test[:,[1,2,3]]

    # Separacao do conjunto de dados em treinamento e teste dos dados LOS
    valid_coord_LOS_info_train = LOS_coord_all_info[(LOS_coord_all_info[:, 0] < limit_ep_train + 1)]
    valid_coord_LOS_info_test = LOS_coord_all_info[(LOS_coord_all_info[:, 0]) > limit_ep_train]

    # coordenadas x,y,z do grupo de train e test LOS
    coord_LOS_train = valid_coord_LOS_info_train[:, [1, 2, 3]]
    coord_LOS_test = valid_coord_LOS_info_test[:, [1, 2, 3]]

    # Separacao do conjunto de dados em treinamento e teste dos dados NLOS
    valid_coord_NLOS_info_train = NLOS_coord_all_info[(NLOS_coord_all_info[:, 0] < limit_ep_train + 1)]
    valid_coord_NLOS_info_test = NLOS_coord_all_info[(NLOS_coord_all_info[:, 0]) > limit_ep_train]

    # coordenadas x,y,z do grupo de train e test NLOS
    coord_NLOS_train = valid_coord_NLOS_info_train[:, [1, 2, 3]]
    coord_NLOS_test = valid_coord_NLOS_info_test[:, [1, 2, 3]]

    return valid_coord_train, valid_coord_test, coord_LOS_train, coord_LOS_test, coord_NLOS_train, coord_NLOS_test


def preprocess_coordinates():
    coord_train = 1
    coord_test = 1

    return coord_train, coord_test
