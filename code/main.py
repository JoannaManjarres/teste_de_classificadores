import coordinates as obj_coord
import beams as obj_beam
import classificadores as obj_classfier
import lidar as obj_lidar

#test()


#----------------

antenna_conf = '8X32'
data_set = ['ALL', 'LOS', 'NLOS']
type_of_beams = ['combined', 'rx', 'tx']
data_input = ['coord','LiDAR']

all_info = obj_coord.read_coordinates()
valid_coord_train, valid_coord_test, coord_LOS_train, coord_LOS_test, coord_NLOS_train, coord_NLOS_test = obj_coord.all_valid_LOS_NLOS_coord(all_info)

data_input = 'LiDAR'
data_set = 'LOS'
type_of_beams = 'combined'




if data_set == 'ALL':
    if data_input == 'coord':
        x_train = valid_coord_train
        x_test = valid_coord_test
    if data_input == 'LiDAR':
        x_train, x_test = obj_lidar.read_all_LiDAR_data()

    index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test, index_beam_combined_train, index_beam_combined_test = obj_beam.read_all_beams(antenna_conf)

    if type_of_beams == 'combined':
        y_train = index_beam_combined_train
        y_test = index_beam_combined_test
    if type_of_beams == 'rx':
        y_train = index_beam_rx_train
        y_test = index_beam_rx_test
    if type_of_beams == 'tx':
        y_train = index_beam_tx_train
        y_test = index_beam_tx_test

elif data_set == 'LOS':
    if data_input == 'coord':
        x_train = coord_LOS_train
        x_test = coord_LOS_test
    if data_input == 'LiDAR':
        x_train, x_test = obj_lidar.read_LOS_LiDAR_data()

    index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test, index_beam_combined_LOS_train, index_beam_combined_LOS_test = obj_beam.read_LOS_beams(antenna_conf)

    if type_of_beams == 'combined':
        y_train = index_beam_combined_LOS_train
        y_test = index_beam_combined_LOS_test
    if type_of_beams == 'rx':
        y_train = index_beam_rx_LOS_train
        y_test = index_beam_rx_LOS_test
    if type_of_beams == 'tx':
        y_train = index_beam_tx_LOS_train
        y_test = index_beam_tx_LOS_test

elif data_set =='NLOS':
    if data_input == 'coord':
        x_train = coord_NLOS_train
        x_test = coord_NLOS_test
    if data_input == 'LiDAR':
        x_train, x_test = obj_lidar.read_NLOS_LiDAR_data()

    index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test, label_combined_NLOS_train, label_combined_NLOS_test = obj_beam.read_NLOS_beams(antenna_conf)

    if type_of_beams == 'combined':
        y_train = label_combined_NLOS_train
        y_test = label_combined_NLOS_test
    if type_of_beams == 'rx':
        y_train = index_beam_rx_NLOS_train
        y_test = index_beam_rx_NLOS_test
    if type_of_beams == 'tx':
        y_train = index_beam_tx_NLOS_train
        y_test = index_beam_tx_NLOS_test

#obj_classfier.classificador_KNeighbors(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams,data_input)
#obj_classfier.classificador_Randon_Forest(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams, data_input)

#obj_classfier.classificador_Decision_Tree(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams, data_input)

obj_classfier.classificador_svc(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams,data_input)


a=0