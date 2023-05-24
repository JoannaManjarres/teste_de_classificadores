import coordinates as obj_coord
import beams as obj_beam
import classificadores as obj_classfier

#test()


#----------------

antenna_conf = '8X32'
data_set = ['All', 'LOS', 'NLOS']
type_of_beams = ['combined', 'rx', 'tx']
all_info = obj_coord.read_coordinates()
valid_coord_train, valid_coord_test, coord_LOS_train, coord_LOS_test, coord_NLOS_train, coord_NLOS_test = obj_coord.all_valid_LOS_NLOS_coord(all_info)


data_set = 'ALL'
type_of_beams = 'rx'




if data_set == 'ALL':
    x_train = valid_coord_train
    x_test = valid_coord_test
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
    x_train = coord_LOS_train
    x_test = coord_LOS_test
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
    x_train = coord_NLOS_train
    x_test = coord_NLOS_test
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

obj_classfier.classificador_KNeighbors(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams)
#obj_classfier.classificador_Randon_Forest(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams)

#obj_classfier.classificador_Decision_Tree(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams)

#obj_classfier.classificador_svc(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams)


a=0