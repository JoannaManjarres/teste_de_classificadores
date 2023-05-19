import coordinates as obj_coord
import beams as obj_beam
import classificadores as obj_classfier

#test()


#----------------

all_info = obj_coord.read_coordinates()
coord_train, coord_test = obj_coord.preprocess_valid_coord(all_info)
index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = obj_beam.read_all_beams('1X8')


#obj_classfier.classificador_KNeighbors(coord_train, coord_test, index_beam_tx_train, index_beam_tx_test)
#obj_classfier.classificador_Randon_Forest(coord_train, coord_test, index_beam_tx_train, index_beam_tx_test)
#obj_classfier.classificador_Decision_Tree(coord_train, coord_test, index_beam_tx_train, index_beam_tx_test)
obj_classfier.classificador_svc(coord_train, coord_test, index_beam_tx_train, index_beam_tx_test)


a=0