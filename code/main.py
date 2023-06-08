import coordinates as obj_coord
import beams as obj_beam
import classificadores as obj_classfier
import lidar as obj_lidar
import classificadores_top_k as obj_class_top_k
import matplotlib.pyplot as plt
import pandas as pd
import analise_resultados as obj_analise

#test()

def plot_results(x_data,
                 y_data_1, label_1,
                 y_data_2, label_2,
                 y_data_3, label_3,
                 y_data_4, label_4,
                 antenna_conf,
                 data_set,
                 type_of_beams,
                 data_input,
                 path,
                 name_figure):

#x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path, name_figure):

    x_label = "top-k"
    y_label = "Acuracia"
    title_figure = "Acuracia top-k com  \n antena cofig: "+antenna_conf + " Entrada: "+data_input + " Dataset: "+data_set

    plt.plot(x_data, y_data_1, color = 'r', label=label_1) #linestyle = 'solid'
    #plt.plot(x_data, y_data_1, 'ro')

    plt.plot(x_data, y_data_2, color = 'm',  label=label_2) #'g-o' linestyle = 'dashed',
    #plt.plot(x_data, y_data_2, 'go')

    plt.plot(x_data, y_data_3, color='g', label=label_3) #'b-s' linestyle = 'dashdot'
    #plt.plot(x_data, y_data_3, 'bo')

    plt.plot(x_data, y_data_4, color='c', label=label_4) #'ro', linestyle = 'dotted'
    #plt.plot(x_data, y_data_4, 'bo')


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.text(x_pos_tex, y_pos_tex, text)
    #plt.ylim(min_y_lim, max_y_lim)
    plt.title(title_figure, fontsize=11)
    plt.legend()
    plt.grid(True)
    plt.savefig(path + name_figure, dpi=300, bbox_inches='tight')
    plt.show()


def selecao_beam_top_k(x_train,
                       x_test,
                       y_train,
                       y_test,
                       antenna_conf,
                       data_set,
                       type_of_beams,
                       data_input):

    top_k = [1, 5, 10, 20, 30, 40, 50]

    #acuracia_svc = obj_class_top_k.classificador_svc_top_k(x_train,
    #                                                       x_test,
    #                                                       y_train,
    #                                                       y_test,
    #                                                       antenna_conf,
    #                                                       data_set,
    #                                                       type_of_beams, data_input)
    acuracia_random_forest = obj_class_top_k.classificador_Random_Forest_top_k(x_train,
                                                                               x_test,
                                                                               y_train,
                                                                               y_test,
                                                                               antenna_conf,
                                                                               data_set,
                                                                               type_of_beams,
                                                                               data_input)
    acuracia_KNN = obj_class_top_k.classificador_KNeighbors_top_k(x_train,
                                                                  x_test,
                                                                  y_train,
                                                                  y_test,
                                                                  antenna_conf,
                                                                  data_set,
                                                                  type_of_beams,
                                                                  data_input)
    acuracia_decision_tree = obj_class_top_k.classificador_Decision_Tree_top_k(x_train,
                                                                               x_test,
                                                                               y_train,
                                                                               y_test,
                                                                               antenna_conf,
                                                                               data_set,
                                                                               type_of_beams,
                                                                               data_input)


    ruta = "../results/top-k/" + data_input + "/" + data_set + "/"
    name_figure = "acuracia_beam_selection_top_k_"+ data_input + "_" + data_set + ".png"
    title = "seleção de beam com top-k e 264 pares \n entrada: " + data_input

    acuracia_svc =[0,0,0,0,0,0,0]

    plot_results(top_k,
                 acuracia_KNN, "K Neighbors",
                 acuracia_svc, "SVC",
                 acuracia_decision_tree, "Decision Tree",
                 acuracia_random_forest, "Random Forest",
                 antenna_conf,
                 data_set, type_of_beams, data_input, ruta, name_figure)

   # df_acuracia_SCV = pd.DataFrame(acuracia_svc)
    df_acuracia_KNN = pd.DataFrame(acuracia_KNN)
    df_acuracia_decision_tree = pd.DataFrame(acuracia_decision_tree)
    df_acuracia_random_forest = pd.DataFrame(acuracia_random_forest)

    #df_acuracia_SCV.to_csv(ruta + 'acuracia_SVC_' + data_input + '_' + data_set + '_top_k.csv')
    df_acuracia_KNN.to_csv(ruta + 'acuracia_KNN_' + data_input + '_' + data_set + '_top_k.csv')
    df_acuracia_decision_tree.to_csv(ruta + 'acuracia_decision_tree_' + data_input + '_' + data_set + '_top_k.csv')
    df_acuracia_random_forest.to_csv(ruta + 'acuracia_random_forest_' + data_input + '_' + data_set + '_top_k.csv')

#----------------- MAIN ----------------


obj_analise.read_data()
antenna_conf = '8X32'
data_set = ['ALL', 'LOS', 'NLOS']
type_of_beams = ['combined', 'rx', 'tx']
data_input = ['coord', 'LiDAR']

all_info = obj_coord.read_coordinates()
valid_coord_train, valid_coord_test, coord_LOS_train, coord_LOS_test, coord_NLOS_train, coord_NLOS_test = obj_coord.all_valid_LOS_NLOS_coord(all_info)

data_input = 'LiDAR'
data_set = 'ALL'
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
#obj_classfier.classificador_svc(x_train, x_test, y_train, y_test, antenna_conf, data_set, type_of_beams,data_input)

selecao_beam_top_k(x_train,
                   x_test,
                   y_train,
                   y_test,
                   antenna_conf,
                   data_set,
                   type_of_beams,
                   data_input)


a=0