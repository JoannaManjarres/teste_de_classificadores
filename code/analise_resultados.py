import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    # LOS
    algoritmos = ['decision_tree', 'KNN', 'random_forest', 'SVC']
    data_Set = ['LOS', 'NLOS', 'ALL']
    top_k = [1, 5, 10, 20, 30, 40, 50]
    antenna_conf= '8X32'
    data_set = 'ALL'
    type_of_beams = 'combined'
    data_input = 'LiDAR'
    path = '../results/top-k/'+data_input+'/'
    name_figure = 'comparacao_algoritmos_top_k_'+data_input+'_'+data_set+'.png'

    file = '../results/top-k/'+data_input+'/'+data_set+'/acuracia_KNN_'+data_input+'_'+data_set+'_top_k.csv'
    data = pd.read_csv(file, usecols=[1])
    KNN = data.to_numpy()

    file = '../results/top-k/'+data_input+'/'+data_set+'/acuracia_decision_tree_'+data_input+'_'+data_set+'_top_k.csv'
    data = pd.read_csv(file, usecols=[1])
    decision_tree = data.to_numpy()

    if data_input == 'LiDAR':
        SVC = [0,0,0,0,0,0,0]

    if data_input == 'coord':
        ile = '../results/top-k/' + data_input + '/' + data_set + '/acuracia_SVC_'+data_input+'_' + data_set + '_top_k.csv'
        data = pd.read_csv(file, usecols=[1])
        SVC = data.to_numpy()

    file = '../results/top-k/'+data_input+'/'+data_set+'/acuracia_random_forest_'+data_input+'_'+data_set+'_top_k.csv'
    data = pd.read_csv(file, usecols=[1])
    random_forest = data.to_numpy()

    file = '../results/top-k/'+data_input+'/'+data_set+'/acuracia_wisard_'+data_input+'_'+data_set+'_top_k.csv'
    data = pd.read_csv(file, usecols=[1])
    wisard = data.to_numpy()




    plot_results(top_k,
                 KNN, "KNN",
                 decision_tree, "Decision Tree",
                 SVC, "SVC",
                 random_forest, "Random Forest",
                 wisard, "Wisard",
                 antenna_conf,
                 data_set,
                 type_of_beams,
                 data_input,
                 path,
                 name_figure)

def plot_results(x_data,
                     y_data_1, label_1,
                     y_data_2, label_2,
                     y_data_3, label_3,
                     y_data_4, label_4,
                     y_data_5, label_5,
                     antenna_conf,
                     data_set,
                     type_of_beams,
                     data_input,
                     path,
                     name_figure):
        # x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path, name_figure):
        sns.set()
        x_label = "top-k"
        y_label = "Acuracia"
        title_figure = "comparacao entre algoritmos top-k com  \n antena cofig: " + antenna_conf + " Entrada: " + data_input + " Dataset: " + data_set

        plt.plot(x_data, y_data_1, color='r', linestyle = 'dotted', label=label_1)  # linestyle = 'solid'
        plt.plot(x_data, y_data_1, 'ro')

        plt.plot(x_data, y_data_2, color='m', linestyle = 'dashdot', label=label_2)  # 'g-o' linestyle = 'dashed',
        plt.plot(x_data, y_data_2, 'mo')

        plt.plot(x_data, y_data_3, color='orange', linestyle = 'dotted', label=label_3)  # 'b-s' linestyle = 'dashdot'
        plt.plot(x_data, y_data_3, 'go')

        plt.plot(x_data, y_data_4, color='c', linestyle = 'dotted', label=label_4)  # 'ro', linestyle = 'dotted'
        plt.plot(x_data, y_data_4, 'co')

        plt.plot(x_data, y_data_5, color='k', linestyle = 'dashed', label=label_5)  # 'ro', linestyle = 'dotted'
        plt.plot(x_data, y_data_5, 'ko')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.text(x_pos_tex, y_pos_tex, text)
        # plt.ylim(min_y_lim, max_y_lim)
        plt.title(title_figure, fontsize=11)
        plt.legend()
        plt.grid(True)
        plt.savefig(path + name_figure, dpi=300, bbox_inches='tight')
        plt.show()


    