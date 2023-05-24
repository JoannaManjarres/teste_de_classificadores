# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz
from sklearn.tree import plot_tree


def plot_results(x_data, y_data, x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path, name_figure):


    plt.plot(x_data, y_data, 'r--')
    plt.plot(x_data, y_data, 'bo')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(x_pos_tex, y_pos_tex, text)
    plt.ylim(min_y_lim, max_y_lim)
    plt.title(title_figure, fontsize=11)
    plt.grid(True)
    # plt.show()
    plt.savefig(path + name_figure)
def classificador_svc(X_train, X_test, y_train, y_test, antenna_conf, data_set, type_of_beams):
    '''Support Vector Classifier (SVC) is a form of Support Vector Machines (SVM)
    capable of categorizing inputs under supervised training.'''

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001) # kernel is define like exponencial,
                               # for use a polynomial kernel use coef0,
                               # for use a sigmoid kernel use coef0


    '''
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted, zero_division=0)}\n"
    )
    '''

    print("Tuning Hyperparameters for SVM classifier for antenna confi", antenna_conf, "com o dataset: ", data_set)
    c = [0.1, 1, 10, 100, 1000, 2000]
    gamma = [1, 0.1, 0.01, 0.001, 0.0001] #Taxa de aprendizado
    kernel = 'rbf' #examples for kernel: 'rbf', 'precomputed', 'linear', 'sigmoid', 'poly'

    scores_list=[]
    for i in range(len(gamma)):
        clf = svm.SVC(gamma=gamma[i])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    max_accuracy = max(scores_list)
    ind_max_accuracy = scores_list.index(max_accuracy)
    optimize_learn_rate = gamma[ind_max_accuracy]
    max_y_lim = max(scores_list)+0.02
    min_y_lim = min(scores_list)-0.02

    print("Valores otimos: Taxa de aprendizado otima =", optimize_learn_rate, "Acuracia max =", max_accuracy)

    x_data = gamma
    y_data = scores_list
    x_label = 'Taxa de aprendizado'
    y_label = 'Acurácia'
    x_pos_tex = optimize_learn_rate+0.0001
    y_pos_tex = max_accuracy
    text = round(max_accuracy, 4)
    title_figure = 'Definição dos Hiperparametros do classificador SVM \n kernel= rbf conf da antena '+ antenna_conf +', Dataset: '+ data_set+type_of_beams
    path = '../results/svm/'+antenna_conf+'/'+data_set+'/'
    name_figure = 'svm_tunning_gama_'+antenna_conf+'_'+data_set+'_'+type_of_beams+'.png'

    plot_results(x_data, y_data, x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path, name_figure)

    #-------------------------------------------------

    scores_list_c = []
    for i in range(len(c)):
        clf_c = svm.SVC(C=c[i], gamma=optimize_learn_rate)
        clf_c.fit(X_train, y_train)
        y_pred_c = clf_c.predict(X_test)
        scores_list_c.append(metrics.accuracy_score(y_test, y_pred_c))

    max_accuracy_c = max(scores_list_c)
    ind_max_accuracy_c = scores_list_c.index(max_accuracy_c)
    optimize_c = c[ind_max_accuracy_c]
    max_y_lim_c = max(scores_list_c) + 0.02
    min_y_lim_c = min(scores_list_c) - 0.02

    print("Valores otimos: Taxa de aprendizado otima=", optimize_learn_rate, " c =", optimize_c, "Acuracia max = ", max_accuracy_c)

    x_data = c
    y_data = scores_list_c
    x_label = 'c'
    y_label = 'Acurácia'
    x_pos_tex = optimize_c + 0.001
    y_pos_tex = max_accuracy_c
    text = round(max_accuracy_c, 4)
    min_y_lim = min_y_lim_c
    max_y_lim = max_y_lim_c
    title_figure = 'Definição dos Hiperparametros do classificador SVM \n kernel= rbf e Taxa de apre = ' + str(optimize_learn_rate) +', conf da antena '+ antenna_conf + ', Dataset:'+data_set+type_of_beams
    name_figure = 'svm_tunning_c_'+antenna_conf+'_'+data_set+'_'+type_of_beams+'.png'

    plot_results(x_data, y_data, x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path,
                 name_figure)


    ''' MELHOR MODELO '''
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=optimize_learn_rate, C=optimize_c)  # kernel is define like exponencial,
    # for use a polynomial kernel use coef0,
    # for use a sigmoid kernel use coef0

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted, zero_division=0)}\n"
    )


def classificador_KNeighbors(X_train, X_test, y_train, y_test, antenna_conf, data_set, type_of_beams):
    k_range = range(10, 400, 20)
    neighbors = [30, 40, 80, 120,154,156,158,160,162,164,166,168,200,240,280]
    scores_list = []



    # Treinando
    for k in range(len(neighbors)):
        knn = KNeighborsClassifier(n_neighbors=neighbors[k])
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    # Imprimindo acurácia
    print("-------------------------------")
    print("CLASIFICADOR KNEIGHBORS")
    print("\t Nro de vizinhos: ", neighbors)
    print("\t Acurácia: % s" % scores_list)
    max_accuracy = scores_list[scores_list.index(max(scores_list))]
    optimize_neighbor = neighbors[scores_list.index(max(scores_list))]

    print("\t Valores otimos: Nro vizinhos ",optimize_neighbor,"Acuracia max ",max_accuracy)
    print("-------------------------------")

    x_data = neighbors
    y_data = scores_list
    x_label = 'Numero de vizinhos'
    y_label = 'Acurácia'
    x_pos_tex = optimize_neighbor+1
    y_pos_tex = max(scores_list)
    text = round(max(scores_list), 4)
    min_y_lim = min(scores_list)-0.1
    max_y_lim = max(scores_list)+0.1
    title_figure = 'Definição dos Hiperparametros do classificador k-vizinhos \n conf da antena ' + antenna_conf + ', Dataset:' + data_set + type_of_beams
    path ='../results/k_neighbors/'+ antenna_conf + '/' + data_set+'/'
    name_figure = 'tunning_k_vizinhos_' + antenna_conf + '_' + data_set + '_'+type_of_beams+'.png'

    plot_results(x_data, y_data, x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path,
                 name_figure)

    #plt.plot(neighbors, scores_list, 'r--')
    #plt.plot(neighbors, scores_list, 'bo')
    #plt.xlabel('Numero de vizinhos')
    #plt.ylabel('Acurácia')
    #plt.text(optimize_neighbor+1, max(scores_list), round(max(scores_list), 4))
    #plt.ylim(0.60, 0.70)
    #plt.title("Definição dos Hiperparametros do classificador \n k-vizinhos", fontsize=16)
    #plt.grid(True)
    #plt.show()
    #plt.savefig('../results/k_neighbors/tunning_k_neighbors.png')


def classificador_Decision_Tree(X_train, X_test, y_train, y_test, antenna_conf, data_set, type_user):

    '''Decision Trees (DTs) are a non-parametric supervised learning method
    used for classification and regression. The goal is to create a model that
    predicts the value of a target variable by learning simple decision rules
    inferred from the data features. A tree can be seen as a piecewise
    constant approximation.

    - The cost of using the tree (i.e., predicting data) is logarithmic in the
    number of data points used to train the tree.'''

    deep_of_tree = [1,2,3,4,5,6,7,8,9,10,15,20]
    samples_for_leaf = [1,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450]
    scores_list = []

    #parametros:
    #               max_depth -> ramificacoes max que tera a arvores
    #               min_samples_leaf. minimo de amostras nos nós finais

    for k in range(len(deep_of_tree)):
        clf = tree.DecisionTreeClassifier(max_depth=deep_of_tree[k], random_state=0)
        clf = clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))


    print("-------------------------------")
    print("CLASSIFICADOR DECISION TREE CLASSIFIER")
    print("\t prof da arvore = ", deep_of_tree)
    print("\t Acuracia = ", scores_list)
    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_deep_tree = deep_of_tree[index_acuracy_max]
    min_lim_y = min(scores_list)-0.1
    max_lim_y = accuracy_max+0.1

    print("\t *** Acuracia Max =", accuracy_max)
    print("\t *** Profundidade da Arvore =", optimize_deep_tree)

    x_data = deep_of_tree
    y_data = scores_list
    x_label = 'Profundidade da Arvore'
    y_label = 'Acurácia'
    x_pos_tex = optimize_deep_tree+0.02
    y_pos_tex = accuracy_max+0.02
    text = round(max(scores_list), 3)
    min_y_lim = min_lim_y
    max_y_lim = max_lim_y
    title_figure = 'Definição dos Hiperparametros do classificador Arvore de Decisão  \n '+ ', config da antena ' + antenna_conf + ', Dataset:' + data_set + type_user
    name_figure = 'tunning_deep_of_tree_' + antenna_conf + '_' + data_set +'_'+type_user+ '.png'
    path = '../results/Decision_tree/'+antenna_conf+'/'+data_set+'/'
    plot_results(x_data, y_data, x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path,
                 name_figure)



    #clf = tree.DecisionTreeClassifier(max_depth=7, random_state=0)
    #clf = clf.fit(X_train, y_train)
    #plt.figure(figsize=(15,10))

    #plot_tree(clf, feature_names=['x', 'y', 'z'], class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8'], fontsize=6)
    #plt.title('xxx')
    #plt.show()

    scores_list = []
    accuracy_max = 0
    index_acuracy_max_leaf = 0
    optimize_samples_for_leaf = 0
    min_lim_y = 0
    max_lim_y = 0
    for i in range(len(samples_for_leaf)):
        clf = tree.DecisionTreeClassifier(min_samples_leaf=samples_for_leaf[i], max_depth=optimize_deep_tree, random_state=0)
        clf = clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    print("\t Amostras nas folhas finais = ", samples_for_leaf)
    print("\t Acuracia = ", scores_list)
    accuracy_max = max(scores_list)
    index_acuracy_max_leaf = scores_list.index(accuracy_max)
    optimize_samples_for_leaf = samples_for_leaf[index_acuracy_max_leaf]
    min_lim_y = min(scores_list) - 0.1
    max_lim_y = accuracy_max + 0.1

    print("\t *** Acuracia Max =", accuracy_max)
    print("\t *** Numero de Amostras nas folhas finais =", optimize_samples_for_leaf)

    x_data = samples_for_leaf
    y_data = scores_list
    x_label = 'Numero de Amostras nas folhas finais'
    y_label = 'Acurácia'
    x_pos_tex = optimize_samples_for_leaf+0.02
    y_pos_tex = accuracy_max + 0.02
    text = round(accuracy_max,3)
    min_y_lim = min_lim_y
    max_y_lim = max_lim_y
    title_figure = 'Definição dos Hiperparametros do classificador Arvore de Decisão  \n ' + ', prof da arvpore: '+str(deep_of_tree[index_acuracy_max])+', config da antena: ' + antenna_conf + ', Dataset:' + data_set + type_user
    name_figure = 'tunning_samples_for_leaf_' + antenna_conf + '_' + data_set +'_'+type_user+ '.png'
    path = '../results/Decision_tree/' + antenna_conf + '/' + data_set + '/'
    plot_results(x_data, y_data, x_label, y_label, x_pos_tex, y_pos_tex, text, min_y_lim, max_y_lim, title_figure, path,
                 name_figure)
    '''
    plt.plot(samples_for_leaf, scores_list, 'r--')
    plt.plot(samples_for_leaf, scores_list, 'bo')
    plt.xlabel('Numero de Amostras nas folhas finais')
    plt.ylabel('Acurácia')
    plt.text(optimize_samples_for_leaf+0.02, accuracy_max+0.02, round(accuracy_max,3))
    plt.title("Definição dos Hiperparametros do classificador \n Arvore de Decisão", fontsize=16)
    plt.ylim(min_lim_y, max_lim_y)
    plt.grid(True)
    plt.savefig('../results/Decision_tree/tunning_samples_for_leaf.png')
    plt.show()
'''
    #plt.figure()
    #plot_tree(clf, filled=True, fontsize=8)
    #plot_tree(clf,feature_names=['x','y','z'], class_names=['0','1','2','3','4','5','6','7','8'],fontsize=6)
    #plt.title('xxx')
    #plt.show()

def classificador_Randon_Forest(X_train, X_test, y_train, y_test, antenna_conf, data_set,type_of_beams):
    '''Soa criadas varias arvores de decisao e o resultado sera a média de todas elas'''


    print("-------------------------------")
    print("CLASSIFICADOR RADOM FOREST")
    n_samples, n_features = X_train.shape
    n_outputs = 8

    # PARAMETROS
    # n_jobs -> numero de arvores em paralelo que ele deve criar
    # n_estimator -> numero de arvores
    number_of_trees = [1,2,3,4,5,6,7,8,10,11,12]#,20,30,40,50]#,100,500]#,750,1000,1500,1750,2000,2250]
    number_of_trees_big = [1,10,30,50,80,100,500,1000,1500,1750,2000,2250]
    scores_list = []

    scores_list_1 = []
    for i in range(len(number_of_trees_big)):
        forest_1 = RandomForestClassifier(n_estimators=number_of_trees_big[i], random_state=1)  # )#, n_jobs=-1)
        forest_1.fit(X_train, y_train)
        y_predict_1 = forest_1.predict(X_test)
        scores_list_1.append(metrics.accuracy_score(y_test, y_predict_1))


    print("\t Nro Arvores : ", number_of_trees_big)
    print("\t Acuracia : ", scores_list_1)

    accuracy_max_big = max(scores_list_1)
    index_acuracy_max_big = scores_list_1.index(accuracy_max_big)
    optimize_number_of_trees_big = number_of_trees[index_acuracy_max_big]
    min_lim_y_big = min(scores_list_1) - 0.05
    max_lim_y_big = accuracy_max_big + 0.05

    print("\t *** Quantidade de arvores otimas = ",optimize_number_of_trees_big)
    print("\t *** Acurácia Max = ",accuracy_max_big)
    print("\t ---------------------------------")



    for i in range(len(number_of_trees)):
        forest = RandomForestClassifier(n_estimators=number_of_trees[i], random_state=1)#)#, n_jobs=-1)
        forest.fit(X_train, y_train)
        y_predict = forest.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    print("\t Nro Arvores : ", number_of_trees)
    print("\t Acuracia : ", scores_list)

    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_number_of_trees = number_of_trees[index_acuracy_max]
    min_lim_y = min(scores_list) - 0.1
    max_lim_y = accuracy_max + 0.1

    print("\t *** Acuracia Max =", accuracy_max)
    print("\t *** Quantidade de Arvores =", optimize_number_of_trees)


    plt.subplot(2,1,2)
    plt.plot(number_of_trees, scores_list, 'r--')
    plt.plot(number_of_trees, scores_list, 'bo')
    plt.ylabel('Acurácia')
    plt.text(optimize_number_of_trees+0.02, accuracy_max+0.02, round(accuracy_max, 4))
    plt.ylim(min_lim_y, max_lim_y)
    plt.xlabel('Numero de arvores')
    plt.grid(True)


    plt.subplot(2,1,1)
    plt.plot(number_of_trees_big, scores_list_1, 'r--')
    plt.plot(number_of_trees_big, scores_list_1, 'bo')
    plt.ylabel('Acurácia')
    plt.text(optimize_number_of_trees_big+0.02, accuracy_max_big+0.02, round(max(scores_list_1), 3))
    plt.ylim(min_lim_y_big, max_lim_y_big)
    plt.grid(True)

    plt.suptitle("Definição dos Hiperparametros do classificador  Random Forest \n config antena: "+antenna_conf+" Dataset: "+data_set+'_'+type_of_beams, fontsize=12)
    path = '../results/random_forest/'+ antenna_conf +'/' + data_set +'/'
    plt.savefig(path + 'tunning_number_of_trees_'+antenna_conf+'_'+data_set+'_'+type_of_beams+'.png')
    plt.show()
    a=0
    #---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    scores_list = []
    accuracy_max = 0
    min_lim_y = 0
    max_lim_y = 0
    samples_in_leaf = [1,10,20,50,80,100,125,150,175,200]
    for i in range(len(samples_in_leaf)):
        forest = RandomForestClassifier(n_estimators=optimize_number_of_trees, min_samples_leaf=samples_in_leaf[i], random_state=1)  # , n_jobs=-1)
        forest.fit(X_train, y_train)
        y_predict = forest.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    print("\t min quant de amostras nas folhas = ", samples_in_leaf)
    print("\t Acuracia = ", scores_list)

    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_number_of_samples_in_leaf = samples_in_leaf[index_acuracy_max]
    min_lim_y = min(scores_list) - 0.05
    max_lim_y = accuracy_max + 0.05

    print("\t *** Acuracia Max =", accuracy_max)
    print("\t *** Quantidade de amostras nas folhas das Arvores =", optimize_number_of_samples_in_leaf)

    plt.plot(samples_in_leaf, scores_list, 'r--')
    plt.plot(samples_in_leaf, scores_list, 'bo')
    plt.xlabel('Numero min das amostras nas folhas das arvores')
    plt.ylabel('Acurácia')
    plt.text(optimize_number_of_samples_in_leaf+0.01, accuracy_max+0.01, round(accuracy_max, 2))
    plt.title("Definição dos Hiperparametros do classificador Random Forest \n Nro de Arvores: "+str(optimize_number_of_trees)+", config antena: "+antenna_conf+", Dataset: "+data_set+'_'+type_of_beams, fontsize=12)
    plt.ylim(min_lim_y, max_lim_y)
    plt.grid(True)
    path = '../results/random_forest/' + antenna_conf + '/' + data_set + '/'
    plt.savefig(path + 'tunning_samples_in_leaf_' + antenna_conf + '_' + data_set +'_'+type_of_beams+ '.png')
    #plt.show()

