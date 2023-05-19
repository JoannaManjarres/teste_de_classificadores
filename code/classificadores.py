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



def classificador_svc(X_train, X_test, y_train, y_test):
    '''Support Vector Classifier (SVC) is a form of Support Vector Machines (SVM)
    capable of categorizing inputs under supervised training.'''

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001) # kernel is define like exponencial,
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

    print("Tuning Hyperparameters")
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
    max_y_lim = max(scores_list)+0.002
    min_y_lim = min(scores_list)-0.002

    print("Valores otimos: Taxa de aprendizado otima= ", optimize_learn_rate, "Acuracia max= ", max_accuracy)

    plt.plot(gamma, scores_list, 'r--')
    plt.plot(gamma, scores_list, 'bo')
    plt.xlabel('Taxa de aprendizado')
    plt.ylabel('Acurácia')
    plt.text(optimize_learn_rate+0.0001, max_accuracy, round(max_accuracy, 4))
    plt.ylim(min_y_lim, max_y_lim)
    plt.title("Definição dos Hiperparametros do classificador \n SVM com kernel= rbf", fontsize=16)
    plt.grid(True)
    plt.show()

    #-------------------------------------------------

    scores_list_c = []
    for i in range(len(c)):
        clf_c = svm.SVC(C=c[i], gamma=0.01)
        clf_c.fit(X_train, y_train)
        y_pred_c = clf_c.predict(X_test)
        scores_list_c.append(metrics.accuracy_score(y_test, y_pred_c))

    max_accuracy_c = max(scores_list_c)
    ind_max_accuracy_c = scores_list_c.index(max_accuracy_c)
    optimize_c = c[ind_max_accuracy_c]
    max_y_lim_c = max(scores_list_c) + 0.002
    min_y_lim_c = min(scores_list_c) - 0.002

    print("Valores otimos: Taxa de aprendizado otima=0.01, c = ", optimize_c, "Acuracia max= ", max_accuracy_c)

    plt.plot(c, scores_list_c, 'r--')
    plt.plot(c, scores_list_c, 'bo')
    plt.xlabel('c')
    plt.ylabel('Acurácia')
    plt.text(optimize_c + 0.0001, max_accuracy_c, round(max_accuracy_c, 4))
    plt.ylim(min_y_lim_c, max_y_lim_c)
    plt.title("Definição dos Hiperparametros do classificador \n SVM com kernel= rbf e Taxa de apre = 0.01", fontsize=16)
    plt.grid(True)
    plt.show()

    a=0


def classificador_KNeighbors(X_train, X_test, y_train, y_test):
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
    print("Nro de vizinhos: ", neighbors)
    print("Acurácia: % s" % scores_list)
    max_accuracy = scores_list[scores_list.index(max(scores_list))]
    optimize_neighbor = neighbors[scores_list.index(max(scores_list))]

    print("Valores otimos: Nro vizinhos ",optimize_neighbor,"Acuracia max ",max_accuracy)

    plt.plot(neighbors, scores_list, 'r--')
    plt.plot(neighbors, scores_list, 'bo')
    plt.xlabel('Numero de vizinhos')
    plt.ylabel('Acurácia')
    plt.text(optimize_neighbor+1, max(scores_list), round(max(scores_list), 4))
    #plt.ylim(0.60, 0.70)
    plt.title("Definição dos Hiperparametros do classificador \n k-vizinhos", fontsize=16)
    plt.grid(True)
    plt.show()

def classificador_Decision_Tree(X_train, X_test, y_train, y_test):

    '''Decision Trees (DTs) are a non-parametric supervised learning method
    used for classification and regression. The goal is to create a model that
    predicts the value of a target variable by learning simple decision rules
    inferred from the data features. A tree can be seen as a piecewise
    constant approximation.

    - The cost of using the tree (i.e., predicting data) is logarithmic in the
    number of data points used to train the tree.'''

    deep_of_tree = [1,2,3,4,5,6,7,8,9,10]
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

    print("Acuracia Decision Tree Classifier = ", scores_list)
    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_deep_tree = deep_of_tree[index_acuracy_max]
    min_lim_y = min(scores_list)-0.1
    max_lim_y = accuracy_max+0.1

    print("Acuracia Max =", accuracy_max)
    print("Profundidade da Arvore =", optimize_deep_tree)


    plt.plot(deep_of_tree, scores_list, 'r--')
    plt.plot(deep_of_tree, scores_list, 'bo')
    plt.xlabel('Numero de maximo de Ramificaçoes da Arvore')
    plt.ylabel('Acurácia')
    plt.text(optimize_deep_tree+0.02, accuracy_max+0.02, round(max(scores_list), 2))
    plt.ylim(min_lim_y, max_lim_y)
    plt.title("Definição dos Hiperparametros do classificador \n Arvore de Decisão", fontsize=16)
    plt.grid(True)
    plt.show()


    clf = tree.DecisionTreeClassifier(max_depth=7, random_state=0)
    clf = clf.fit(X_train, y_train)
    plt.figure(figsize=(15,10))

    plot_tree(clf, feature_names=['x', 'y', 'z'], class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8'], fontsize=6)
    plt.title('xxx')
    plt.show()

    scores_list = []
    accuracy_max = 0
    index_acuracy_max = 0
    optimize_samples_for_leaf = 0
    min_lim_y = 0
    max_lim_y = 0
    for i in range(len(samples_for_leaf)):
        clf = tree.DecisionTreeClassifier(min_samples_leaf=samples_for_leaf[i], random_state=0)
        clf = clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    print("Acuracia Decision Tree Classifier = ", scores_list)
    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_samples_for_leaf = samples_for_leaf[index_acuracy_max]
    min_lim_y = min(scores_list) - 0.1
    max_lim_y = accuracy_max + 0.1

    print("Acuracia Max =", accuracy_max)
    print("Numero de Amostras nas folhas finais =", optimize_samples_for_leaf)

    plt.plot(samples_for_leaf, scores_list, 'r--')
    plt.plot(samples_for_leaf, scores_list, 'bo')
    plt.xlabel('Numero de Amostras nas folhas finais')
    plt.ylabel('Acurácia')
    plt.text(optimize_samples_for_leaf+0.02, accuracy_max+0.02, round(accuracy_max,2))
    plt.title("Definição dos Hiperparametros do classificador \n Arvore de Decisão", fontsize=16)
    plt.ylim(min_lim_y, max_lim_y)
    plt.grid(True)
    plt.show()

    plt.figure()
    plot_tree(clf, filled=True, fontsize=8)
    plot_tree(clf,feature_names=['x','y','z'], class_names=['0','1','2','3','4','5','6','7','8'],fontsize=6)
    plt.title('xxx')
    plt.show()

def classificador_Randon_Forest(X_train, X_test, y_train, y_test):
    '''Soa criadas varias arvores de decisao e o resultado sera a média de todas elas'''

    print("Classificador Random Forest")
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

    print("Nro Arvores : ", number_of_trees_big)
    print("Acuracia : ", scores_list_1)

    accuracy_max_big = max(scores_list_1)
    index_acuracy_max_big = scores_list_1.index(accuracy_max_big)
    optimize_number_of_trees_big = number_of_trees[index_acuracy_max_big]
    min_lim_y_big = min(scores_list_1) - 0.05
    max_lim_y_big = accuracy_max_big + 0.05

    print("Quantidade de arvores otimas = ",optimize_number_of_trees_big)
    print("Acurácia Max = ",accuracy_max_big)
    print("---------------------------------")



    for i in range(len(number_of_trees)):
        forest = RandomForestClassifier(n_estimators=number_of_trees[i], random_state=1)#)#, n_jobs=-1)
        forest.fit(X_train, y_train)
        y_predict = forest.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    print("Nro Arvores : ", number_of_trees)
    print("Acuracia : ", scores_list)

    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_number_of_trees = number_of_trees[index_acuracy_max]
    min_lim_y = min(scores_list) - 0.1
    max_lim_y = accuracy_max + 0.1

    print("Acuracia Max =", accuracy_max)
    print("Quantidade de Arvores =", optimize_number_of_trees)


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

    plt.suptitle("Definição dos Hiperparametros do classificador \n Random Forest", fontsize=16)
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
        forest = RandomForestClassifier(n_estimators=5, min_samples_leaf=samples_in_leaf[i], random_state=1)  # , n_jobs=-1)
        forest.fit(X_train, y_train)
        y_predict = forest.predict(X_test)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))

    print("Acuracia com Random Forest com variacao de minimo numero de amostras nas folhas das arvores: ", scores_list)
    accuracy_max = max(scores_list)
    index_acuracy_max = scores_list.index(accuracy_max)
    optimize_number_of_samples_in_leaf = samples_in_leaf[index_acuracy_max]
    min_lim_y = min(scores_list) - 0.05
    max_lim_y = accuracy_max + 0.05

    print("Acuracia Max =", accuracy_max)
    print("Quantidade de amostras nas folhas das Arvores =", optimize_number_of_samples_in_leaf)

    plt.plot(samples_in_leaf, scores_list, 'r--')
    plt.plot(samples_in_leaf, scores_list, 'bo')
    plt.xlabel('Numero min das amostras nas folhas das arvores')
    plt.ylabel('Acurácia')
    plt.text(optimize_number_of_samples_in_leaf+0.01, accuracy_max+0.01, round(accuracy_max, 2))
    plt.title("Definição dos Hiperparametros do classificador \n Random Forest", fontsize=16)
    plt.ylim(min_lim_y, max_lim_y)
    plt.grid(True)
    plt.show()

