import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import collections
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz
from sklearn.tree import plot_tree


def classificador_svc_top_k(X_train, X_test, y_train, y_test, antenna_conf, data_set, type_of_beams, data_input):
    '''Support Vector Classifier (SVC) is a form of Support Vector Machines (SVM)
    capable of categorizing inputs under supervised training.'''

    # Create a classifier: a support vector classifier
    #clf = svm.SVC(gamma=0.001, c=) # kernel is define like exponencial,
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
    print("---------------------------------")
    print("Classficador SVM top-k")
    print("   Conf Antena:", antenna_conf, "Dataset: ", data_set, "Tipo de entrada: ", data_input)

    top_k=[1,5,10,20,30,40,50]

    clf = svm.SVC(gamma=0.001, probability=True)
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)

    lista_das_classes_do_modelo = clf.classes_
    index_dos_elementos_y_pred_prob = (-y_pred_prob).argsort()  # coleta dos indices das prob previstas na ordem descendente
    y_pred = lista_das_classes_do_modelo[index_dos_elementos_y_pred_prob]


    acuracia = []
    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra in range(len(y_pred)):
            lista_dos_beams_por_amostra = y_pred[amostra]
            #lista_das_classes_na_ordem_decrescente = sorted(lista_das_classes, reverse=True)
            #index_dos_elementos_da_lista_ordenada = (-lista_das_classes).argsort() #coleta dos indices ou as classes na ordem descendente

            grupo_de_beams_no_top_k = lista_dos_beams_por_amostra[0:top_k[i]]

            if (y_test[amostra] in grupo_de_beams_no_top_k):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto/len(y_pred))

    print("Acuracia SVC top-k...")
    print(top_k)
    print(acuracia)

    path_result = "../results/smv/" + antenna_conf + "/" + data_set + "/" +data_input + "/"
    name_figure = "acuracia_SVC_top_k"
    #df = pd.DataFrame(acuracia,
    #                  columns=['score_top_1', 'score_top_5', 'score_top_10', 'score_top_20', 'score_top_30',
    #                           'score_top_40', 'score_top_50'])
    # df = pd.DataFrame(vector_scores, columns=['score_top_20','score_top_30'])
    #df.to_csv(path_result + name_figure + '.csv')

    #return top_k, acuracia

    return acuracia

def classificador_Randon_Forest_top_k(X_train, X_test, y_train, y_test, antenna_conf, data_set,type_of_beams, data_input):
    '''Soa criadas varias arvores de decisao e o resultado sera a média de todas elas'''

    print("---------------------------------")
    print("Classficador RANDOM FOREST top-k")
    print("   Conf Antena:",antenna_conf,"Dataset: ",data_set, "Tipo de entrada: ", data_input)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    forest = RandomForestClassifier(n_estimators=7, min_samples_leaf=11, random_state=1)  # )#, n_jobs=-1)
    forest.fit(X_train, y_train)
    lista_das_classes_do_modelo = forest.classes_

    y_pred_prob = forest.predict_proba(X_test)


    index_dos_elementos_y_pred_prob = (-y_pred_prob).argsort()  # coleta dos indices das prob previstas na ordem descendente
    y_pred = lista_das_classes_do_modelo[index_dos_elementos_y_pred_prob]
    acuracia = []
    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra in range(len(y_pred)):
            lista_dos_beams_por_amostra = y_pred[amostra]
            # lista_das_classes_na_ordem_decrescente = sorted(lista_das_classes, reverse=True)
            # index_dos_elementos_da_lista_ordenada = (-lista_das_classes).argsort() #coleta dos indices ou as classes na ordem descendente

            grupo_de_beams_no_top_k = lista_dos_beams_por_amostra[0:top_k[i]]

            if (y_test[amostra] in grupo_de_beams_no_top_k):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(y_pred))

    print("Acuracia Random Forest top-k...")
    print(top_k)
    print(acuracia)

    return acuracia

def classificador_KNeighbors_top_k(X_train, X_test, y_train, y_test, antenna_conf, data_set, type_of_beams, data_input):
    print("---------------------------------")
    print("Classficador KNEIGHBORS top-k")
    print("   Conf Antena:", antenna_conf, "Dataset: ", data_set, "Tipo de entrada: ", data_input)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    knn = KNeighborsClassifier(n_neighbors=40)
    knn.fit(X_train, y_train)
    y_pred_prob = knn.predict_proba (X_test)

    lista_das_classes_do_modelo = knn.classes_


    index_dos_elementos_y_pred_prob = (-y_pred_prob).argsort()  # coleta dos indices das prob previstas na ordem descendente
    y_pred = lista_das_classes_do_modelo[index_dos_elementos_y_pred_prob]

    acuracia = []
    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra in range(len(y_pred)):
            lista_dos_beams_por_amostra = y_pred[amostra]
            # lista_das_classes_na_ordem_decrescente = sorted(lista_das_classes, reverse=True)
            # index_dos_elementos_da_lista_ordenada = (-lista_das_classes).argsort() #coleta dos indices ou as classes na ordem descendente

            grupo_de_beams_no_top_k = lista_dos_beams_por_amostra[0:top_k[i]]

            if (y_test[amostra] in grupo_de_beams_no_top_k):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(y_pred))

    print("Acuracia KNN top-k...")
    print(top_k)
    print(acuracia)

    return acuracia

def classificador_Decision_Tree_top_k(X_train, X_test, y_train, y_test, antenna_conf, data_set, type_user, data_input):
    print("---------------------------------")
    print("Classficador DECISION TREE top-k")
    print("   Conf Antena:", antenna_conf, "Dataset: ", data_set, "Tipo de entrada: ", data_input)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    clf = tree.DecisionTreeClassifier(min_samples_leaf=10, max_depth=8, random_state=0)
    clf = clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba (X_test)

    lista_das_classes_do_modelo = clf.classes_

    index_dos_elementos_y_pred_prob = (-y_pred_prob).argsort()  # coleta dos indices das prob previstas na ordem descendente
    y_pred = lista_das_classes_do_modelo[index_dos_elementos_y_pred_prob]

    acuracia = []
    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra in range(len(y_pred)):
            lista_dos_beams_por_amostra = y_pred[amostra]
            # lista_das_classes_na_ordem_decrescente = sorted(lista_das_classes, reverse=True)
            # index_dos_elementos_da_lista_ordenada = (-lista_das_classes).argsort() #coleta dos indices ou as classes na ordem descendente

            grupo_de_beams_no_top_k = lista_dos_beams_por_amostra[0:top_k[i]]

            if (y_test[amostra] in grupo_de_beams_no_top_k):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(y_pred))

    print("Acuracia Decision Tree top-k...")
    print(top_k)
    print(acuracia)

    return acuracia