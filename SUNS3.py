import funs
import pandas as pd
import keras as keras
import sklearn.tree as tree

from graphviz import Source
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

if __name__ == '__main__':
    ########################################################################
    # 1 - LOAD, CLEAN, REMOVE OUTLIERS, NORMALIZE###########################
    ########################################################################
    # LOAD
    train_data = funs.load_data('./data3/train.csv')
    test_data = funs.load_data('./data3/test.csv')

    # CLEAN NA
    train_data = funs.drop_all_na(train_data)
    test_data = funs.drop_all_na(test_data)

    # ENCODE [STAR, GALAXY, QSO] -> [0, 1, 2]
    train_data = train_data.replace('STAR', 0)
    test_data = test_data.replace('STAR', 0)

    train_data = train_data.replace('GALAXY', 1)
    test_data = test_data.replace('GALAXY', 1)

    train_data = train_data.replace('QSO', 2)
    test_data = test_data.replace('QSO', 2)

    # NORMALIZE AND SELECT COLUMNS
    # train/test input - classifier
    train_input = train_data[['u', 'g', 'r', 'i', 'z', 'plate', 'x_coord', 'y_coord', 'z_coord']].copy()
    test_input = test_data[['u', 'g', 'r', 'i', 'z', 'plate', 'x_coord', 'y_coord', 'z_coord']].copy()

    sc = StandardScaler()
    sc.fit(train_input)
    train_input[['u', 'g', 'r', 'i', 'z', 'plate', 'x_coord', 'y_coord', 'z_coord']] = sc.transform(train_input)
    test_input[['u', 'g', 'r', 'i', 'z', 'plate', 'x_coord', 'y_coord', 'z_coord']] = sc.transform(test_input)

    # train/test output - classifier
    train_output = train_data['class'].copy()
    test_output = test_data['class'].copy()

    # CORRELATION IN TRAIN DATA
    corr_mat = pd.DataFrame.corr(train_data)
    print(corr_mat)

    ########################################################################
    # 2 - OBJECT CLASSIFIERS################################################
    ########################################################################
    # # Weak classifier - One decision tree
    # clf_tree = tree.DecisionTreeClassifier()
    # clf_tree.fit(train_input, train_output)
    # test_predict_decision_tree = clf_tree.predict(test_input)
    # cm_desision_tree = confusion_matrix(test_output, test_predict_decision_tree)
    # acc_decision_tree = accuracy_score(test_output, test_predict_decision_tree)
    #
    # # Draw
    # #dotfile = open("./tree.dot", 'r+')
    # #graph = Source(tree.export_graphviz(clf, dotfile, feature_names=train_input.columns))
    # #graph.format = 'png'
    # #graph.render('image2.png', view=True)
    # #dotfile.close()
    #
    # # Strong classifier - Random Forrest#################################################
    # #clf_forrest = RandomForestClassifier(max_depth=3, n_estimators=10, random_state=42) # Acc = 0.94
    # clf_forrest = RandomForestClassifier(max_depth=5, n_estimators=20, random_state=42)  # Acc = 0.97
    # clf_forrest.fit(train_input, train_output)
    #
    # # Test the model after training
    # test_predict_forrest_classifier = clf_forrest.predict(test_input)
    # cm_forrest_classifier = confusion_matrix(test_output, test_predict_forrest_classifier)
    # acc_forrest_classifier = accuracy_score(test_output, test_predict_forrest_classifier)
    #
    # # export tree
    # #dotfile = open("./dot_graphs/tree_clf_0.dot", 'w')
    # dotfile = open("./dot_graphs/tree_bigger_clf_0.dot", 'w')
    # tree.export_graphviz(clf_forrest.estimators_[0], dotfile, feature_names=train_input.columns)
    # dotfile.close()
    #
    # #dotfile = open("./dot_graphs/tree_clf_1.dot", 'w')
    # dotfile = open("./dot_graphs/tree_bigger_clf_1.dot", 'w')
    # tree.export_graphviz(clf_forrest.estimators_[1], dotfile, feature_names=train_input.columns)
    # dotfile.close()
    #
    # # accuracy of clf_forrest_estimator[0]
    # test_predict_estimator_0 = clf_forrest.estimators_[0].predict(test_input)
    # acc_forrest_estimator_0 = accuracy_score(test_output, test_predict_estimator_0)
    #
    # # accuracy of clf_forrest_estimator[1]
    # test_predict_estimator_1 = clf_forrest.estimators_[1].predict(test_input)
    # acc_forrest_estimator_1 = accuracy_score(test_output, test_predict_estimator_1)
    #
    # # Neural Network - Keras#############################################
    # # Create the model
    # clf_keras = Sequential()
    # clf_keras.add(Dense(9, input_dim=9, activation='sigmoid'))
    # clf_keras.add(Dense(9, activation='sigmoid'))
    # clf_keras.add(Dense(3, activation='softmax'))
    #
    # # Configure the model and start training
    # optimizer = keras.optimizers.Adam(lr=0.001)
    # clf_keras.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    # history = clf_keras.fit(train_input.to_numpy(), pd.get_dummies(train_output), epochs=500, batch_size=100, verbose=1,
    #                         validation_split=0.15, callbacks=[early_stopping])
    #
    # # Test the model after training
    # test_predict_keras_classifier = clf_keras.predict_classes(test_input.to_numpy())
    # cm_keras_classifier = confusion_matrix(test_output, test_predict_keras_classifier)
    # acc_keras_classifier = accuracy_score(test_output, test_predict_keras_classifier)
    #
    # # train data evaluation
    # train_results = clf_keras.evaluate(train_input.to_numpy(), pd.get_dummies(train_output), verbose=1)
    # print(f'Keras Train results - Loss: {train_results[0]} - Accuracy: {train_results[1]}%')
    #
    # # test data evaluation
    # test_results = clf_keras.evaluate(test_input.to_numpy(), pd.get_dummies(test_output), verbose=1)
    # print(f'Keras Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    #
    # # COMPARISON TODO big tree and forrest[0,1] accuracy TODO small tree accuracy
    # print(f'Accuracy forrest: {acc_forrest_classifier}% - Accuracy keras: {acc_keras_classifier}%')

    # ########################################################################
    # # 3 - REGRESSION #######################################################
    # ########################################################################
    train_input_regression = train_data[['u', 'g', 'r', 'i', 'z', 'run', 'class', 'plate', 'mjd']].copy()
    test_input_regression = test_data[['u', 'g', 'r', 'i', 'z', 'run', 'class', 'plate', 'mjd']].copy()

    train_output_regression = train_data[['x_coord', 'y_coord', 'z_coord']].copy()
    test_output_regression = test_data[['x_coord', 'y_coord', 'z_coord']].copy()

    # normalize
    sc = StandardScaler()
    sc.fit(train_input_regression)
    train_input_regression[['u', 'g', 'r', 'i', 'z', 'run', 'class', 'plate', 'mjd']] = sc.transform(train_input_regression)
    test_input_regression[['u', 'g', 'r', 'i', 'z', 'run', 'class', 'plate', 'mjd']] = sc.transform(test_input_regression)

    # Bagging regression ######################################################
    reg_random_forrest = RandomForestRegressor(n_estimators=100, random_state=42, verbose=True)
    reg_random_forrest.fit(train_input_regression, train_output_regression)

    # test model after training
    test_predict_random_forrest = reg_random_forrest.predict(test_input_regression)

    # r2
    r2_reg_random_forrest = r2_score(test_output_regression, test_predict_random_forrest)

    # mean squared error
    mse_reg_random_forrest = mean_squared_error(test_output_regression, test_predict_random_forrest)

    # Neural Network Regression - Keras #######################################################
    # Create the model
    reg_keras = Sequential()
    reg_keras.add(Dense(30, input_dim=9, activation='relu'))
    reg_keras.add(Dense(30, activation='relu'))
    reg_keras.add(Dense(30, activation='relu'))
    reg_keras.add(Dense(30, activation='relu'))
    reg_keras.add(Dense(3, activation='linear'))

    # Configure the model and start training
    optimizer = keras.optimizers.Adam(lr=0.01)
    reg_keras.compile(optimizer=optimizer, metrics=['mse'], loss=['mse'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
    history_regression = reg_keras.fit(train_input_regression, train_output_regression, epochs=1000, batch_size=500, verbose=1,
                                       validation_split=0.15, callbacks=[early_stopping])

    # Test the model after training
    test_predict_keras_regression = reg_keras.predict(test_input_regression)

    # r2
    r2_network = r2_score(test_output_regression, test_predict_keras_regression)

    # mean squared error
    mse_network = mean_squared_error(test_output_regression, test_predict_keras_regression)

    # SVR ##########################################################################
    # modelSVR = SVR(kernel='linear', verbose=True)
    # wrapper = RegressorChain(modelSVR)
    # wrapper.fit(train_input_regression, train_output_regression)
    # pred_y_SVR = wrapper.predict(test_input_regression)
    #
    # # r2
    # r2_SVR = r2_score(test_output_regression, pred_y_SVR)
    #
    # # mean squared error
    # mse_SVR = mean_squared_error(test_output_regression, pred_y_SVR)

    end = 1
