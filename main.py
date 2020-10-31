# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.svm import SVC
from scipy import stats



def load_data(filepath):
    return pd.read_csv(filepath, ',')


def drop_all_na(dataframe):
    return pd.DataFrame.dropna(dataframe)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # LOAD
    train_data = load_data('./data/train.csv')
    test_data = load_data('./data/test.csv')

    # CLEAN NA
    train_data_clean = drop_all_na(train_data)
    test_data_clean = drop_all_na(test_data)

    # ENCODE GENRE
    train_data_encoded = train_data_clean.copy()
    test_data_encoded = test_data_clean.copy()

    unique_genre = test_data_clean.playlist_genre.unique()
    for i, val in enumerate(unique_genre):
        train_data_encoded = train_data_encoded.replace(val, i)
        test_data_encoded = test_data_encoded.replace(val, i)

    # NORMALIZE
    sc = StandardScaler()
    train_data_normalized = train_data_encoded[['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy(deep=True)
    train_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']] = sc.fit_transform(train_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']])

    test_data_normalized = test_data_encoded[['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy(deep=True)
    test_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']] = sc.fit_transform(test_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']])

    # OUTLIERS
    train_data_valid = train_data_normalized[(np.abs(stats.zscore(train_data_normalized)) < 3).all(axis=1)]

    # MLP - PERCEPTRON
    # Train / Test data
    X_train = train_data_valid.drop(['playlist_genre'], axis=1).head(500)
    Y_train = train_data_valid.playlist_genre.head(500)
    X_test = test_data_normalized.drop(['playlist_genre'], axis=1)
    Y_test = test_data_normalized.playlist_genre

    # Create the model
    model = Sequential()
    model.add(Dense(150, input_dim=12, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax')) #sigmoid  ?

    # Configure the model and start training
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(X_train.to_numpy(), pd.get_dummies(Y_train), epochs=1000, batch_size=100, verbose=1, validation_split=0.15, callbacks=[early_stopping])

    # Test the model after training
    test_results = model.evaluate(X_test.to_numpy(), pd.get_dummies(Y_test), verbose=1)
    pred_Y = model.predict(X_test.to_numpy())
    pred_Y_classes = model.predict_classes(X_test.to_numpy())
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

    # Plot training process
    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # accuracy on test data
    cm_test = confusion_matrix(Y_test, pred_Y_classes)
    acc_test = accuracy_score(Y_test, pred_Y_classes)

    # SVM
    # params = dict(gamma=[0.1], C=[1])
    #
    # cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    # grid = GridSearchCV(SVC(verbose=1), param_grid=params, cv=cv, verbose=1)
    # grid.fit(X_train, Y_train)
    # scores = grid.cv_results_['mean_test_score'].reshape(1, 1)

    # simple SVM
    # clf = make_pipeline(StandardScaler(), SVC(C=1, gamma=0.1, verbose=1))
    # svm_history = clf.fit(X_train, Y_train)
    # svm_score_train = clf.score(X_train, Y_train)
    # svm_score_test = clf.score(X_test, Y_test)
    # print(f'Test results : {svm_score_test} - Train result: {svm_score_train}')
    end = 1
