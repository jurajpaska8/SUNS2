import keras
import datetime
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

def export_images_to_npy(list_of_images, path):
    x_data = []
    filenames_all = []
    for root, dirs, filenames in os.walk(path):
        filenames_all = filenames

    for filename in list_of_images:
        if filename in filenames_all:
            full_path = os.path.join(path, filename)
            image = keras.preprocessing.image.load_img(full_path)
            image_arr = keras.preprocessing.image.img_to_array(image)
            image_arr /= 255.0
            x_data.append(image_arr)

    return np.array(x_data)


def plot_confusion_matrix(cm, labels, title):
    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    return fig


if __name__ == '__main__':
    # BONUS
    train_valid_size = 3000
    test_size = int(train_valid_size*0.2)
    train_shuffled = pd.read_csv('./data5/shuffled_train.csv', ',', index_col=0)
    train = train_shuffled[:train_valid_size]

    # image generators - first 3000 rows
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.2,
    )

    training_generator = image_generator.flow_from_dataframe(
        dataframe=train,
        directory='./data5/images',
        x_col='path',
        y_col='masterCategory',
        target_size=(80, 60),
        batch_size=200,
        subset='training'
    )

    validation_generator = image_generator.flow_from_dataframe(
        dataframe=train,
        directory='./data5/images',
        x_col='path',
        y_col='masterCategory',
        target_size=(80, 60),
        batch_size=200,
        subset='validation'
    )

    test_set = train_shuffled[train_valid_size:int(train_valid_size + test_size)]

    # save and load model
    loaded_model = keras.models.load_model('data5/models/keras20201209-125300')

    # own test set
    list_of_files = ['apparel1.jpg', 'apparel2.jpg', 'apparel3.jpg', 'foorwear3.jpg']
    test = export_images_to_npy(list_of_files, './data5/mytest')
    pred = loaded_model.predict(test)
    predicted_classes_indices = np.argmax(pred, axis=1)
    #['Accessories', 'Apparel', 'Footwear', 'Personal Care'], [0, 1, 2, 3]

    # confusion test
    list_of_files = test_set.path.tolist()
    test = export_images_to_npy(list_of_files, './data5/images')
    pred = loaded_model.predict(test)
    predicted_classes_indices = np.argmax(pred, axis=1)
    true_classes = test_set.masterCategory.replace(['Accessories', 'Apparel', 'Footwear', 'Personal Care'], [0, 1, 2, 3]).to_numpy()#true_classes = validation_generator.classes
    cm_test = confusion_matrix(true_classes, predicted_classes_indices)
    acc_test = accuracy_score(true_classes, predicted_classes_indices)

    # confusion validation
    list_of_files = validation_generator.filenames
    valid = export_images_to_npy(list_of_files, './data5/images')
    pred = loaded_model.predict(valid)
    predicted_classes_indices = np.argmax(pred, axis=1)
    true_classes = np.array(validation_generator.classes)
    cm_valid = confusion_matrix(true_classes, predicted_classes_indices)
    acc_valid = accuracy_score(true_classes, predicted_classes_indices)

    end = 1

