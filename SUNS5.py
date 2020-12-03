import os
import pandas as pd
import numpy as np
import keras.preprocessing
import datetime
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
import tensorflow as tf


if __name__ == '__main__':
    ########################################################################
    # 1 - LOAD, NORMALIZE, TRAIN+VALID and TEST SET#########################
    ########################################################################
    # load csv
    train_data = pd.read_csv('./data5/styles.csv', ',', error_bad_lines=False) # , index_col=0
    x_data = []
    filenames_all = []

    # load first 1000 images and save it and get filenames
    for root, dirs, filenames in os.walk('./data5/images'):
        filenames_all = sorted(filenames, key=lambda s: int(s[:-4]))
    #     for filename in filenames[:1000]:
    #         full_path = os.path.join(root, filename)
    #         image = keras.preprocessing.image.load_img(full_path)
    #         # image.show()
    #         image_arr = keras.preprocessing.image.img_to_array(image)
    #         image_arr /= 255.0
    #         x_data.append(image_arr)
    #
    # # save
    # np.save('./data5/images_encoded_1000', np.array(x_data))
    # x_data = np.load('./data5/images_encoded_1000.npy', allow_pickle=True)

    ################################################# load images with keras
    # load category and path
    train_labels = train_data[['id', 'masterCategory']].copy()
    train_labels['path'] = train_labels.apply(lambda row: str(row['id']) + '.jpg', axis=1)
    # sort
    train_labels = train_labels.sort_values(by=['id'])

    # drop rows with few values counts in df
    train_labels = train_labels[train_labels['masterCategory'] != 'Home']
    train_labels = train_labels[train_labels['masterCategory'] != 'Sporting Goods']
    train_labels = train_labels[train_labels['masterCategory'] != 'Free Items']

    # drop rows without image in filenames
    indices_to_drop = []
    for index in train_labels.index:
        if train_labels.loc[index, 'path'] not in filenames_all:
            indices_to_drop.append(index)

    train_labels_dropped = train_labels.drop(indices_to_drop)

    # image generators - first 900 rows
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.2,
    )

    training_generator = image_generator.flow_from_dataframe(
        dataframe=train_labels_dropped[:1000],
        directory='./data5/images',
        x_col='path',
        y_col='masterCategory',
        target_size=(80, 60),
        batch_size=50,
        subset='training'
    )

    validation_generator = image_generator.flow_from_dataframe(
        dataframe=train_labels_dropped[:1000],
        directory='./data5/images',
        x_col='path',
        y_col='masterCategory',
        target_size=(80, 60),
        batch_size=50,
        subset='validation'
    )

    test_set = train_labels_dropped[1000:1100]

    # keras network
    classes_train = len(training_generator.class_indices)
    classes_val = len(validation_generator.class_indices)

    optimizer = keras.optimizers.Adam(lr=0.001)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    log_dir = './logs/SUNS5/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = Sequential()
    model.add(Conv2D(60, (3, 3), padding='same', input_shape=(80, 60, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(120, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(120, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(120, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_train, activation='softmax'))

    # compile
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    step_size_train = training_generator.n // training_generator.batch_size
    step_size_validation = validation_generator.n // validation_generator.batch_size

    training = model.fit_generator(generator=training_generator,
                                   validation_data=validation_generator,
                                   steps_per_epoch=step_size_train,
                                   validation_steps=step_size_validation,
                                   epochs=30,
                                   callbacks=[early_stopping, tensorboard_callback])

    # val = n.load
    # pred = model.predict(val)
    # predicted_classes_indices = np.argmax(pred, axis=1)
    # true_classes = validation_generator.classes

    # conffusion true/predicted
    # figcm = go.Figure
    # figcm.show()

    # loss and score graphs

    # tensorboard --logdir logs/SUNS520201203-172909 TODO to venv and open localhost
    end = 1
