import os
import keras
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from classification_models.keras import Classifiers
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt


genres = ['War', 'Fantasy', 'Mystery', 'TV Movie', 'Science Fiction', 'Western'
    , 'Comedy', 'Documentary', 'Crime', 'Action', 'Music', 'Adventure', 'Family'
    , 'Thriller', 'History', 'Horror', 'Foreign', 'Drama', 'Romance', 'Animation']

def plot(training_Accu, testing_Accu):
    # Create count of the number of epochs
    epoch_count = range(1, len(training_Accu) + 1)
    # Visualize loss history
    plt.figure()
    plt.plot(epoch_count, training_Accu, 'r--')
    plt.plot(epoch_count, testing_Accu, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

def preprocess(img):
    img = resize(img, (224, 224))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img

def run_model():

    # Load data
    train_partition = pd.read_csv('data/train.csv')
    validate_partition = pd.read_csv('data/validate.csv')
    train_labels = pd.read_csv('data/train_label.csv')
    validate_labels = pd.read_csv('data/validate_label.csv')
	
    # Convert to lists
    train_partition = train_partition.values.tolist()
    validate_partition = validate_partition.values.tolist()
    train_labels = train_labels.values.tolist()
    validate_labels = validate_labels.values.tolist()

    print('Loading train data')
    x_tr = []
    y_tr = []
    for i in range(len(train_partition)):
        if os.path.exists('data/images/' + str((train_partition[i])[0]) + '.jpg'):
            img = imread(
                'data/images/' + str((train_partition[i])[0]) + '.jpg')
            img = preprocess(img)
            if img.shape != (224,224,3):
              continue
            x_tr.append(img)
            y_tr.append(train_labels[i])

    x_tr = np.array(x_tr, dtype='float32')
    y_tr = np.array(y_tr, dtype='uint8')

    print('training data length = ' + str(len(x_tr)))

    print('Loading validation data')
    x_val = []
    y_val = []
    for i in range(len(validate_partition)):
        if os.path.exists('data/images/' + str((validate_partition[i])[0]) + '.jpg'):
            img = imread(
                'data/images/' + str((validate_partition[i])[0]) + '.jpg')
            img = preprocess(img)
            if img.shape != (224,224,3):
              continue
            x_val.append(img)
            y_val.append(validate_labels[i])
            

    x_val = np.array(x_val, dtype='float32')
    y_val = np.array(y_val, dtype='uint8')

    print('Validating data length = ' + str(len(x_val)))

    print('Creating model')
    num_classes = len(genres)
    
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(num_classes, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    hist = model.fit(x_tr, y_tr, batch_size=32, epochs=50, validation_data=(x_val, y_val))
    
    plot(hist.history['accuracy'], hist.history['val_accuracy'])

if __name__ == '__main__':
    run_model()