import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
import tensorflow as tf

def convert_to_one_hot_encode(labels):
    return LabelBinarizer().fit_transform(labels)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def augment_images(data,labels):
    for i in range(len(data)):
        print(i)
        new_image = None

        rnd_choice = random.randint(0, 1)
        if rnd_choice == 0:
            new_image = tf.image.flip_left_right(data[i])
        else:
            new_image = tf.image.flip_up_down(data[i])


        #new_image = tf.image.random_brightness(new_image, 0.2, seed=None)
        new_image = tf.keras.preprocessing.image.random_rotation(new_image,20)
        rnd_choice = random.randint(0, 1)
        if rnd_choice == 0:
            new_image = tf.keras.preprocessing.image.random_shift(new_image,0.15,0.15)
        rnd_choice = random.randint(0, 1)
        #if rnd_choice == 0:
            #new_image = tf.keras.preprocessing.image.random_zoom(new_image,(0.8,0.8))
       # rnd_choice = random.randint(0, 1)
        #if rnd_choice == 0:
            #new_image = tf.image.random_contrast(new_image, 0.7, 0.9)

        data = np.concatenate((data, np.expand_dims(np.array(new_image),axis=0)), axis=0)
        labels = np.append(labels, labels[i])

        if i <= 5:
            figure, axis = plt.subplots(1, 2)
            axis[0].imshow(np.array(data[i]).T,interpolation='nearest')
            axis[0].set_title("Original Image")
            # For Sine Function
            axis[1].imshow(np.array(new_image).T , interpolation='nearest')
            axis[1].set_title("Image After Augmentation")
            plt.show()

    #return np.array(data),np.array(labels)
    np.save('data_augmented_2.npy', np.array(data))
    np.save('labels_augmented_2.npy', np.array(labels))

def fix_test_set(y):
    false_labels_indexes = [2405,2804,1227,5690,2592,9227]
    correct_labels = [7,6,6,5,5,10]
    for i in range(len(false_labels_indexes)):
        y[false_labels_indexes[i]] = correct_labels[i]-1
    return y


def plot_images(data,labels,labels_text=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],pred=np.array([])):
    data = np.array(data)
    amount = len(data)
    fig = plt.figure()
    columns = 4
    rows = 5
    selected_images = []
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        while True:
            random_image = random.randint(0, amount - 1)
            if random_image not in selected_images:
                selected_images.append(random_image)
                break
        plt.imshow(data[random_image].astype('uint8').T,interpolation='nearest')
        if pred.any():
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            ax.set_title("True Label: "+str(labels_text[np.argmax(labels[random_image])])+"\nPredicted Label: "+str(labels_text[np.argmax(pred[random_image])]))
        else:
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            ax.set_title("True Label: " + str(labels_text[labels[random_image]]))
    plt.show()

def plot_class_balance(y):
    from yellowbrick.target import ClassBalance
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
    visualizer.fit(y)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

def get_cifar_10_dataset(split=True):
    data = []
    labels = []

    for i in range(1,6):
        data.append(dict(unpickle("cifar-10-batches-py/data_batch_"+str(i)))[b'data'])
        labels.append(dict(unpickle("cifar-10-batches-py/data_batch_"+str(i)))[b'labels'])

    data_test =dict(unpickle("cifar-10-batches-py/test_batch"))[b'data']
    labels_test = dict(unpickle("cifar-10-batches-py/test_batch"))[b'labels']

    data = np.array(data)
    labels = np.array(labels)

    con = data[0]
    con_labels = labels[0]
    for i in range(1,len(data)):
        con = np.concatenate((con, data[i]), axis=0)
        con_labels = np.concatenate((con_labels, labels[i]), axis=0)

    data = con
    data = np.reshape(data, (data.shape[0], 3 , 32 , 32))
    labels = con_labels

    data_validation = data[0:10000,:]
    labels_validation = labels[0:10000]

    data = data[10000:50000,:]
    labels = labels[10000:50000]

    data_test = np.array(data_test)
    labels_test = np.array(labels_test)
    #labels_test = fix_test_set(labels_test)
    data_test = np.reshape(data_test, (data_test.shape[0], 3 , 32 , 32))

    plot_images(data_test,labels_test)

    if not os.path.isfile('data_augmented.npy') and not os.path.isfile('labels_augmented.npy.npy'):
        augment_images(data,labels)

    plot_class_balance(labels)
    plot_class_balance(labels_test)


    if split:
        labels = convert_to_one_hot_encode(labels)
        labels_test = convert_to_one_hot_encode(labels_test)
        labels_validation = convert_to_one_hot_encode(labels_validation)
        return data,labels,data_test,labels_test,data_validation,labels_validation,10
    else:
        data = np.concatenate((data, data_test), axis=0)
        data = np.concatenate((data, data_validation), axis=0)
        labels = np.concatenate((labels, labels_test), axis=0)
        labels = np.concatenate((labels, labels_validation), axis=0)
        labels = convert_to_one_hot_encode(labels)
        return data,labels,10