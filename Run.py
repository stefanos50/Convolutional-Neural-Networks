import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import device
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
import DataPreprocessing
import HelperMethods
from CNN import CNN_Model


def run(X_train,y_train,X_test,y_test,X_val,y_val,num_classes,batch_size=32,arch=None,mode="train",num_channels=1):
    device = HelperMethods.initialize_hardware("cuda")


    #if mode == "train":
    X_train = np.concatenate((X_train, np.load('data_augmented.npy')[40000:80000,:]), axis=0)
    y_train = np.concatenate((y_train, DataPreprocessing.convert_to_one_hot_encode(np.load('labels_augmented.npy'))[40000:80000,:]), axis=0)

    X_train = np.concatenate((X_train, np.load('data_augmented_2.npy')[40000:80000,:]), axis=0)
    y_train = np.concatenate((y_train, DataPreprocessing.convert_to_one_hot_encode(np.load('labels_augmented_2.npy'))[40000:80000,:]), axis=0)

    #DataPreprocessing.plot_images(np.load('data_augmented.npy')[40000:80000,:],DataPreprocessing.convert_to_one_hot_encode(np.load('labels_augmented.npy'))[40000:80000])
    X_train, y_train = shuffle(X_train, y_train)

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)

    Traindataset = TensorDataset(X_train, y_train)
    Valdataset = TensorDataset(X_val, y_val)
    Testdataset = TensorDataset(X_test, y_test)

    trainDataLoader = DataLoader(Traindataset, shuffle=True, batch_size=batch_size)
    valDataLoader = DataLoader(Valdataset, batch_size=batch_size, shuffle=True)
    testDataLoader = DataLoader(Testdataset, batch_size=batch_size)

    if mode == "train":
        #lr=0.001 , wd=0.0001,adam
        model = CNN_Model(numChannels=num_channels, classes=num_classes, loss_function='cross-entropy',learning_rate=0.001, device=device, input_shape=X_train.shape, architecture=arch,weight_decay=0.0001,optimizer='adam',momentum=0.9).to(device)

        start = time.time()
        model.fit(100, trainDataLoader=trainDataLoader, valDataLoader=valDataLoader)
        end = time.time()

        accuracy_train = round(model.evaluate(trainDataLoader).cpu().detach().numpy().tolist() * 100, 2)
        accuracy_test = round(model.evaluate(testDataLoader).cpu().detach().numpy().tolist() * 100, 2)

        return model.History, end - start, accuracy_train, accuracy_test
    else:
        X_train = torch.Tensor(X_train)

        model = CNN_Model(numChannels=num_channels, classes=num_classes, loss_function='binary-cross-entropy', device=device, input_shape=X_train.shape, architecture=arch,learning_rate=0.001,optimizer="adam",momentum=0.7).to(device)
        model.load_state_dict(torch.load("saved_cnn_model.pth"))
        model.eval()
        print("Train accuracy of the saved model: "+str(round(model.evaluate(trainDataLoader).cpu().detach().numpy().tolist() * 100, 2))+"%")
        print("Test accuracy of the saved model: "+str(round(model.evaluate(testDataLoader).cpu().detach().numpy().tolist() * 100, 2))+"%")

        real_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        for i in range(num_classes):
            data_class = []
            labels_class = []
            for idx, (data, target) in enumerate(testDataLoader):
                for j in range(len(data)):
                    if np.argmax(target[j].numpy()) == i:
                        data_class.append(data[j].numpy().tolist())
                        labels_class.append(target[j].numpy().tolist())
            data_class = torch.Tensor(np.array(data_class))
            labels_class = torch.Tensor(np.array(labels_class))
            classdataset = TensorDataset(data_class, labels_class)
            classDataLoader = DataLoader(classdataset, batch_size=batch_size)
            print("Model Accuracy for class "+str(real_names[i])+": " + str(round(model.evaluate(classDataLoader).cpu().detach().numpy().tolist() * 100, 2)) + "%")
