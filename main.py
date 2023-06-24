# This is a sample Python script.
from sklearn.model_selection import KFold
from torchvision import models
import DataPreprocessing
import HelperMethods
import Run
from torch.nn import ReLU,Dropout,Tanh,Sigmoid,LeakyReLU,Mish,ELU
from torch.nn import LogSoftmax,Softmax
import torch

from CNN import CNN_Model

print(torch.__version__)
exit(1)
def plot_results(history,train_accuracy,test_accuracy,time,type):
    HelperMethods.plot_result(history['loss'], history['val_loss'], "Loss Plot "+str(type), "Epochs", "Loss",
                              "Train Loss", "Val Loss")
    HelperMethods.plot_result(history['accuracy'], history['val_accuracy'], "Accuracy Plot "+str(type), "Epochs",
                              "Accuracy", "Train Accuracy", "Val Accuracy")


    print("\n")
    print(str(type)+" train accuracy: " + str(train_accuracy) + "%")
    print(str(type)+" test accuracy: " + str(test_accuracy) + "%")
    print(str(type)+" fit execution time: " + str(time) + "s")

def history_average(history):
    loss = []
    val_loss = []
    accuracy = []
    val_accuracy = []
    epoch_time = []
    for dict in history:
        loss.append(dict['loss'])
        val_loss.append(dict['val_loss'])
        accuracy.append(dict['accuracy'])
        val_accuracy.append(dict['val_accuracy'])
        epoch_time.append(dict['epoch_time'])

    history_new = {}
    history_new['loss'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*loss)]
    history_new['val_loss'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*val_loss)]
    history_new['accuracy'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*accuracy)]
    history_new['val_accuracy'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*val_accuracy)]
    history_new['epoch_time'] = [sum(sub_list) / len(sub_list) for sub_list in zip(*epoch_time)]
    return history_new

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
K=0
mode = "eval"
#Adam,wd=0.0001,w=100
test_param = "Mish + LogSoftMax"
out_chan = 48 #48

network_architecture = { 'Conv2d_1': {'out_channels' : out_chan, 'kernel_size' : (3,3)},
                         'BatchNorm2d_1': {},
                        'activation_1': {'function': ReLU()},
                        'Dropout_2':{'function':Dropout(p=0.1)},
                        'Conv2d_2': {'out_channels' : out_chan, 'kernel_size' : (3,3)},
                         'BatchNorm2d_2': {},
                         'activation_2': {'function': ReLU()},
                        'Dropout_3':{'function':Dropout(p=0.1)},
                         'MaxPool_2': {'kernel_size' : (2,2),'stride': (2,2)},

                        'Conv2d_3': {'out_channels' : out_chan*2, 'kernel_size' : (2,2)},
                        'BatchNorm2d_3': {},
                        'activation_5': {'function': ReLU()},
                        'Dropout_4':{'function':Dropout(p=0.1)},
                         'Conv2d_4': {'out_channels' : out_chan*2, 'kernel_size' : (2,2)},
                         'BatchNorm2d_4': {},
                         'activation_6': {'function': ReLU()},
                         'Dropout_5':{'function':Dropout(p=0.1)},
                         'MaxPool_5': {'kernel_size' : (2,2),'stride': (2,2)},

                         'Conv2d_5': {'out_channels': out_chan*4, 'kernel_size': (2, 2)},
                         'BatchNorm2d_5': {},
                         'activation_7': {'function': ReLU()},
                         'Dropout_6':{'function':Dropout(p=0.1)},
                         'Conv2d_6': {'out_channels': out_chan*4, 'kernel_size': (2, 2)},
                         'BatchNorm2d_6': {},
                         'activation_8': {'function': ReLU()},
                         'Dropout_7':{'function':Dropout(p=0.1)},
                         'MaxPool_6': {'kernel_size': (3, 3), 'stride': (2, 2)},



                         'Flatten': {},
                         'Linear_1': {'out_features' : 350},
                         'BatchNorm1d_7': {},
                         'activation_3': {'function': ReLU()},
                         'Dropout_1':{'function':Dropout(p=0.5)},
                         'Linear_3': {},
                         'activation_4': {'function': LogSoftmax(dim=1)}}
if mode=='train':
    if K>0:
        X, y, classes = DataPreprocessing.get_cifar_10_dataset(False)
        kf = KFold(n_splits=K)
        historyls = []
        timels = []
        test_accuracyls = []
        train_accuracyls = []
        for train_index, test_index in kf.split(X):
            train_data_x, test_data_x = X[train_index], X[test_index]
            train_data_y, test_data_y = y[train_index], y[test_index]

            history, time, train_accuracy, test_accuracy = Run.run(train_data_x, train_data_y, test_data_x, test_data_y,num_classes=classes,arch=network_architecture,num_channels=train_data_x.shape[1])
            historyls.append(history)
            timels.append(time)
            test_accuracyls.append(test_accuracy)
            train_accuracyls.append(train_accuracy)

        hist_avg = history_average(historyls)
        plot_results(hist_avg, sum(train_accuracyls)/len(train_accuracyls), sum(test_accuracyls)/len(test_accuracyls), sum(timels)/len(timels), str(K)+"-Cross-Validation Results")
        print("Average epoch time: " + str(sum(hist_avg['epoch_time']) / len(hist_avg['epoch_time'])))
        HelperMethods.plot_result_single(hist_avg['epoch_time'], "Epoch Time Plot", "Epochs", "Time Elapsed")

        hist_avg['train_accuracy'] = sum(train_accuracyls)/len(train_accuracyls)
        hist_avg['test_accuracy'] =  sum(test_accuracyls)/len(test_accuracyls)
        hist_avg['time'] = sum(timels)/len(timels)
        hist_avg['architecture'] = network_architecture
        hist_avg['param'] = test_param
        HelperMethods.save_history(hist_avg)
    else:
        train_data_x, train_data_y,test_data_x,test_data_y,val_data_x,val_data_y, classes = DataPreprocessing.get_cifar_10_dataset(True)
        history, time, train_accuracy, test_accuracy = Run.run(train_data_x, train_data_y, test_data_x, test_data_y,val_data_x,val_data_y, num_classes=classes,arch=network_architecture,num_channels=train_data_x.shape[1])
        plot_results(history,train_accuracy,test_accuracy,time,"No Cross-Validation Results")
        print("Average epoch time: "+str(sum(history['epoch_time'])/len(history['epoch_time'])))
        HelperMethods.plot_result_single(history['epoch_time'], "Epoch Time Plot", "Epochs", "Time Elapsed")

        history['train_accuracy'] = train_accuracy
        history['test_accuracy'] =  test_accuracy
        history['time'] = time
        history['architecture'] = network_architecture
        history['param'] = test_param
        HelperMethods.save_history(history)

else:
    train_data_x, train_data_y,test_data_x,test_data_y,val_data_x,val_data_y,classes = DataPreprocessing.get_cifar_10_dataset(True)
    Run.run(train_data_x, train_data_y, test_data_x, test_data_y,val_data_x,val_data_y,num_classes=classes, arch=network_architecture,mode=mode,num_channels=train_data_x.shape[1])