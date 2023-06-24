import os
import pickle
import torch
from matplotlib import pyplot as plt
import cpuinfo

def plot_result(x,y,title,y_label,x_label,x_legend,y_legend):
    plt.plot(x)
    plt.plot(y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend([x_legend, y_legend], loc='upper left')
    plt.show()

def plot_result_multiple(x,title,y_label,x_label,params):
    for i in range(len(x)):
        plt.plot(x[i])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend([param for param in params], loc='upper left')
    plt.show()

def plot_result_single(x,title,y_label,x_label):
    plt.plot(x)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def initialize_hardware(hw_choice='cuda'):
    if hw_choice == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available() and hw_choice == 'cuda':
        print('Using device: ', device)
        print('Using gpu: ',torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print('Using device: ', device)
    return device

def save_history(hist):
    try:
        prev_dict = retrieve_history()
    except:
        prev_dict = None
    if prev_dict == None:
        f = open("saved/history.pkl","wb")
        prev_dict = {}
        prev_dict[1] = hist
        pickle.dump(prev_dict, f)
        print('History saved successfully to file')
        f.close()
        return
    else:
        f = open("saved/history.pkl", "wb")
        prev_dict[list(prev_dict)[-1]+1] = hist
        pickle.dump(prev_dict, f)
        print('History saved successfully to file')
        f.close()
        return

def retrieve_history():
    # open a file, where you stored the pickled data
    file = open('saved/history.pkl', 'rb')

    # dump information to that file
    hist = pickle.load(file)

    # close the file
    file.close()

    return hist