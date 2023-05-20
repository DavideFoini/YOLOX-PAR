from datetime import datetime

import numpy as np
import sys

import torchvision
from matplotlib import pyplot as plt
from tabulate import tabulate
from torchinfo import summary


def show_image(image, label, true, labels_list):
    img_text = ""
    for l, v, t in zip(labels_list, label, true):
        img_text += "{} t:{} p:{} \n".format(str(l), str(int(t.item())), str(int(v.item())))

    plt.imshow(image.permute(1, 2, 0))
    plt.text(1.02, 0.5, img_text, size=8, va='center', transform=plt.gca().transAxes)
    plt.show()

def plot_results(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, export=True, show=False, name="train_plot"):
    figure, axis = plt.subplots(2, 2)
    x = np.arange(0, len(train_loss_hist), 1)

    axis[0, 0].plot(x, train_loss_hist)
    axis[0, 0].set_title("train_loss_hist")

    axis[0, 1].plot(x, train_acc_hist)
    axis[0, 1].set_title("train_acc_hist")

    axis[1, 0].plot(x, val_loss_hist)
    axis[1, 0].set_title("val_loss_hist")

    axis[1, 1].plot(x, val_acc_hist)
    axis[1, 1].set_title("val_acc_hist")

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    if export:
        now = datetime.now()
        name = name + now.strftime(" %m%d%Y-%H%M%S")
        plt.savefig("./multi-label/results/plots/" + name + '.png')

    if show:  plt.show()

def plot_multiresults(history, export=True, show=False, name="train_plot"):
    figure, axis = plt.subplots(2, 2)
    x = np.arange(0, len(history[0]), 1)

    axis[0, 0].plot(x, history[0], label="label")
    axis[0, 0].plot(x, history[4], label="gender")
    axis[0, 0].plot(x, history[8], label="pov")
    axis[0, 0].set_title("train_loss_hist")
    axis[0, 0].legend(loc="upper right")

    axis[0, 1].plot(x, history[1], label="label")
    axis[0, 1].plot(x, history[5], label="gender")
    axis[0, 1].plot(x, history[9], label="pov")
    axis[0, 1].set_title("train_acc_hist")
    axis[0, 1].legend(loc="lower right")

    axis[1, 0].plot(x, history[2], label="label")
    axis[1, 0].plot(x, history[6], label="gender")
    axis[1, 0].plot(x, history[10], label="pov")
    axis[1, 0].set_title("val_loss_hist")
    axis[1, 0].legend(loc="upper right")

    axis[1, 1].plot(x, history[3], label="label")
    axis[1, 1].plot(x, history[7], label="gender")
    axis[1, 1].plot(x, history[11], label="pov")
    axis[1, 1].set_title("val_acc_hist")
    axis[1, 1].legend(loc="lower right")

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    if export:
        now = datetime.now()
        name = name + now.strftime(" %m%d%Y-%H%M%S")
        plt.savefig("./multi-label/results/plots/" + name + '.png')

    if show:  plt.show()

def model_summary(model, input_size = (1, 3, 250, 130)):
    print(summary(model, input_size))

def label_accuracy(true, pred, labels_acc):
    for i in range(0, len(labels_acc)):
        if true[i] == pred[i]: labels_acc[i] += 1

def show_labels_accuracy(labels_accuracy, labels_list, export = False, des=None):
    mean_acc = np.average(labels_accuracy)
    for i in range(0, len(labels_accuracy)):
        labels_accuracy[i] = "{}%".format(str(labels_accuracy[i]*100))
    table = zip(labels_list, labels_accuracy)
    table = tabulate(table, headers=("Label", "Accuracy"), tablefmt='fancy_grid')
    print(table)
    if export:
        now = datetime.now()
        name = "Labels_Accuracy-" + now.strftime(" %m%d%Y-%H%M%S") + ".txt"
        original_stdout = sys.stdout
        with open("./multi-label/results/accuracies/" + name, 'w') as f:
            sys.stdout = f
            print(des)
            print(table)
            print("Average Accuracy: {}".format(str(round(mean_acc*100,2))))
            sys.stdout = original_stdout

def norm_labels_accuracy(labels_accuracy, tot):
    labels_accuracy = [round(x / tot, 2) for x in labels_accuracy]
    return labels_accuracy
