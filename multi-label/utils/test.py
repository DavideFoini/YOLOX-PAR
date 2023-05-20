import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm

from .utils import label_accuracy, norm_labels_accuracy

def test(model, dataset):
    model.eval()
    labels_acc = [0] * len(dataset.labels_list)
    test_accuracy = 0

    for i in range(0, len(dataset)):
        sample, target = dataset[i]
        sample = sample[None, :, :, :]
        output = model(sample)
        output = np.squeeze(output)
        eval_out, eval_target = torch.sigmoid(output).cpu().detach().numpy(), target.cpu().detach().numpy()
        eval_out = np.round(eval_out)
        test_accuracy += balanced_accuracy_score(eval_target, eval_out)
        label_accuracy(eval_target, eval_out, labels_acc)
        # print("Computed sample {}/{}".format(i, len(dataset)))

    labels_acc = norm_labels_accuracy(labels_acc, len(dataset))

    return round(test_accuracy / len(dataset), 2), labels_acc

def test_onlygender(model, dataset):
    model.eval()
    test_accuracy = 0
    loop = tqdm(range(0, len(dataset)))

    for i in loop:
        sample, target = dataset[i]
        sample = sample[None, :, :, :]
        output = model(sample)
        output = np.squeeze(output)
        eval_out, eval_target = torch.sigmoid(output).cpu().detach().numpy(), target.cpu().detach().numpy()
        eval_out = np.round(eval_out)
        test_accuracy += (eval_out == eval_target)
        loop.set_description(f"Tested sample [{i}/{len(dataset)}]")

    return round(test_accuracy / len(dataset), 2)

def test_multitask(model, dataset):
    model.eval()
    labels_acc = [0] * len(dataset.labels_list)
    test_accuracy = 0
    loop = tqdm(range(0, len(dataset)))
    for i in loop:
        sample, t_gender, t_pov, t_sleeve, t_label = dataset[i]
        sample = sample[None, :, :, :]

        gender, pov, sleeve, label = model(sample)
        gender, pov, sleeve, label = np.squeeze(gender), np.squeeze(pov),  np.squeeze(sleeve), np.squeeze(label)

        gender, t_gender = torch.sigmoid(gender).cpu().detach().numpy(), t_gender.cpu().detach().numpy()
        gender = np.round(gender)

        pov, t_pov = torch.sigmoid(pov).cpu().detach().numpy(), t_pov.cpu().detach().numpy()
        pov = np.round(pov)

        sleeve, t_sleeve = torch.sigmoid(sleeve).cpu().detach().numpy(), t_sleeve.cpu().detach().numpy()
        sleeve = np.round(sleeve)

        label, t_label = torch.sigmoid(label).cpu().detach().numpy(), t_label.cpu().detach().numpy()
        label = np.round(label)

        t_label = np.insert(t_label, 0, t_gender)
        label = np.insert(label, 0, gender)

        t_label = np.insert(t_label, 4, t_pov)
        label = np.insert(label, 4, pov)

        t_label = np.insert(t_label, 13, t_sleeve)
        label = np.insert(label, 13, sleeve)

        test_accuracy += balanced_accuracy_score(t_label, label)
        label_accuracy(t_label, label, labels_acc)
        # print("Computed sample {}/{}".format(i, len(dataset)))
        loop.set_description(f"Tested sample [{i}/{len(dataset)}]")

    labels_acc = norm_labels_accuracy(labels_acc, len(dataset))

    return round(test_accuracy / len(dataset), 2), labels_acc



