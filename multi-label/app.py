from datetime import datetime

import torch
from torch.utils.data import DataLoader

import datasets.PA100K as PA100K
from models import *
from utils import train, test, utils
from utils.utils import show_labels_accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTask4()
model.freeze_backbone()
model.to(device)
# utils.model_summary(multitask)

dataset = PA100K.Train('./PA-100K', device=device, multitask=True)
label, gender, pov, sleeve = train.train_multitask4(model, 20, 1e-2, dataset, plot=True, tolerance=1, resume=True, check_path="./multi-label/checkpoints/05182023-195520 MultiTaskV3_checkpoint10left.pt")

model.label = label
model.gender = gender
model.pov = pov
model.sleeve = sleeve

# label, gender, pov, sleeve = train.train_multitask4(model, 20, 1e-3, dataset, plot=True, tolerance=10, of_label=False)
#
# model.label = label
# model.gender = gender
# model.pov = pov
# model.sleeve = sleeve

dataset = PA100K.Test('./PA-100K', device=device, multitask=True)
test_acc, labels_acc = test.test_multitask(model, dataset)
show_labels_accuracy(labels_acc, dataset.labels_list, export=True, des="Multitask Genderconv")


# now = datetime.now()
# name = model.name + "" + now.strftime(" %m%d%Y-%H%M%S")
# torch.save(model, "./multi-label/results/models/{} - {}.pth".format(name, test_acc))

