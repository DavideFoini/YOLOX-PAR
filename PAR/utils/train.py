import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping, OverfittingLoss
from .utils import plot_results, plot_multiresults


def train(model, num_epochs, lr, dataset, val_split=0.1, plot=False, resume=False, check_path=None, only_gender=False):
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping()
    overfitting = OverfittingLoss(5)
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(non_frozen_parameters, lr=lr)
    best_model = None

    if not resume:
        train_loss_hist = []
        train_acc_hist = []
        val_loss_hist = []
        val_acc_hist = []
    else:
        model, optimizer, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, num_epochs = load_checkpoint(
            model, optimizer, check_path)

    train_data, val_data = torch.utils.data.random_split(dataset, (1 - val_split, val_split))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0)
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader))
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        for (inputs, targets) in train_loop:
            batch_acc = []

            optimizer.zero_grad()
            outputs = model(inputs)
            if only_gender: targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            epoch_train_loss += train_loss
            eval_out, eval_target = torch.sigmoid(outputs).cpu().detach().numpy(), targets.cpu().detach().numpy()
            eval_out = np.round(eval_out)
            for pred, truth in zip(eval_out, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_acc.append(accuracy)

            batch_mean_acc = np.mean(batch_acc)
            epoch_train_accuracy += batch_mean_acc
            train_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Training")
            # train_loop.set_postfix(train_loss=train_loss, train_acc=batch_mean_acc)

        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_accuracy = epoch_train_accuracy / len(train_loader)
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_accuracy)

        # validation
        model.eval()
        val_loop = tqdm(val_loader, total=len(val_loader))
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        with torch.no_grad():
            for (inputs, targets) in val_loop:
                batch_acc = []
                outputs = model(inputs)

                if only_gender: targets = targets.unsqueeze(1)

                loss = criterion(outputs, targets)
                val_loss = loss.item()
                epoch_val_loss += val_loss
                eval_out, eval_target = torch.sigmoid(outputs).cpu().detach().numpy(), targets.cpu().detach().numpy()
                eval_out = np.round(eval_out)
                for pred, truth in zip(eval_out, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_acc.append(accuracy)

                batch_mean_acc = np.mean(batch_acc)
                epoch_val_accuracy += batch_mean_acc
                val_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Validation")
                # val_loop.set_postfix(val_loss=val_loss, val_acc=batch_mean_acc)

        epoch_val_loss = epoch_val_loss / len(val_loader)
        epoch_val_accuracy = epoch_val_accuracy / len(val_loader)
        print("Epoch {}: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_loss, 2), round(epoch_train_accuracy, 2), round(epoch_val_loss, 2), round(epoch_val_accuracy, 2)))
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_accuracy)

        overfitting(epoch_val_loss)
        if overfitting.best:
            best_model = model
            print("###### New best model! ######")
        if overfitting.stop:
            print("Stopped due to overfitting:", epoch)
            if plot: plot_results(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist)
            return best_model

        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            if plot: plot_results(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist)
            return best_model

        # SAVE CHECKPOINT EVERY 5 EPOCHS
        if epoch % 5 == 0: save_checkpoint(model, optimizer, num_epochs - epoch, train_loss_hist, train_acc_hist,
                                           val_loss_hist, val_acc_hist)

    if plot: plot_results(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist)
    return best_model

def train_multitask(model, num_epochs, lr, dataset, val_split=0.1, plot=False, resume=False,
                    check_path=None, label_w=1, gender_w=1, tolerance=5):
    criterion = nn.BCEWithLogitsLoss()
    labelOverfitting = OverfittingLoss(tolerance)
    genderOverfitting = OverfittingLoss(tolerance)
    best_model = None

    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(non_frozen_parameters, lr=lr)

    if not resume:
        train_label_loss_hist = []
        train_gender_loss_hist = []
        train_label_acc_hist = []
        train_gender_acc_hist = []
        val_label_loss_hist = []
        val_gender_loss_hist = []
        val_label_acc_hist = []
        val_gender_acc_hist = []
    else:
        model, optimizer, train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist, \
            train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist, \
            num_epochs = load_checkpoint(model, optimizer, check_path, multitask=True)

    train_data, val_data = torch.utils.data.random_split(dataset, (1 - val_split, val_split))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0)
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader))
        epoch_train_label_loss = 0.0
        epoch_train_gender_loss = 0.0
        epoch_train_label_accuracy = 0.0
        epoch_train_gender_accuracy = 0.0
        for (inputs, t_genders, t_labels) in train_loop:
            batch_label_acc = []
            batch_gender_acc = []

            optimizer.zero_grad()
            genders, labels = model(inputs)
            labels_loss = criterion(labels, t_labels)
            gender_loss = criterion(genders.squeeze(), t_genders)
            loss = (labels_loss * label_w) + (gender_loss * gender_w)
            loss.backward()
            optimizer.step()

            epoch_train_label_loss += labels_loss.item()
            epoch_train_gender_loss += gender_loss.item()

            eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
            eval_lab = np.round(eval_lab)

            for pred, truth in zip(eval_lab, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_label_acc.append(accuracy)

            eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
            eval_gen = np.round(eval_gen)
            for pred, truth in zip(eval_gen, eval_target):
                if pred == truth:
                    batch_gender_acc.append(1)
                else:
                    batch_gender_acc.append(0)

            batch_mean_label_acc = np.mean(batch_label_acc)
            batch_mean_gender_acc = np.mean(batch_gender_acc)
            epoch_train_label_accuracy += batch_mean_label_acc
            epoch_train_gender_accuracy += batch_mean_gender_acc

            train_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Training")


        epoch_train_label_loss = epoch_train_label_loss / len(train_loader)
        epoch_train_gender_loss = epoch_train_gender_loss / len(train_loader)

        epoch_train_label_accuracy = epoch_train_label_accuracy / len(train_loader)
        epoch_train_gender_accuracy = epoch_train_gender_accuracy / len(train_loader)

        train_label_loss_hist.append(epoch_train_label_loss)
        train_gender_loss_hist.append(epoch_train_gender_loss)

        train_label_acc_hist.append(epoch_train_label_accuracy)
        train_gender_acc_hist.append(epoch_train_gender_accuracy)

        # validation
        model.eval()
        val_loop = tqdm(val_loader, total=len(val_loader))
        epoch_val_label_loss = 0.0
        epoch_val_gender_loss = 0.0
        epoch_val_label_accuracy = 0.0
        epoch_val_gender_accuracy = 0.0
        with torch.no_grad():
            for (inputs, t_genders, t_labels) in val_loop:
                batch_label_acc = []
                batch_gender_acc = []
                genders, labels = model(inputs)
                labels_loss = criterion(labels, t_labels)
                gender_loss = criterion(genders.squeeze(), t_genders)

                epoch_val_label_loss += labels_loss.item()
                epoch_val_gender_loss += gender_loss.item()

                eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
                eval_lab = np.round(eval_lab)
                for pred, truth in zip(eval_lab, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_label_acc.append(accuracy)

                eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
                eval_gen = np.round(eval_gen)
                for pred, truth in zip(eval_gen, eval_target):
                    if pred == truth:
                        batch_gender_acc.append(1)
                    else:
                        batch_gender_acc.append(0)

                batch_mean_label_acc = np.mean(batch_label_acc)
                batch_mean_gender_acc = np.mean(batch_gender_acc)
                epoch_val_label_accuracy += batch_mean_label_acc
                epoch_val_gender_accuracy += batch_mean_gender_acc

                val_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Validation")

        epoch_val_label_loss = epoch_val_label_loss / len(val_loader)
        epoch_val_gender_loss = epoch_val_gender_loss / len(val_loader)

        epoch_val_label_accuracy = epoch_val_label_accuracy / len(val_loader)
        epoch_val_gender_accuracy = epoch_val_gender_accuracy / len(val_loader)

        val_label_loss_hist.append(epoch_val_label_loss)
        val_gender_loss_hist.append(epoch_val_gender_loss)

        val_label_acc_hist.append(epoch_val_label_accuracy)
        val_gender_acc_hist.append(epoch_val_gender_accuracy)

        print("Epoch {} label: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_label_loss, 2), round(epoch_train_label_accuracy, 2), round(epoch_val_label_loss, 2), round(epoch_val_label_accuracy, 2)))
        print("Epoch {} gender: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_gender_loss, 2), round(epoch_train_gender_accuracy, 2), round(epoch_val_gender_loss, 2), round(epoch_val_gender_accuracy, 2)))


        labelOverfitting(epoch_val_label_loss)
        if labelOverfitting.stop:
            print("Stopped due to label overfitting:", epoch)
            if plot:  plot_multiresults(train_label_loss_hist, train_label_acc_hist, val_label_loss_hist,
                                        val_label_acc_hist,
                                        train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist,
                                        val_gender_acc_hist)
            return best_model

        genderOverfitting(epoch_val_gender_loss)
        if genderOverfitting.best:
            best_model = model
            print("###### New best model! ######")
        if genderOverfitting.stop:
            print("Stopped due to gender overfitting:", epoch)
            if plot:  plot_multiresults(train_label_loss_hist, train_label_acc_hist, val_label_loss_hist,
                                        val_label_acc_hist,
                                        train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist,
                                        val_gender_acc_hist)
            return best_model

        if epoch % 5 == 0: save_checkpoint_multitask(model, optimizer, num_epochs - epoch, train_label_loss_hist,
                                                     train_label_acc_hist, val_label_loss_hist,
                                                     val_label_acc_hist, train_gender_loss_hist, train_gender_acc_hist,
                                                     val_gender_loss_hist, val_gender_acc_hist)

    if plot: plot_multiresults(train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist,
                               train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist)

    return best_model

def train_multitask2(model, num_epochs, lr, dataset, val_split=0.1, plot=False, resume=False,
                    check_path=None, label_w=1, gender_w=1, tolerance=5, of_label=True, of_gender=True):
    criterion = nn.BCEWithLogitsLoss()

    labelOverfitting = OverfittingLoss(tolerance)
    genderOverfitting = OverfittingLoss(tolerance)

    best_label = None
    best_gender = None

    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(non_frozen_parameters, lr=lr)

    if not resume:
        train_label_loss_hist = []
        train_gender_loss_hist = []
        train_label_acc_hist = []
        train_gender_acc_hist = []
        val_label_loss_hist = []
        val_gender_loss_hist = []
        val_label_acc_hist = []
        val_gender_acc_hist = []
    else:
        model, optimizer, train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist, \
            train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist, \
            num_epochs = load_checkpoint(model, optimizer, check_path, multitask=True)

    train_data, val_data = torch.utils.data.random_split(dataset, (1 - val_split, val_split))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0)
    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader))
        epoch_train_label_loss = 0.0
        epoch_train_gender_loss = 0.0
        epoch_train_label_accuracy = 0.0
        epoch_train_gender_accuracy = 0.0
        for (inputs, t_genders, t_labels) in train_loop:
            batch_label_acc = []
            batch_gender_acc = []

            optimizer.zero_grad()
            genders, labels = model(inputs)
            labels_loss = criterion(labels, t_labels)
            gender_loss = criterion(genders.squeeze(), t_genders)
            loss = (labels_loss * label_w) + (gender_loss * gender_w)
            loss.backward()
            optimizer.step()

            epoch_train_label_loss += labels_loss.item()
            epoch_train_gender_loss += gender_loss.item()

            eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
            eval_lab = np.round(eval_lab)

            for pred, truth in zip(eval_lab, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_label_acc.append(accuracy)

            eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
            eval_gen = np.round(eval_gen)
            for pred, truth in zip(eval_gen, eval_target):
                if pred == truth:
                    batch_gender_acc.append(1)
                else:
                    batch_gender_acc.append(0)

            batch_mean_label_acc = np.mean(batch_label_acc)
            batch_mean_gender_acc = np.mean(batch_gender_acc)
            epoch_train_label_accuracy += batch_mean_label_acc
            epoch_train_gender_accuracy += batch_mean_gender_acc

            train_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Training")


        epoch_train_label_loss = epoch_train_label_loss / len(train_loader)
        epoch_train_gender_loss = epoch_train_gender_loss / len(train_loader)

        epoch_train_label_accuracy = epoch_train_label_accuracy / len(train_loader)
        epoch_train_gender_accuracy = epoch_train_gender_accuracy / len(train_loader)

        train_label_loss_hist.append(epoch_train_label_loss)
        train_gender_loss_hist.append(epoch_train_gender_loss)

        train_label_acc_hist.append(epoch_train_label_accuracy)
        train_gender_acc_hist.append(epoch_train_gender_accuracy)

        # validation
        model.eval()
        val_loop = tqdm(val_loader, total=len(val_loader))
        epoch_val_label_loss = 0.0
        epoch_val_gender_loss = 0.0
        epoch_val_label_accuracy = 0.0
        epoch_val_gender_accuracy = 0.0
        with torch.no_grad():
            for (inputs, t_genders, t_labels) in val_loop:
                batch_label_acc = []
                batch_gender_acc = []
                genders, labels = model(inputs)
                labels_loss = criterion(labels, t_labels)
                gender_loss = criterion(genders.squeeze(), t_genders)

                epoch_val_label_loss += labels_loss.item()
                epoch_val_gender_loss += gender_loss.item()

                eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
                eval_lab = np.round(eval_lab)
                for pred, truth in zip(eval_lab, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_label_acc.append(accuracy)

                eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
                eval_gen = np.round(eval_gen)
                for pred, truth in zip(eval_gen, eval_target):
                    if pred == truth:
                        batch_gender_acc.append(1)
                    else:
                        batch_gender_acc.append(0)

                batch_mean_label_acc = np.mean(batch_label_acc)
                batch_mean_gender_acc = np.mean(batch_gender_acc)
                epoch_val_label_accuracy += batch_mean_label_acc
                epoch_val_gender_accuracy += batch_mean_gender_acc

                val_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Validation")


        l = len(val_loader)
        epoch_val_label_loss = epoch_val_label_loss / l
        epoch_val_gender_loss = epoch_val_gender_loss / l

        epoch_val_label_accuracy = epoch_val_label_accuracy / l
        epoch_val_gender_accuracy = epoch_val_gender_accuracy / l

        val_label_loss_hist.append(epoch_val_label_loss)
        val_gender_loss_hist.append(epoch_val_gender_loss)

        val_label_acc_hist.append(epoch_val_label_accuracy)
        val_gender_acc_hist.append(epoch_val_gender_accuracy)

        print("Epoch {} label: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_label_loss, 2), round(epoch_train_label_accuracy, 2), round(epoch_val_label_loss, 2), round(epoch_val_label_accuracy, 2)))
        print("Epoch {} gender: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_gender_loss, 2), round(epoch_train_gender_accuracy, 2), round(epoch_val_gender_loss, 2), round(epoch_val_gender_accuracy, 2)))

        if of_label:
            labelOverfitting(epoch_val_label_loss)
            if genderOverfitting.best:
                best_label = model.label
                print("###### New best label model! ######")
            if labelOverfitting.stop:
                print("Stopped due to label overfitting:", epoch)
                if plot:  plot_multiresults(train_label_loss_hist, train_label_acc_hist, val_label_loss_hist,
                                            val_label_acc_hist,
                                            train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist,
                                            val_gender_acc_hist)
                return best_label, best_gender

        if of_gender:
            genderOverfitting(epoch_val_gender_loss)
            if genderOverfitting.best:
                best_gender = model.gender
                print("###### New best gender model! ######")
            if genderOverfitting.stop:
                print("Stopped due to gender overfitting:", epoch)
                if plot:  plot_multiresults(train_label_loss_hist, train_label_acc_hist, val_label_loss_hist,
                                            val_label_acc_hist,
                                            train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist,
                                            val_gender_acc_hist)
                return best_label, best_gender

        if epoch % 5 == 0: save_checkpoint_multitask(model, optimizer, num_epochs - epoch, train_label_loss_hist,
                                                     train_label_acc_hist, val_label_loss_hist,
                                                     val_label_acc_hist, train_gender_loss_hist, train_gender_acc_hist,
                                                     val_gender_loss_hist, val_gender_acc_hist)

    if plot: plot_multiresults(train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist,
                               train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist)

    return best_label, best_gender

def train_multitask3(model, num_epochs, lr, dataset, val_split=0.1, plot=False, resume=False,
                    check_path=None, label_w=1, gender_w=1, pov_w=1, tolerance=5, of_label=True, of_gender=True, of_pov=True):
    criterion = nn.BCEWithLogitsLoss()

    labelOverfitting = OverfittingLoss(tolerance)
    genderOverfitting = OverfittingLoss(tolerance)
    povOverfitting = OverfittingLoss(tolerance)

    best_label = None
    best_gender = None
    best_pov = None
    history = ()

    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(non_frozen_parameters, lr=lr)

    if not resume:
        train_label_loss_hist = []
        train_gender_loss_hist = []
        train_pov_loss_hist = []
        train_label_acc_hist = []
        train_gender_acc_hist = []
        train_pov_acc_hist = []
        val_label_loss_hist = []
        val_gender_loss_hist = []
        val_pov_loss_hist = []
        val_label_acc_hist = []
        val_gender_acc_hist = []
        val_pov_acc_hist = []
    else:
        model, optimizer, history, num_epochs = load_checkpoint(model, optimizer, check_path, multitask=True)
        train_label_loss_hist = history['train_label_loss_hist']
        train_gender_loss_hist = history['train_gender_loss_hist']
        train_pov_loss_hist = history['train_pov_loss_hist']
        train_label_acc_hist = history['train_label_acc_hist']
        train_gender_acc_hist = history['train_gender_acc_hist']
        train_pov_acc_hist = history['train_pov_acc_hist']
        val_label_loss_hist = history['val_label_loss_hist']
        val_gender_loss_hist = history['val_gender_loss_hist']
        val_pov_loss_hist = history['val_pov_loss_hist']
        val_label_acc_hist = history['val_label_acc_hist']
        val_gender_acc_hist = history['val_gender_acc_hist']
        val_pov_acc_hist = history['val_pov_acc_hist']

    train_data, val_data = torch.utils.data.random_split(dataset, (1 - val_split, val_split))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0)

    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader))

        epoch_train_label_loss = 0.0
        epoch_train_gender_loss = 0.0
        epoch_train_pov_loss = 0.0
        epoch_train_label_accuracy = 0.0
        epoch_train_gender_accuracy = 0.0
        epoch_train_pov_accuracy = 0.0

        for (inputs, t_genders, t_povs, t_labels) in train_loop:
            batch_label_acc = []
            batch_gender_acc = []
            batch_pov_acc = []

            optimizer.zero_grad()
            genders, povs, labels = model(inputs)

            labels_loss = criterion(labels, t_labels)
            gender_loss = criterion(genders.squeeze(), t_genders)
            pov_loss = criterion(povs, t_povs)

            loss = (labels_loss * label_w) + (gender_loss * gender_w) + (pov_loss * pov_w)

            loss.backward()
            optimizer.step()

            epoch_train_label_loss += labels_loss.item()
            epoch_train_gender_loss += gender_loss.item()
            epoch_train_pov_loss += pov_loss.item()

            eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
            eval_lab = np.round(eval_lab)

            for pred, truth in zip(eval_lab, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_label_acc.append(accuracy)

            eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
            eval_gen = np.round(eval_gen)
            for pred, truth in zip(eval_gen, eval_target):
                if pred == truth:
                    batch_gender_acc.append(1)
                else:
                    batch_gender_acc.append(0)

            eval_pov, eval_target = torch.sigmoid(povs).cpu().detach().numpy(), t_povs.cpu().detach().numpy()
            eval_pov = np.round(eval_pov)

            for pred, truth in zip(eval_pov, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_pov_acc.append(accuracy)

            batch_mean_label_acc = np.mean(batch_label_acc)
            batch_mean_gender_acc = np.mean(batch_gender_acc)
            batch_mean_pov_acc = np.mean(batch_pov_acc)

            epoch_train_label_accuracy += batch_mean_label_acc
            epoch_train_gender_accuracy += batch_mean_gender_acc
            epoch_train_pov_accuracy += batch_mean_pov_acc

            train_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Training")

        length = len(train_loader)

        epoch_train_label_loss = epoch_train_label_loss / length
        epoch_train_gender_loss = epoch_train_gender_loss / length
        epoch_train_pov_loss = epoch_train_pov_loss / length

        epoch_train_label_accuracy = epoch_train_label_accuracy / length
        epoch_train_gender_accuracy = epoch_train_gender_accuracy / length
        epoch_train_pov_accuracy = epoch_train_pov_accuracy / length

        train_label_loss_hist.append(epoch_train_label_loss)
        train_gender_loss_hist.append(epoch_train_gender_loss)
        train_pov_loss_hist.append(epoch_train_pov_loss)

        train_label_acc_hist.append(epoch_train_label_accuracy)
        train_gender_acc_hist.append(epoch_train_gender_accuracy)
        train_pov_acc_hist.append(epoch_train_pov_accuracy)

        # validation
        model.eval()
        val_loop = tqdm(val_loader, total=len(val_loader))

        epoch_val_label_loss = 0.0
        epoch_val_gender_loss = 0.0
        epoch_val_pov_loss = 0.0

        epoch_val_label_accuracy = 0.0
        epoch_val_gender_accuracy = 0.0
        epoch_val_pov_accuracy = 0.0

        with torch.no_grad():
            for (inputs, t_genders, t_povs, t_labels) in val_loop:
                batch_label_acc = []
                batch_gender_acc = []
                batch_pov_acc = []
                genders, povs, labels = model(inputs)
                labels_loss = criterion(labels, t_labels)
                gender_loss = criterion(genders.squeeze(), t_genders)
                pov_loss = criterion(povs, t_povs)

                epoch_val_label_loss += labels_loss.item()
                epoch_val_gender_loss += gender_loss.item()
                epoch_val_pov_loss += pov_loss.item()

                eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
                eval_lab = np.round(eval_lab)
                for pred, truth in zip(eval_lab, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_label_acc.append(accuracy)

                eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
                eval_gen = np.round(eval_gen)
                for pred, truth in zip(eval_gen, eval_target):
                    if pred == truth:
                        batch_gender_acc.append(1)
                    else:
                        batch_gender_acc.append(0)

                eval_pov, eval_target = torch.sigmoid(povs).cpu().detach().numpy(), t_povs.cpu().detach().numpy()
                eval_pov = np.round(eval_pov)
                for pred, truth in zip(eval_pov, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_pov_acc.append(accuracy)

                batch_mean_label_acc = np.mean(batch_label_acc)
                batch_mean_gender_acc = np.mean(batch_gender_acc)
                batch_mean_pov_acc = np.mean(batch_pov_acc)
                epoch_val_label_accuracy += batch_mean_label_acc
                epoch_val_gender_accuracy += batch_mean_gender_acc
                epoch_val_pov_accuracy += batch_mean_pov_acc

                val_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Validation")


        l = len(val_loader)
        epoch_val_label_loss = epoch_val_label_loss / l
        epoch_val_gender_loss = epoch_val_gender_loss / l
        epoch_val_pov_loss = epoch_val_pov_loss / l

        epoch_val_label_accuracy = epoch_val_label_accuracy / l
        epoch_val_gender_accuracy = epoch_val_gender_accuracy / l
        epoch_val_pov_accuracy = epoch_val_pov_accuracy / l

        val_label_loss_hist.append(epoch_val_label_loss)
        val_gender_loss_hist.append(epoch_val_gender_loss)
        val_pov_loss_hist.append(epoch_val_pov_loss)

        val_label_acc_hist.append(epoch_val_label_accuracy)
        val_gender_acc_hist.append(epoch_val_gender_accuracy)
        val_pov_acc_hist.append(epoch_val_pov_accuracy)

        print("Epoch {} label: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_label_loss, 2), round(epoch_train_label_accuracy, 2), round(epoch_val_label_loss, 2), round(epoch_val_label_accuracy, 2)))
        print("Epoch {} gender: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_gender_loss, 2), round(epoch_train_gender_accuracy, 2), round(epoch_val_gender_loss, 2), round(epoch_val_gender_accuracy, 2)))
        print("Epoch {} pov: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_pov_loss, 2), round(epoch_train_pov_accuracy, 2), round(epoch_val_pov_loss, 2), round(epoch_val_pov_accuracy, 2)))

        history = (train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist,
                   train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist,
                   train_pov_loss_hist, train_pov_acc_hist, val_pov_loss_hist, val_pov_acc_hist)
        if of_label:
            labelOverfitting(epoch_val_label_loss)
            if genderOverfitting.best:
                best_label = model.label
                print("###### New best label model! ######")
            if labelOverfitting.stop:
                print("Stopped due to label overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov

        if of_gender:
            genderOverfitting(epoch_val_gender_loss)
            if genderOverfitting.best:
                best_gender = model.gender
                print("###### New best gender model! ######")
            if genderOverfitting.stop:
                print("Stopped due to gender overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov

        if of_pov:
            povOverfitting(epoch_val_pov_loss)
            if povOverfitting.best:
                best_pov = model.pov
                print("###### New best pov model! ######")
            if povOverfitting.stop:
                print("Stopped due to pov overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov

        if epoch % 5 == 0: save_checkpoint_multitask(model, optimizer, num_epochs - epoch, history)

    if plot: plot_multiresults(history)

    return best_label, best_gender, best_pov

def train_multitask4(model, num_epochs, lr, dataset, val_split=0.1, plot=False, resume=False, tolerance=5, check_path=None,
                     label_w=1, gender_w=1, pov_w=1, sleeve_w=1,
                     of_label=True, of_gender=True, of_pov=True, of_sleeve=True):
    criterion = nn.BCEWithLogitsLoss()

    labelOverfitting = OverfittingLoss(tolerance)
    genderOverfitting = OverfittingLoss(tolerance)
    povOverfitting = OverfittingLoss(tolerance)
    sleeveOverfitting = OverfittingLoss(tolerance)

    best_label = None
    best_gender = None
    best_pov = None
    best_sleeve = None

    history = ()

    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(non_frozen_parameters, lr=lr)

    if not resume:
        train_label_loss_hist = []
        train_gender_loss_hist = []
        train_pov_loss_hist = []
        train_sleeve_loss_hist = []
        train_label_acc_hist = []
        train_gender_acc_hist = []
        train_pov_acc_hist = []
        train_sleeve_acc_hist = []
        val_label_loss_hist = []
        val_gender_loss_hist = []
        val_pov_loss_hist = []
        val_sleeve_loss_hist = []
        val_label_acc_hist = []
        val_gender_acc_hist = []
        val_pov_acc_hist = []
        val_sleeve_acc_hist = []
    else:
        model, optimizer, history, num_epochs = load_checkpoint(model, optimizer, check_path, multitask=True)
        train_label_loss_hist = history[0]
        train_gender_loss_hist = history[4]
        train_pov_loss_hist = history[8]
        train_sleeve_loss_hist = history[12]
        train_label_acc_hist = history[1]
        train_gender_acc_hist = history[5]
        train_pov_acc_hist = history[9]
        train_sleeve_acc_hist = history[13]
        val_label_loss_hist = history[2]
        val_gender_loss_hist = history[6]
        val_pov_loss_hist = history[10]
        val_sleeve_loss_hist = history[14]
        val_label_acc_hist = history[3]
        val_gender_acc_hist = history[7]
        val_pov_acc_hist = history[11]
        val_sleeve_acc_hist = history[15]

    train_data, val_data = torch.utils.data.random_split(dataset, (1 - val_split, val_split))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0)

    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader))

        epoch_train_label_loss = 0.0
        epoch_train_gender_loss = 0.0
        epoch_train_pov_loss = 0.0
        epoch_train_sleeve_loss = 0.0

        epoch_train_label_accuracy = 0.0
        epoch_train_gender_accuracy = 0.0
        epoch_train_pov_accuracy = 0.0
        epoch_train_sleeve_accuracy = 0.0

        for (inputs, t_genders, t_povs, t_sleeves, t_labels) in train_loop:
            batch_label_acc = []
            batch_gender_acc = []
            batch_pov_acc = []
            batch_sleeve_acc = []

            optimizer.zero_grad()
            genders, povs, sleeves, labels = model(inputs)

            labels_loss = criterion(labels, t_labels)
            gender_loss = criterion(genders.squeeze(), t_genders)
            pov_loss = criterion(povs, t_povs)
            sleeve_loss = criterion(sleeves, t_sleeves)

            loss = (labels_loss * label_w) + (gender_loss * gender_w) + \
                   (pov_loss * pov_w) + (sleeve_loss * sleeve_w)

            loss.backward()
            optimizer.step()

            epoch_train_label_loss += labels_loss.item()
            epoch_train_gender_loss += gender_loss.item()
            epoch_train_pov_loss += pov_loss.item()
            epoch_train_sleeve_loss += sleeve_loss.item()

            eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
            eval_lab = np.round(eval_lab)
            for pred, truth in zip(eval_lab, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_label_acc.append(accuracy)

            eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
            eval_gen = np.round(eval_gen)
            for pred, truth in zip(eval_gen, eval_target):
                if pred == truth:
                    batch_gender_acc.append(1)
                else:
                    batch_gender_acc.append(0)

            eval_pov, eval_target = torch.sigmoid(povs).cpu().detach().numpy(), t_povs.cpu().detach().numpy()
            eval_pov = np.round(eval_pov)
            for pred, truth in zip(eval_pov, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_pov_acc.append(accuracy)

            eval_sleeve, eval_target = torch.sigmoid(sleeves).cpu().detach().numpy(), t_sleeves.cpu().detach().numpy()
            eval_sleeve = np.round(eval_sleeve)
            for pred, truth in zip(eval_sleeve, eval_target):
                accuracy = accuracy_score(truth, pred, normalize=True)
                batch_sleeve_acc.append(accuracy)

            batch_mean_label_acc = np.mean(batch_label_acc)
            batch_mean_gender_acc = np.mean(batch_gender_acc)
            batch_mean_pov_acc = np.mean(batch_pov_acc)
            batch_mean_sleeve_acc = np.mean(batch_sleeve_acc)

            epoch_train_label_accuracy += batch_mean_label_acc
            epoch_train_gender_accuracy += batch_mean_gender_acc
            epoch_train_pov_accuracy += batch_mean_pov_acc
            epoch_train_sleeve_accuracy += batch_mean_sleeve_acc

            train_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Training")

        length = len(train_loader)

        epoch_train_label_loss = epoch_train_label_loss / length
        epoch_train_gender_loss = epoch_train_gender_loss / length
        epoch_train_pov_loss = epoch_train_pov_loss / length
        epoch_train_sleeve_loss = epoch_train_sleeve_loss / length

        epoch_train_label_accuracy = epoch_train_label_accuracy / length
        epoch_train_gender_accuracy = epoch_train_gender_accuracy / length
        epoch_train_pov_accuracy = epoch_train_pov_accuracy / length
        epoch_train_sleeve_accuracy = epoch_train_sleeve_accuracy / length

        train_label_loss_hist.append(epoch_train_label_loss)
        train_gender_loss_hist.append(epoch_train_gender_loss)
        train_pov_loss_hist.append(epoch_train_pov_loss)
        train_sleeve_loss_hist.append(epoch_train_sleeve_loss)

        train_label_acc_hist.append(epoch_train_label_accuracy)
        train_gender_acc_hist.append(epoch_train_gender_accuracy)
        train_pov_acc_hist.append(epoch_train_pov_accuracy)
        train_sleeve_acc_hist.append(epoch_train_sleeve_accuracy)

        # validation
        model.eval()
        val_loop = tqdm(val_loader, total=len(val_loader))

        epoch_val_label_loss = 0.0
        epoch_val_gender_loss = 0.0
        epoch_val_pov_loss = 0.0
        epoch_val_sleeve_loss = 0.0

        epoch_val_label_accuracy = 0.0
        epoch_val_gender_accuracy = 0.0
        epoch_val_pov_accuracy = 0.0
        epoch_val_sleeve_accuracy = 0.0

        with torch.no_grad():
            for (inputs, t_genders, t_povs, t_sleeves, t_labels) in val_loop:
                batch_label_acc = []
                batch_gender_acc = []
                batch_pov_acc = []
                batch_sleeve_acc = []

                genders, povs, sleeves, labels = model(inputs)
                labels_loss = criterion(labels, t_labels)
                gender_loss = criterion(genders.squeeze(), t_genders)
                pov_loss = criterion(povs, t_povs)
                sleeve_loss = criterion(sleeves, t_sleeves)

                epoch_val_label_loss += labels_loss.item()
                epoch_val_gender_loss += gender_loss.item()
                epoch_val_pov_loss += pov_loss.item()
                epoch_val_sleeve_loss += sleeve_loss.item()

                eval_lab, eval_target = torch.sigmoid(labels).cpu().detach().numpy(), t_labels.cpu().detach().numpy()
                eval_lab = np.round(eval_lab)
                for pred, truth in zip(eval_lab, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_label_acc.append(accuracy)

                eval_gen, eval_target = torch.sigmoid(genders).cpu().detach().numpy(), t_genders.cpu().detach().numpy()
                eval_gen = np.round(eval_gen)
                for pred, truth in zip(eval_gen, eval_target):
                    if pred == truth:
                        batch_gender_acc.append(1)
                    else:
                        batch_gender_acc.append(0)

                eval_pov, eval_target = torch.sigmoid(povs).cpu().detach().numpy(), t_povs.cpu().detach().numpy()
                eval_pov = np.round(eval_pov)
                for pred, truth in zip(eval_pov, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_pov_acc.append(accuracy)

                eval_sleeve, eval_target = torch.sigmoid(sleeves).cpu().detach().numpy(), t_sleeves.cpu().detach().numpy()
                eval_sleeve = np.round(eval_sleeve)
                for pred, truth in zip(eval_sleeve, eval_target):
                    accuracy = accuracy_score(truth, pred, normalize=True)
                    batch_sleeve_acc.append(accuracy)

                batch_mean_label_acc = np.mean(batch_label_acc)
                batch_mean_gender_acc = np.mean(batch_gender_acc)
                batch_mean_pov_acc = np.mean(batch_pov_acc)
                batch_mean_sleeve_acc = np.mean(batch_sleeve_acc)

                epoch_val_label_accuracy += batch_mean_label_acc
                epoch_val_gender_accuracy += batch_mean_gender_acc
                epoch_val_pov_accuracy += batch_mean_pov_acc
                epoch_val_sleeve_accuracy += batch_mean_sleeve_acc

                val_loop.set_description(f"Epoch [{epoch}/{num_epochs}] - Validation")


        l = len(val_loader)
        epoch_val_label_loss = epoch_val_label_loss / l
        epoch_val_gender_loss = epoch_val_gender_loss / l
        epoch_val_pov_loss = epoch_val_pov_loss / l
        epoch_val_sleeve_loss = epoch_val_sleeve_loss / l

        epoch_val_label_accuracy = epoch_val_label_accuracy / l
        epoch_val_gender_accuracy = epoch_val_gender_accuracy / l
        epoch_val_pov_accuracy = epoch_val_pov_accuracy / l
        epoch_val_sleeve_accuracy = epoch_val_sleeve_accuracy / l

        val_label_loss_hist.append(epoch_val_label_loss)
        val_gender_loss_hist.append(epoch_val_gender_loss)
        val_pov_loss_hist.append(epoch_val_pov_loss)
        val_sleeve_loss_hist.append(epoch_val_sleeve_loss)

        val_label_acc_hist.append(epoch_val_label_accuracy)
        val_gender_acc_hist.append(epoch_val_gender_accuracy)
        val_pov_acc_hist.append(epoch_val_pov_accuracy)
        val_sleeve_acc_hist.append(epoch_val_sleeve_accuracy)

        print("Epoch {} label: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_label_loss, 2), round(epoch_train_label_accuracy, 2), round(epoch_val_label_loss, 2), round(epoch_val_label_accuracy, 2)))
        print("Epoch {} gender: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_gender_loss, 2), round(epoch_train_gender_accuracy, 2), round(epoch_val_gender_loss, 2), round(epoch_val_gender_accuracy, 2)))
        print("Epoch {} pov: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_pov_loss, 2), round(epoch_train_pov_accuracy, 2), round(epoch_val_pov_loss, 2), round(epoch_val_pov_accuracy, 2)))
        print("Epoch {} sleeve: t_loss {} t_acc {} - v_loss {} v_acc {}".format(epoch, round(epoch_train_sleeve_loss, 2), round(epoch_train_sleeve_accuracy, 2), round(epoch_val_sleeve_loss, 2), round(epoch_val_sleeve_accuracy, 2)))

        history = (train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist,
                   train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist,
                   train_pov_loss_hist, train_pov_acc_hist, val_pov_loss_hist, val_pov_acc_hist,
                   train_sleeve_loss_hist, train_sleeve_acc_hist, val_sleeve_loss_hist, val_sleeve_acc_hist)

        if of_label:
            labelOverfitting(epoch_val_label_loss)
            if labelOverfitting.best:
                best_label = model.label
                print("###### New best label model! ######")
            if labelOverfitting.stop:
                print("Stopped due to label overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov, best_sleeve

        if of_gender:
            genderOverfitting(epoch_val_gender_loss)
            if genderOverfitting.best:
                best_gender = model.gender
                print("###### New best gender model! ######")
            if genderOverfitting.stop:
                print("Stopped due to gender overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov, best_sleeve

        if of_pov:
            povOverfitting(epoch_val_pov_loss)
            if povOverfitting.best:
                best_pov = model.pov
                print("###### New best pov model! ######")
            if povOverfitting.stop:
                print("Stopped due to pov overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov, best_sleeve

        if of_sleeve:
            sleeveOverfitting(epoch_val_sleeve_loss)
            if sleeveOverfitting.best:
                best_sleeve = model.sleeve
                print("###### New best sleeve model! ######")
            if sleeveOverfitting.stop:
                print("Stopped due to sleeve overfitting:", epoch)
                if plot:  plot_multiresults(history)
                return best_label, best_gender, best_pov, best_sleeve

        if epoch % 5 == 0: save_checkpoint_multitask(model, optimizer, num_epochs - epoch, history)

    if plot: plot_multiresults(history)

    return best_label, best_gender, best_pov, best_sleeve

def save_checkpoint(model, optimizer, epoch, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist):
    print("Saving checkpoint with {} epochs left".format(epoch))
    now = datetime.now()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_hist': train_loss_hist,
        'train_acc_hist': train_acc_hist,
        'val_loss_hist': val_loss_hist,
        'val_acc_hist': val_acc_hist
    }, "./multi-label/checkpoints/" + now.strftime("%m%d%Y-%H%M%S ") + model.name + "_checkpoint" + str(
        epoch) + "left.pt")


def save_checkpoint_multitask(model, optimizer, epoch, history):
    print("Saving checkpoint with {} epochs left".format(epoch))
    now = datetime.now()
    hist = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_label_loss_hist': history[0],
        'train_label_acc_hist': history[1],
        'train_gender_loss_hist': history[4],
        'train_gender_acc_hist': history[5],
        'train_pov_loss_hist': history[8],
        'train_pov_acc_hist': history[9],
        'train_sleeve_loss_hist': history[12],
        'train_sleeve_acc_hist': history[13],
        'val_label_loss_hist': history[2],
        'val_label_acc_hist': history[3],
        'val_gender_loss_hist': history[6],
        'val_gender_acc_hist': history[7],
        'val_pov_loss_hist': history[10],
        'val_pov_acc_hist': history[11],
        'val_sleeve_loss_hist': history[14],
        'val_sleeve_acc_hist': history[15]
    }
    torch.save(hist, "./multi-label/checkpoints/" + now.strftime("%m%d%Y-%H%M%S ") + model.name + "_checkpoint" + str(
        epoch) + "left.pt")


def load_checkpoint(model, optimizer, path, multitask=False):
    if os.path.isfile(path):
        print("loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if multitask:
            train_label_loss_hist = checkpoint['train_label_loss_hist']
            train_label_acc_hist = checkpoint['train_label_acc_hist']
            train_gender_loss_hist = checkpoint['train_gender_loss_hist']
            train_gender_acc_hist = checkpoint['train_gender_acc_hist']
            train_pov_loss_hist = checkpoint['train_pov_loss_hist']
            train_pov_acc_hist = checkpoint['train_pov_acc_hist']
            train_sleeve_loss_hist = checkpoint['train_sleeve_loss_hist']
            train_sleeve_acc_hist = checkpoint['train_sleeve_acc_hist']
            val_label_loss_hist = checkpoint['val_label_loss_hist']
            val_label_acc_hist = checkpoint['val_label_acc_hist']
            val_gender_loss_hist = checkpoint['val_gender_loss_hist']
            val_gender_acc_hist = checkpoint['val_gender_acc_hist']
            val_pov_loss_hist = checkpoint['val_pov_loss_hist']
            val_pov_acc_hist = checkpoint['val_pov_acc_hist']
            val_sleeve_loss_hist = checkpoint['val_sleeve_loss_hist']
            val_sleeve_acc_hist = checkpoint['val_sleeve_acc_hist']

            history = (train_label_loss_hist, train_label_acc_hist, val_label_loss_hist, val_label_acc_hist,
                       train_gender_loss_hist, train_gender_acc_hist, val_gender_loss_hist, val_gender_acc_hist,
                       train_pov_loss_hist, train_pov_acc_hist, val_pov_loss_hist, val_pov_acc_hist,
                       train_sleeve_loss_hist, train_sleeve_acc_hist, val_sleeve_loss_hist, val_sleeve_acc_hist)

            return model, optimizer, history, epoch
        else:
            train_loss_hist = checkpoint['train_loss_hist']
            train_acc_hist = checkpoint['train_acc_hist']
            val_loss_hist = checkpoint['val_loss_hist']
            val_acc_hist = checkpoint['val_acc_hist']
            return model, optimizer, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, epoch
    else:
        print("=> no checkpoint found at '{}'".format(path))
        return None
