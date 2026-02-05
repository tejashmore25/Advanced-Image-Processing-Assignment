import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random

def eval(model, test_dataloader, loss_fn, device, showProgress = False):
    model.eval()
    correct = 0
    total = 0
    cum_loss = 0.0

    if showProgress:
        bar = tqdm(test_dataloader)
    else:
        bar = test_dataloader

    with torch.no_grad():
        for inp, label in bar:
            inp, label = inp.to(device), label.to(device)
            outputs = model(inp)
            cum_loss += loss_fn(outputs, label).item() * inp.size(0)
            
            _, predicted = torch.max(outputs, 1)    # returns maxValue, Index of the maximum logit value
            
            total += label.size(0)
            correct += (predicted == label).sum().item()

    epoch_test_acc = correct / total
    epoch_test_loss = cum_loss / total

    return epoch_test_acc, epoch_test_loss

def train_model(model, optimizer, loss_fn, train_dataloader, num_epochs, device, test_per_epoch = False, test_dataloader = None):
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    test_loss_list = []
    
    for epoch in range(num_epochs):

        model.train()

        cum_loss = 0.0
        correct = 0
        total = 0

        # Bar Visualizer
        bar = tqdm(train_dataloader)

        for inp, label in bar:
            inp, label = inp.to(device), label.to(device)

            optimizer.zero_grad()

            outputs = model(inp)
            _, predicted = torch.max(outputs, 1)
            loss = loss_fn(outputs, label)

            loss.backward()
            optimizer.step()

            cum_loss += loss.item() * inp.size(0)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        epoch_loss = cum_loss / total
        train_loss_list.append(epoch_loss)
        epoch_acc = correct / total
        train_acc_list.append(epoch_acc)

        if test_per_epoch:
            test_acc, test_loss = eval(model, test_dataloader, loss_fn)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            print(f"""Epoch: {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.3f} | Test Acc: {test_acc:.3f}""")
        else:
            print(f"""Epoch: {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.3f}""")

    return model, train_loss_list, train_acc_list, test_acc_list, test_loss_list