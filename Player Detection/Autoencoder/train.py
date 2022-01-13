"""
Contains: train_autoencoder(), plot_training_curve(), get_accuracy()

On Run: Trains and saves a model with the specifications provided
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt

from Autoencoder.objects import DataLoader, AutoEncoder

def train_auto_encoder(model, dataloader, epochs=100, batch_size=1, learning_rate = 1e-4):
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  train_acc = []
  val_acc = []
  train_loss = []
  val_loss = []

  valid_acc = 0
  valid_loss = 0

  for e in range(0, epochs):
    if e != 0:
      del train
      del val
    train, val = dataloader.training(batch_size)
    total_accuracy = 0
    
    for datapoint in train:
      start = time.perf_counter()
      input = datapoint[0]
      label = datapoint[1]

      #JUST FOR BATCH SIZE = 1 CASE | NOT YET BATCHED PROPERLY
      input = torch.from_numpy(input).float()
      label = torch.from_numpy(label).float()

      output = model(input)
      output = output.squeeze(1)

      #Accuracy Output
      sig_out = F.sigmoid(output)
      sig_output = (sig_out > 0.5).float().squeeze()
      
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      #Type 1 Accuracy:
      single_acc = int(torch.sum(sig_output == label)) / int(sig_output.numel())
      total_accuracy = total_accuracy + single_acc

      #Type 2 Accuracy
    
    final_acc = total_accuracy / len(train)
    train_loss.append(loss)
    train_acc.append(final_acc)

    #val_acc, val_loss = get_accuracy(model, val, batch_size, criterion)

    end = time.perf_counter()
    
    print("Train Acc: {} | Val Acc: {} | Train Loss: {} | Val Loss: {} | Time: {}".format(final_acc, valid_acc, loss, valid_loss,end-start))
    
    if e % (epochs / 5) == 0:
      #valid_acc, valid_loss = get_accuracy(model, val, batch_size, criterion)
      valid_acc = 0
      valid_loss = 0
      val_acc.append(valid_acc)
      val_loss.append(valid_loss)
      output_img = np.array(sig_output)
      output_img = np.array(output_img[0])

      plt.figure()
      plt.imshow(output_img)
    
  inter = epochs / 5
  torch.save(model.state_dict(), 'Autoencoder\Models\model_weights_100samples.pth')

def plot_training_curve(train_acc, train_loss, val_acc, val_loss):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.
    """
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
    
def get_accuracy(model, dataset, batch_size, criterion):
  #NOT PREPPED FOR BATCHES YET
  total_accuracy = 0
  loss_total = 0
  for datapoint in dataset:
    input = datapoint[0]
    label = datapoint[1]

    input = torch.from_numpy(input).float()
    label = torch.from_numpy(label).float()
    start = time.perf_counter()
    output = model(input)
    output = output.squeeze(1)
    loss = criterion(output, label)
    end = time.perf_counter()
    print("Time for Model = {}".format(end-start))
    
    #Accuracy Output
    sig_out = F.sigmoid(output)
    sig_output = (sig_out > 0.5).float()
    loss_total = loss_total + loss

    single_acc = int(torch.sum(sig_output == label)) / int(sig_output.numel())
    total_accuracy = total_accuracy + single_acc
  total_accuracy = total_accuracy / len(dataset)
  loss_total = loss_total / len(dataset)
  return total_accuracy, loss_total


if __name__ == "__main__": 
    print("Training Script for Autoencoder")
    print("***WIP***")