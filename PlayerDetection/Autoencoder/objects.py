import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
import time
import sys
import os
from PIL import Image


class DataLoader():
  def __init__(self):
    self.image_path = 'Media\Full_Court_Photo_Set'
    self.data_dict = self.generate_parsed_dict()
    self.train_share = 0.7
    self.valid_share = 0.2
    self.test_share = 0.1
    self.full_data_list = []
    
    for datapoint in self.data_dict:
      entry_info = self.data_dict[datapoint]

      #Image Conversion -> [H, W, C]
      image = Image.open("{}\{}".format(self.image_path, entry_info['file_name']))
      data_img = np.asarray(image)

      #Label Conversion -> 
      full_poly_list = self.draw_polygon(vertex_list=entry_info["vertices"])
      poly_dict = self.generate_poly_slice_dict(full_poly_list)
      data_label = np.zeros([entry_info["height"], entry_info["width"]])
      for x_val in poly_dict.keys():
        min = np.min(poly_dict[x_val])
        max = np.max(poly_dict[x_val])
        data_label[min:max, x_val] = 1

      '''
      f, axarr = plt.subplots(1,2)
      axarr[0].imshow(data_img)
      axarr[1].imshow(data_label)
      '''
      #Final Formating
      pairing = (data_img, data_label)
      self.full_data_list.append(pairing)
      
    random.shuffle(self.full_data_list)
    split_index = round(len(self.full_data_list) * (self.valid_share + self.train_share))
    self.training_pool = self.full_data_list[0:split_index]
    self.test_pool = self.full_data_list[split_index:]
    print("Training Pool Size: {} | Test Pool Size: {}".format(len(self.training_pool), len(self.test_pool)))
  
  def generate_parsed_dict(self):
    info_descriptor = open("PlayerDetection\\Autoencoder\\Annotations\\100_Samples.json")
    info_dict = json.load(info_descriptor)
    parsed_dict = {}
    for entry in info_dict["images"]:
      parsed_dict[entry["id"]]={}
      parsed_dict[entry["id"]]["file_name"] = entry["file_name"]
      parsed_dict[entry["id"]]["width"] = entry["width"]
      parsed_dict[entry["id"]]["height"] = entry["height"]

    for id in parsed_dict.keys():
      vertex_list = []
      for keypoint in info_dict["annotations"]:
        if keypoint["image_id"] == id:
          keypoint_x = keypoint["keypoints"][0]
          keypoint_y = keypoint["keypoints"][1]
          #clip the keypoints

          if keypoint_x > parsed_dict[id]["width"]:
            keypoint_x = parsed_dict[id]["width"] - 1
          elif keypoint_x < 0:
            keypoint_x = 0

          if keypoint_y > parsed_dict[id]["height"]:
            keypoint_y = parsed_dict[id]["height"] - 1
          elif keypoint_y < 0:
            keypoint_y = 0

          vertex_list.append((keypoint_x, keypoint_y))
      parsed_dict[id]["vertices"] = vertex_list
    return parsed_dict


  def batchify(self, dataset, batch_size):
    random.shuffle(dataset)
    label_set = []
    input_set = []
    output_set = []
    for val in dataset:
      input, label = val
      input = cv2.resize(input, (720, 480), interpolation = cv2.INTER_AREA)
      label = cv2.resize(label, (720, 480), interpolation = cv2.INTER_AREA)
      input = np.swapaxes(input, 0, 1)
      input = np.swapaxes(input, 0, 2)
      input_set.append(input)
      label_set.append(label)
      if len(input_set) == batch_size:
        input_set = np.array(input_set)
        label_set = np.array(label_set)
        output_set.append((input_set, label_set))
        input_set = []
        label_set = []
    return output_set

  def training(self, batch_size):
    random.shuffle(self.training_pool)
    split_index = int(((self.train_share)/(self.train_share + self.valid_share))*len(self.training_pool))
    train_set = self.training_pool[0:split_index]
    valid_set = self.training_pool[split_index:]
    train_set = self.batchify(train_set, batch_size)
    valid_set = self.batchify(valid_set, batch_size)
    return train_set, valid_set
  
  def test(self, batch_size):
    test_set = self.batchify(self.test_pool, batch_size)
    return test_set
  
  def full(self):
    return(self.full_data_list)
  
  def all(self, batch_size):
    a, b = self.training(batch_size)
    c = self.test(batch_size)
    return(a, b, c)
  
  def display(self, dataset):
    for element in dataset:
      input, label = element
      f, axarr = plt.subplots(1,2)
      axarr[0].imshow(input)
      axarr[1].imshow(label)

  def bres_segment_count(self, x0, y0, x1, y1):
    point_list = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 2*int(x0 < x1) - 1
    sy = 2*int(y0 < y1) - 1
    
    err = dx - dy

    x_iter = x0
    y_iter = y0

    while True:
      if x_iter == x1 and y_iter == y1:
        break
      
      point_list.append((x_iter, y_iter))

      e2 = 2 * err
      if e2 > -dy:
        err -= dy
        x_iter += sx

      if e2 < dx:
        err += dx
        y_iter += sy

    return point_list

  def draw_polygon(self, vertex_list):
    full_poly_list = []
    for i, vert in enumerate(vertex_list):
      p1 = vert
      if i >= len(vertex_list) - 1: 
        p2 = vertex_list[0]
      else:
        p2 = vertex_list[i + 1]
      
      line = self.bres_segment_count(p1[0], p1[1], p2[0], p2[1])

      for val in line:
        full_poly_list.append(val)
    return full_poly_list

  def generate_poly_slice_dict(self, polygon_pixels):
    output_dict = {}
    for value in polygon_pixels:
      x_coord = value[0]
      y_coord = value[1]
      if x_coord not in output_dict.keys():
        output_dict[x_coord] = []
      output_dict[x_coord].append(y_coord)
    return output_dict

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride = 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2),
            nn.ReLU(),
            #nn.ConvTranspose2d(256, 256, 1, stride=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        #print("Encoding Output Shape: {}".format(x.shape))
        x = self.decoder(x)
        #print("Final Output Shape: {}".format(x.shape))
        return x