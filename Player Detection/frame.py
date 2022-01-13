
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from PIL import Image
from Autoencoder.objects import AutoEncoder

#Getting the 
def convert_img_to_mask(img, model):
  input = cv2.resize(img, (720, 480), interpolation = cv2.INTER_AREA)
  input = np.swapaxes(input, 0, 1)
  input = np.swapaxes(input, 0, 2)
  input = torch.from_numpy(input).float()
  input = input.unsqueeze(0)

  output = model(input)
  sig_out = F.sigmoid(output)
  sig_output = (sig_out > 0.5).float()
  output_img = np.array(sig_output)
  output_img = np.array(output_img[0])
  output_img = output_img.squeeze(0)
  output_img = cv2.resize(output_img, (1920, 1080))
  #plt.imshow(output_img, cmap="gray")
  return output_img

def get_people(boxes):
  boxes_people = boxes[boxes["name"] == "person"]
  boxes_people["xcenter"] = boxes_people["xcenter"].astype(int)
  boxes_people["ycenter"] = boxes_people["ycenter"].astype(int)
  boxes_people["width"] = boxes_people["width"].astype(int)
  boxes_people["height"] = boxes_people["height"].astype(int)
  return boxes_people

def get_necessary_models():
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load('Player Detection\\Autoencoder\\Models\\model_weights_100samples.pth'))
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return autoencoder, yolov5_model

def get_center_of_play(img, court_detector, player_detector, show_results=False):
  overlap_perc = 0.05
  match_ratio = 0.9

  output = player_detector(img)
  pd_output = output.pandas()
  boxes = get_people(pd_output.xywh[0])
  extracted_img = pd_output.imgs[0]

  mask = convert_img_to_mask(extracted_img, court_detector)
  top_half = mask[:int(mask.shape[0] / 2)]

  index_list = []
  for index, row in boxes.iterrows():
    height = row["height"]
    width = row["width"]
    
    left_ind = row["xcenter"] - int(width/2) 
    right_ind = row["xcenter"] + int(width/2)
    top_ind = row["ycenter"] + int(height/2) - int(height * overlap_perc)
    bot_ind = row["ycenter"] + int(height/2)
    #print("{}. Top: {} | Bot: {} | Left: {} | Right: {}".format(index, top_ind, bot_ind, left_ind, right_ind))
    segment = mask[top_ind:bot_ind]
    seg_size = segment.size
    sumation = segment.sum()
    if float(sumation/seg_size) > match_ratio:
      index_list.append(index)
  
  updated_boxes = boxes.loc[index_list]
  print(updated_boxes)

  avg_x = 0
  avg_y = 0
  for index, row in updated_boxes.iterrows():
    xcenter = row['xcenter']
    ycenter = row['ycenter']
    avg_x += xcenter
    avg_y += ycenter
  avg_x = int(avg_x / (len(updated_boxes)))
  avg_y = int(avg_y / (len(updated_boxes)))
  
  # Create a Rectangle patch and Illustrate Center
  if show_results:
    img_copy = np.array(img)
    print(img_copy.shape)
    
    for index, row in updated_boxes.iterrows():
        height = row["height"]
        width = row["width"]
        cv2.rectangle(img_copy,(row["xcenter"] - int(width / 2), row["ycenter"] - int(height / 2)) , (row["xcenter"] + int(width / 2), row["ycenter"] + int(height / 2)), (255,0,0),3)
    cv2.circle(img_copy,(avg_x, avg_y), 5, (0, 150, 255), -1)

    img_copy = img_copy[:,:,::-1]
    cv2.imshow("Frame", img_copy)
    cv2.waitKey(0)
  return (avg_x, avg_y)

if __name__ == "__main__":
    img = Image.open('Media\\Full_Court_Photo_Set\\court_dp_32.jpg')
    court_detector, player_detector = get_necessary_models()
    print(get_center_of_play(img, court_detector, player_detector, show_results=True))
