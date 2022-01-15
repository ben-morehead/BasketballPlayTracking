import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random

from PlayerDetection.frame import get_necessary_models, get_center_of_play
from PlayerDetection.video import analyze_video
from PlayerDetection.Autoencoder.objects import DataLoader, AutoEncoder
from PIL import Image

THRESHOLD_STEP = 6

def convert_binary_to_rgb(frame):
    new_array = np.zeros((frame.shape[0], frame.shape[1], 3))
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[1]):
            if frame[i][j]:
                new_array[i][j] = np.array([255, 255, 255])
    return new_array.astype(float)

def run_autoencoder_demonstration():
    print("-- Data Specifications --")
    auto = AutoEncoder()
    auto.load_state_dict(torch.load('PlayerDetection\\Autoencoder\\Models\\model_weights_100samples.pth'))
    data_set = DataLoader()
    train, valid, test = data_set.all(batch_size=1)
    print("\t- Dataset Size: {}\n\t- Training Set Size: {}\n\t- Validation Set Size: {}\n\t- Test Set Size: {}".format(len(data_set.full_data_list), len(train), len(valid), len(test)))
    
    print("-- Input Format --")
    sample_input = test[0][0]
    sample_input_og = sample_input
    print("\t- Batch Size: {}\n\t- Value Type: {}\n\t- Width: {}\n\t- Height: {}".format(sample_input.shape[0], "RGB", sample_input.shape[3], sample_input.shape[2]))
    sample_input = np.swapaxes(sample_input[0], 0, 1)
    sample_input = np.swapaxes(sample_input, 1, 2)
    sample_input = sample_input[:,:,::-1]

    print("-- Label Format --")
    sample_label = test[0][1]
    print("\t- Batch Size: {}\n\t- Value Type: {}\n\t- Width: {}\n\t- Height: {}".format(sample_label.shape[0], "BITMAP", sample_label.shape[2], sample_label.shape[1]))

    show_sample = input("-- Show Sample Input and Label? (Y/N): ")
    if show_sample.upper() == "Y":
        cv2.imshow("Sample Input", sample_input)
        cv2.imshow("Sample Label", sample_label[0])
        cv2.waitKey(0)

    print("-- Court Detecting AutoEncoder Architecture --")
    print("\t- Conv Layer Count: {}\n\t- DeConv Layer Count: {}\n\t- Kernel Count Encoder: {}\n\t- Kernel Count Decoder{}".format(4,7,[64,128,256,512],[256,256,128,128,64,64,1]))
    print("\t- Encoder Kernel Sizes: {}\n\t Decoder Kernel Sizes: {}\n\t- Encoder Strides: {}\n\t- Decoder Strides: {}".format([3, 3, 3, 3],[3, 3, 1, 3, 1, 3, 2],[2, 2, 2, 2],[2, 2, 1, 2, 1, 2, 1]))
    print("\t- Activation Function: ReLU")
    
    print("-- Training Specifications --")
    print("\t- Epochs: {}\n\t- Learning Rate: {}\n\t- Batch Size: {}\n\t- Training Accuracy: {}".format(100, 1e-4, 5, 0.936))
    
    print("-- Output Example --")
    
    sample_output = auto(torch.from_numpy(sample_input_og).float())
    sig_out = torch.sigmoid(sample_output)
    sig_output = (sig_out > 0.5).float()
    output_img = np.array(sig_output)
    output_img = np.array(output_img[0])
    sample_output = output_img.squeeze(0)
    #sample_output = cv2.resize(output_img, (1920, 1080))
    show_sample = input("-- Show Sample Input and Output? (Y/N): ")
    if show_sample.upper() == "Y":
        cv2.imshow("Sample Input", sample_input)
        cv2.imshow("Sample Output", sample_output)
        cv2.waitKey(0)

def run_play_tracking_demonstration():
    
    img = Image.open('Media\\Full_Court_Photo_Set\\court_dp_{}.jpg'.format(random.randint(0, 1000)))
    court_detector, player_detector = get_necessary_models()
    x, y = get_center_of_play(img, court_detector, player_detector, show_results=True, demo_usage = True)
    print("-- Center of Play --")
    print("\t- Center of Play X Coordinate: {}\n\t- Center of Play Y Coordinate: {}".format(x, y))

def run_live_tracking_demonstration():
    print("-- Video Live Tracking --")
    print("**This is executed as if a frame was being brough in once every 30th of a second, so the output is tracking in live time. The blue represents the system's internal")
    video_path = "Media\\Video_Samples\\41s_Sample.mp4"
    analyze_video(video_path)

if __name__ == "__main__":
    print("Running Full Demonstration")
    #print("\n***** AutoEncoder *****")
    #run_autoencoder_demonstration()
    #print("\n***** Play Tracking *****")
    #run_play_tracking_demonstration()
    print("\n***** Live Feed Tracking and Camera Simulation*****")
    run_live_tracking_demonstration()