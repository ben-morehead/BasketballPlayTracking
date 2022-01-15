import matplotlib.pyplot as plt
import numpy as np
import cv2

from PlayerDetection.Autoencoder.objects import DataLoader, AutoEncoder

def convert_binary_to_rgb(frame):
    new_array = np.zeros((frame.shape[0], frame.shape[1], 3))
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[1]):
            if frame[i][j]:
                new_array[i][j] = np.array([255, 255, 255])
    return new_array.astype(float)

def run_autoencoder_demonstration():
    print("-- Data Specifications --")
    data_set = DataLoader()
    train, valid, test = data_set.all(batch_size=1)
    print("\t- Dataset Size: {}\n\t- Training Set Size: {}\n\t- Validation Set Size: {}\n\t- Test Set Size: {}".format(len(data_set.full_data_list), len(train), len(valid), len(test)))
    
    print("-- Input Format --")
    sample_input = test[0][0]
    print("\t- Batch Size: {}\n\t- Value Type: {}\n\t- Width: {}\n\t- Height: {}".format(sample_input.shape[0], "RGB", sample_input.shape[3], sample_input.shape[2]))
    
    print("-- Label Format --")
    sample_label = test[0][1]
    print("\t- Batch Size: {}\n\t- Value Type: {}\n\t- Width: {}\n\t- Height: {}".format(sample_label.shape[0], "BITMAP", sample_label.shape[2], sample_label.shape[1]))

    show_sample = input("-- Show Sample Input and Label? (Y/N): ")
    if show_sample.upper() == "Y":
        sample_input = np.swapaxes(sample_input[0], 0, 1)
        sample_input = np.swapaxes(sample_input, 1, 2)
        sample_input = sample_input[:,:,::-1]
        
        cv2.imshow("Sample Input", sample_input)
        cv2.imshow("Sample Label", sample_label[0])
        cv2.waitKey(0)
    print("-- Court Detecting AutoEncoder Architecture --")
    print("-- Training Specifications --")
    print("-- Output Example --")

def run_player_tracking_demonstration():
    pass

def run_live_tracking_demonstration():
    pass

if __name__ == "__main__":
    print("Running Full Demonstration")
    print("\n***** AutoEncoder *****")
    run_autoencoder_demonstration()