# Basketball Play Tracking
![Basketball Play Tracking](Examples/Cover.png)

## Motivation 
I'm looking to create a tool to allow hands free recording of a general basketball game. I find, especially while recording games as a coach, that it would be convenient to just set my camera down and let the computer figure out where it should be looking. Throughout my time studying I have gained experience on everything I would need to accomplish this goal, so I figured I'd just go for it.

I got a lot of my inspiration from the study shown here (Simone Francia https://www.researchgate.net/publication/330534530_Classificazione_di_Azioni_Cestistiche_mediante_Tecniche_di_Deep_Learning). I used it as the blueprint and focused my effort as to how best to reach my personal goal of tracking the focus of the play.

## Contents
[General Overview](https://github.com/ben-morehead/BasketballPlayTracking/blob/readme/README.md#general-overview)

[Court Detecting Autoencoder](https://github.com/ben-morehead/BasketballPlayTracking/blob/readme/README.md#court-detecting-autoencoder)

[Center of Play Detection](https://github.com/ben-morehead/BasketballPlayTracking/blob/readme/README.md#center-of-play-detection)

[Live Tracking the Play](https://github.com/ben-morehead/BasketballPlayTracking/blob/readme/README.md#live-tracking-the-play)

[Next Steps](https://github.com/ben-morehead/BasketballPlayTracking/blob/readme/README.md#next-steps)

[Potential Improvements](https://github.com/ben-morehead/BasketballPlayTracking/blob/readme/README.md#potential-improvements)


## General Overview
The first step for the project was finding a way of representing the center of play so that the camera can focus in on that point of view. I figured if I could find where the majority of players were I could just average the position of them and that would provide a fairly accurate representation of where the play is being focused. By using a pre-trained YOLO model, I could easily detect where people in the frame were, but had to actually differentiate who was in the frame. Francia's paper came very handy here as there were already techniques in place of identifying a court in a given frame, so I could utilize the court as a reference to compare against the person detecting model. 

Putting those pieces together lead to a set of coordinates that represent where the camera should aim to be pointing. I wanted to simulate the camera movement before focusing on the electronics, so I made a script that would do just that. The next steps are listed below, but involve setting up my tripod with a motor rig, and getting it to track a sample object just to ensure the mechanics work before testing on an actual basketball game. My general approach at the moment is to get all the individual pieces that are needed working to some extent, and then before integration of all the parts flushing out and improving the performance of each of those pieces. I'm hoping to have it done by late 2022 or early 2023, but we'll see how it goes!

## Court Detecting Autoencoder
#### Concept
I want to be able to take a photo of a basketball court and convert it into a bitmap representation of which pixels are/aren't the court. An autoencoder allows for this with relatively high accuracy as its able to break down and recreate an input frame in the format desired. The autoencoder consists of a simple convolutional encoder in series with a convolutional decoder.

> Ideal Function of the Autoencoder
> ![Input and Label of Autoencoder dataset](Examples/InputLabel.png)

#### Structure
The autoencoder was adapted from Francia's paper referenced above. The kernel and stride lengths listed in the paper bring up some mathematical issues with the output dimensions of the neural network being different from the input dimensions, but with some alterations in the decoder layers I was able to get the dimensions to match up. After reworking the structure a little, I removed redundant layers to ensure optimal speed when processing datapoints.

#### Data Formatting
The neural network itself takes 720x480 tensor, which is fully formatted by the DataLoader() object that is defined in objects.py. The dataloader works in conjunction with the neural network by preparing all the dataset images for compatibility with the autoencoder object, as well as handling the batching and randomization of the dataset.

The output of the autoencoder is a 720x480 tensor which can be put through a sigmoid function and rounded to achieve a bitmap of whether a pixel is considered "the court" or not.

#### Training Strategy
I at this point run into an issue as I need to train the neural network despite not having a full dataset to work with. I started with about 100 images with corresponding court labels, and a full photo set of 1000 images to label and apply data augmentation techniques to. I wanted to make some more progress on the functionality of the project as a whole and not spend too much initial time labelling datapoints, so I decided to make the best I could out of the datapoints I had labeled. This meant tweaking batch size and epochs very mildy, but I figured it wasn't worth tuning too much without a proper and full data set. That being said I found the best results to be at 100 epochs and a batch size of 5. Shown below is a sample output at different checkpoints in the final training sequence.
> Sample Training Results
> ![Sample Outputs during Training](Examples/Autoencoder_Training.png)

#### Results
I was shockingly surprised by the accuracy that could be acquired off of such a small sample size. The model was able to detect most of the court, meaning it was functional enough to move on to the next challenge in this project. Although the model itself is not anywhere close to being ready for live video tracking on a random court with a random viewing angle, I have the infrastructure in place to easily improve on this part of the design a little later on.

> Final Function of the Autoencoder
> ![Input and Output of Autoencoder dataset](Examples/InputOutput.png)

## Center of Play Detection
#### Concept
Now that I have a way of differentiating who in a given frame is on the court and who is not, I just need to combine that with a person detection model to identify where players are congregated. For the person detection I did some quick research and found that Ultralytics had pre-trained YOLO models in the torch model hub, so I imported that and used the bounding box information.

#### YOLO Output
Shown below is an example of an output of the Ultralytics YOLOv5 model. Provided in the output is a pandas dataframe that contains the information on where the bounding boxes were in the frame, the class prediction, and a confidence score.

> Sample Output of Ultralytics YOLO Model
> ![Ultralytics YOLOv5 Sample](Examples/JustYolo.png)

#### Finding the Players
All that has to be done now is combine the two models together to figure out who is on the court and who isn't, and then average out the center positions of the people of interest. It was a very quick implementation, and can be easily tweaked by adjusting the ratio of matching pixels needed to be considered relevant. In the example below the yellow represents the court that does not interesect the toggle region, and the black section represents the toggle area. 

> Visualization of Intersecting Court and Player Regions
> ![Sample Bounding Box](Examples/ExampleOverlap.png)

If enough of the autoencoder "court pixels" interesect with the detection (say 90%), then we take into account that detections location in our calculation. Below is a filtered version of the yolo model output above, with the white dot marking the average player position.

> Final Filtered Image and Center of Play
> ![Center of Play](Examples/YoloAndEncoder.png)

## Live Tracking the Play
#### Parallelization
#### Camera Simulation
#### Results

## Next Steps
#### Camera Control
#### Camera Tracking
#### Improving the Design
#### Testing

## Potential Improvements
- [ ] **Autoencoder** *Data Set Size*: Need to label more datapoints and apply data augmentation techniques to better train the model
- [ ] **Autoencoder** *Hyperparameter Tuning*: With a full dataset can adjust the parameters of the autoencoder training to improve test accuracy
- [ ] **Live Tracking the Play** *Controller System*: Research control systems to aid in the camera tracing algorithm
- [ ] **Live Tracking the Play** *Introduction of Sectors*: Introduce set spots where the camera can turn to to decrease the amount of camera movement total
- [ ] **Live Tracking the Play** *Additional Processes*: Increase the number of datapoints that can be acquired from the center of play tracking algorithm

## Project Installation
