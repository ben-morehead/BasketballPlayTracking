# Basketball Play Tracking

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
#### Structure
#### Data Formatting
#### Training Strategy
#### Results

## Center of Play Detection
#### Concept
#### YOLO Output
#### Finding the Players

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
