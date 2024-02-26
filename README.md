# Graduation-Project-code
Sign Language Interpreter
# Arabic Sign Language Interpreter Project

## Overview
This project, developed by the Electrical Engineering Department students at Kafrelsheikh University, aims to bridge the communication gap for the deaf and mute community by translating Arabic Sign Language into spoken Arabic. It's a part of the Egypt IOT and AI Challenge 2020.

## Team Members
- Mahmoud Aboud Nada
- Hesham Abdullah ElKhouly
- MennatuAllah Ahmed El-Zahaby
- Muhammad Ali Shokair
- Omar Ibraheem Shaheen
- Asmaa Hasan Dardarah

## Supervisor
- Professor/ Wessam Fekrey

## Introduction
The Arabic Sign Language (ARSL) Interpreter is designed to understand and translate sign language into spoken words, using a deep learning model trained on a comprehensive dataset of Arabic sign language gestures.

## Objectives
- To develop an efficient and accurate system for interpreting Arabic sign language.
- To create a user-friendly interface for real-time communication.
- To contribute to the inclusivity of the deaf and mute community in the digital age.

## Methodology
The project follows a structured approach, starting from data collection, preprocessing, model development, and testing, to deployment. The methodology section details each step involved in creating the ARSL interpreter.

### Data Collection
First, we had to gather our own customed dataset that consists of a set of words that we believe are most common in daily life according to sign language instructors.
- Figure 1: Data Sample ![Data Sample](https://github.com/Hesham-Abdullah/Graduation-Project-code/blob/main/assets/1.png)

we started with 28 words from 4 different signers each word is repeated 10 times so this makes our initial dataset with 1120 videos.
As we were collecting the dataset from local sign language specialists, we started to prepare the code that deals with any video dataset, we used an experimental dataset called UCF101 that is used in action recognition it has 101 classes within each class videos belongs to this class.


### Model Development
First, we started working on the video to text conversion without going into the detection techniques and methods.
4
With the help of UCF dataset, we managed to build the code that preprocess the videos (organize them and extract the frames from each video using ffmpeg tool), we also built an initial small CNN + LSTM model that we used in training.

The Previous Model Did not Perform significantly better that than the first one so we used a similar architecture introduced in an Action Recognition Paper called Very Deep Convolutional Networks for Large-Scale Image Recognition, and used a Bidirectional LSTM at the end as advised by the Deep ASL Paper.

## Implementation
Deploying a Deep learning model is not an easy task, especially in our case where we have multiple models running simultaneously in addition to the fact that these models need to be running in real time as well as being portable e.g. (Phone).

After many researches and experiments we chose to go with this deployment method, and the following steps are to be made: Create a website using flask that can talk directly to the model, it sends a video feed to the model and receives the interpretations back.
• Look up a server that is affordable to us in the meantime, to deploy our system on.
• Last step is making an API that takes the video feed from the Flask website and gives us back the interpretations.

## Results
For our prototype, we built a simple web page using HTML and JavaScript. The web page receives a video then processes it extracting the frames and saving them into a list, after that it calls our Prediction model passing in the preprocessed frames to predict the corresponding word (label) for the given video from our labels list and return the Arabic word as a message through our page with the corresponding voice for the predicted word.

## Future Work
Our current results aren’t perfect and we have many future plans for our project and device to be as helpful as possible to everyone in the deaf and mute community. For now, we are working towards two main aspects:
1- Build a more consistent model that receives real time signs which it can detect the key points of the signer’s hands and face, extract those key points and save them as a dataset of landmarks. The model then trains on those key points landmarks and pass straight to the prediction algorithm where it can predict the correct corresponding label to each word and generate the correct voice for that word in Arabic. It should be a fast working model that’s able to process in real time.

2- Deploy the new model on the hardware device using Raspberry Pi as our main controller with the needed gadgets, a camera, microphone, speaker, etc. the users will use this portable device whenever they need to talk to anyone that is not familiar with Arabic sign language easily and reliably.

## Figures
- Figure 2: The final project flow graph[The final project flow graph](https://github.com/Hesham-Abdullah/Graduation-Project-code/blob/main/assets/2.png)

