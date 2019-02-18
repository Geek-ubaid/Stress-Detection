# Stress Detection
## Description:
This is a real time face stress detection model.The model is image processing based model which is having two parts
- Emotion Recognition
- Stress level calculation  

The emotion recognition model will return the emotion predicted real time. The model classifies face as stressed and not stressed.
A model is trained on the fer2013 dataset.  

The stress level is calculated with the help of eyebrows contraction and displacemnent from the mean position. The distance between the 
left and right eyebrow is being calculated and then the stress level is calculated using exponential function and normalized between 1 to 100.  

*** Important****
Before running the eyebrow_detection.py first download file in the same folder using this [link](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)


## Procedure:
1. Real time web cam feed
2. Detect faces
3. Detect eyebrows both left and right
4. Predict stress/not stress
5. Calculate the stress level  

## Accuracy/Usage:
- Run the eyebrow_detection.py file and sit straight infront of the webcam. 
- Do not run the emotion_recognitio.py. Use only if you want to retrain the model.
- Try to be clear with your emotion, Fakeness cannot be detected.
- The model is moderately accurate because the data could not be arranged within stipulated time.  

## Improvement Strategy:
The model can be improved by including other facial features inputs as well. The feature include:
- Lip movement
- Head positioning
- Eye blinking
- Gaze movement  
The following features can be detected and a cummulative function can be defined to give out the total stress value.



 
