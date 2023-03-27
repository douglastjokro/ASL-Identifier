import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# initialize mediapipe for hand detection and drawing landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# configure mediapipe hands with options for static image mode and minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# set the data directory path
DATA_DIR = './sign_language_dataset'

# create empty lists to store data and labels
data = []
labels = []

# loop through each subdirectory in the data directory
for dir_ in os.listdir(DATA_DIR):

    # loop through each image file in the current subdirectory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        # create an empty list to store the normalized x and y coordinates of each hand landmark
        data_aux = []

        # create empty lists to store the x and y coordinates of each hand landmark
        x_ = []
        y_ = []

        # read the image file, convert it to RGB, and run hand detection using mediapipe
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # if any hand landmarks are detected in the image, loop through them
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                
                # loop through each landmark and add its x and y coordinates to the corresponding lists
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                    
                # loop through each landmark again and add its normalized x and y coordinates to data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # append the normalized landmark data and the corresponding label to the data and labels lists
            data.append(data_aux)
            labels.append(dir_)

# save the data and labels lists
f = open('signdata.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
