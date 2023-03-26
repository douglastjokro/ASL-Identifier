import pickle  
import cv2 
import numpy as np  
import mediapipe as mp  

# Load the trained model
model_dict = pickle.load(open('./trainedModel.p', 'rb'))
trainedModel = model_dict['trainedModel']

# Starting the webcam
cap = cv2.VideoCapture(0)

# Initializing the hand landmarks detection and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Continuously capturing the frames from the webcam, detecting the hand landmarks and interpreting the sign language
while True:

    # Reading a frame from the webcam
    ret, frame = cap.read()
    # Flipping the frame horizontally
    frame = cv2.flip(frame, 1)

    # Getting the height, width and channel count of the frame
    H, W, _ = frame.shape

    # Converting the frame from BGR color space to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting the hand landmarks in the frame
    results = hands.process(frame_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Draw the hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extracting the hand landmarks' positions and normalizing them
        landmarks = results.multi_hand_landmarks[0].landmark
        data = np.array([[landmark.x, landmark.y] for landmark in landmarks]).flatten()
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Predicting the sign language character using the trained model
        predicted_character = trainedModel.predict([data])[0]

        # Creating a white background for the text
        text_bg = np.zeros((100, W, 3), dtype=np.uint8)
        text_bg.fill(255)

        # Adding the predicted sign language character to the white background
        cv2.putText(text_bg, predicted_character, (W // 2 - 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 140, 255), 4, cv2.LINE_AA)

    # If no hand landmarks are detected
    else:
        # Creating an empty white background for the text
        text_bg = np.zeros((100, W, 3), dtype=np.uint8)
        text_bg.fill(255)

    # Concatenating the frame and the text background
    frame_with_text = np.concatenate((frame, text_bg), axis=0)

    # Displaying the frame with the detected hand landmarks and the predicted sign language character
    cv2.imshow('Sign Language Interpreter', frame_with_text)

    # Terminating the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the webcam and closing all windows
cap.release()
cv2.destroyAllWindows()
