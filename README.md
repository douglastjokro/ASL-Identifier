# Sign Language Interpreter
This is a program that can identify sign language from a camera using OpenCV. It is split into four files of code that work together:

1. **dataInput.py** - This program captures image samples of sign language alphabets (6 in this case) using the user's camera and saves them as image files.
2. **dataSetBuild.py** - This program reads in the image files captured by **dataInput.py** and processes them to extract the hand landmarks. It then normalizes and saves the hand landmarks and their corresponding labels using pickle.
3. **trainer.py** - This program reads in the normalized hand landmarks and their corresponding labels from **signdata.pickle**. It trains a random forest classifier model using the data and saves the trained model as a pickled file.
4. **classifier.py** - This program loads the trained model and continuously captures frames from the user's camera to predict the sign language alphabet represented by the hand gesture. The predicted alphabet is displayed as text on a white background overlaid on the video feed.

## Dependencies
- Python 3.x
- OpenCV
- Mediapipe
- Numpy
- Scikit-learn

## Usage
Run **dataInput.py** to capture the image samples. This will create a new directory named sign_language_dataset and populate it with images of the sign language alphabets.
Run **dataSetBuild.py** to process the captured images and create a new file named signdata.pickle.
Run **trainer.py** to train the model using the data in signdata.pickle. This will create a new file named trainedModel.p.
Run **classifier.py** to start the webcam and predict the sign language from your hand gestures.

## Credits
This program was developed by Douglas Tjokrosetio and is based on the Mediapipe library from Google.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
