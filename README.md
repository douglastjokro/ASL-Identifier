# Sign Language Interpreter
This is a program that can identify sign language from a camera using OpenCV. It is split into four files of code that work together:

1. **dataInput.py** - This creates classes and records 100 snapshots from the camera for each class, which takes samples of each alphabet.
2. **dataSetBuild.py** - This takes the picture samples of the sign language alphabet output from dataInput.py, and formats them (crops the picture, focuses on the hands, etc.). This creates a file called signdata.pickle.
3. **trainer.py** - This trains the model using signdata.pickle, and outputs trainedModel.p.
4. **classifier.py** - This tests the model, where we take trainedModel.p, turn on the camera, and test if it works.

## Dependencies
- Python 3.x
- OpenCV
- Mediapipe
- Numpy
- Scikit-learn

## Usage
Run dataInput.py to capture the image samples. This will create a new directory named sign_language_dataset and populate it with images of the sign language alphabets.
Run dataSetBuild.py to process the captured images and create a new file named signdata.pickle.
Run trainer.py to train the model using the data in signdata.pickle. This will create a new file named trainedModel.p.
Run classifier.py to start the webcam and predict the sign language from your hand gestures.

## Credits
This program was developed by Douglas Tjokrosetio and is based on the Mediapipe library from Google.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
