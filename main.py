import os
import random

import cv2

# Load the Haar cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(
    '/home/babylvoob/Desktop/project/Face-reconize/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Load the trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# Create a function to load a trained face recognition model
def load_face_recognizer(model_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    return face_recognizer


# Create a function to generate a random label for a new face
def generate_random_label():
    return random.randint(1, 100000)


# Create a function to create a new model for a face
def create_new_model(face, label, model_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train([face], [label])
    face_recognizer.save(model_path)


# Create a dictionary mapping labels to names
def label_to_name(label):
    names = {
        1: 'Alice',
        2: 'Bob',
        3: 'Charlie',
        4: 'Dave',
        5: 'Eve',
    }
    return names.get(label, 'Unknown')


# Open the video file
video = cv2.VideoCapture('input.mp4')

# Create a list to store the models
models = []

# Loop over all the files in the models directory
path = "./models"
# Check whether the specified path exists or not
is_exist = os.path.exists(path)

if not is_exist:
    os.makedirs(path)

for file in os.listdir('./models'):
    # Load the model if it is a YAML file
    if file.endswith('.yml'):
        models.append(load_face_recognizer(os.path.join('./models', file)))

data_set_path = './raw_data'
is_exist = os.path.exists(data_set_path)
if not is_exist:
    os.makedirs(data_set_path)


def learning_data(data_set_path):
    if os.path.isdir(data_set_path):
        # Loop over all the files in the input directory
        for file in os.listdir(data_set_path):
            print(file, type(file))
            # Load the image if it is a JPG or PNG file
            if file.endswith('.jpg') or file.endswith('.png'):
                image = cv2.imread(os.path.join(data_set_path, file))
                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                # Loop over the detected faces
                for (x, y, w, h) in faces:
                    # Extract the face region and resize it to 150x150
                    face = cv2.resize(gray[y:y + h, x:x + w], (150, 150))
                    # Loop over the models and try to recognize the face
                    recognized = False
                    for model in models:
                        label, confidence = model.predict(face)
                        if confidence < 60:
                            name = label_to_name(label)
                            print(f'Recognized {name} with confidence {confidence}')
            elif file.endswith('.mp4'):
                while True:
                    # Read the next frame from the video
                    _, frame = video.read()

                    # Check if the frame was read successfully
                    if not _:
                        break

                    # Convert the frame to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces in the frame
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    # Loop over the detected faces
                    for (x, y, w, h) in faces:
                        # Extract the face region and resize it to the training size
                        face = cv2.resize(gray[y:y + h, x:x + w], (150, 150))

                        # Predict the label of the face
                        label, confidence = face_recognizer.predict(face)

                        # Draw a rectangle around the face and put the name of the person below it
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, label_to_name(label), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)

                    # Show the frame
                    cv2.imshow('Frame', frame)

                    # Check if the user pressed the 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                learning_data(data_set_path + '/' + file)

# Run recurring function
learning_data(data_set_path)

# Release the video capture and destroy the window
video.release()
cv2.destroyAllWindows()
