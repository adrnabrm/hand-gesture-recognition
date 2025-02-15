import os.path
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from tensorflow.nn import softmax


BOX_MARGIN = 50
COLLECTING = False
NUM_TRAINING_EX = 0
MESSAGE = "Click to collect!"

def data_collection(event,x,y,flags,param):
    """
    Handles mouse event on live feed.

    When the user clicks with the left mouse button, data collection is toggled. The message displayed tells the user whether or not they are collecting data.

    Args:
        event: type of mouse event
        x: x coordinate of mouse event
        y: y coordinate of mouse event
        flags: specific condition or state associated with the mouse event
        param: additional parameters that can be passed
    """
    global COLLECTING
    global MESSAGE
    global NUM_TRAINING_EX
    if event == cv2.EVENT_LBUTTONDOWN:
        COLLECTING = not COLLECTING
        if COLLECTING:
            MESSAGE = f"Collecting Data!"
        else:
            MESSAGE = "Click to collect!"


def video_capture(model, gestures):
    """
    Turns on system's default camera and takes data of hand.

    Using the OpenCV library, the default camera captures frames. The MediaPipe framework is then used to detect hand landmarks in the video and the nodes detected are shown on the live feed. The relative coordinates and distances between certain landmarks are calculated to be written to the training data.

    Returns:
        data: list of normalized relative landmark coords for training
    """
    global MESSAGE
    global NUM_TRAINING_EX

    data = []
    # initializing MediaPipe's hand detection module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    # default camera is used to capture live feed
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Hand Landmarks')

    if model:
        MESSAGE = "Test the gesture model!"
    else:
        cv2.setMouseCallback('Hand Landmarks', data_collection)
        cv2.putText(frame, f"Count: {NUM_TRAINING_EX}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print("Starting video capture.")
    while cap.isOpened():
        # capturing the current frame
        ret, frame = cap.read()
        # getting window dimensions, shape contains height, width, and channels
        height, width, _ = frame.shape

        # if no frame returned, break
        if not ret:
            break

        # flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # adding message onto display to tell user if they are collecting data or not
        cv2.putText(frame, MESSAGE, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # convert the BGR image to RGB for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the frame and find hands
        # process() returns
            # multi_handedness: if left or right hand
            # multi_hand_landmarks: normalized xyz coords of each landmark
            # multi_hand_world_landmarks: real world xyz coords in meters
        results = hands.process(frame_rgb)

        # drawing hand landmarks
        if results.multi_hand_landmarks:
            # for each hand found we calculate the coords of the landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                hand_coords = [ (int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark ]

                # getting min and max of x,y coords
                hand_x_min, hand_x_max = min(x for x,y in hand_coords), max(x for x,y in hand_coords)
                hand_y_min, hand_y_max = min(y for x,y in hand_coords), max(y for x,y in hand_coords)
                # drawing a box around the hand
                rect_start, rect_end = (hand_x_min - BOX_MARGIN, hand_y_min - BOX_MARGIN), (hand_x_max + BOX_MARGIN, hand_y_max + BOX_MARGIN)
                cv2.rectangle(frame, rect_start, rect_end, (255, 0, 0), 2)
                # using MediaPipe library to draw hand landmarks on the display
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # calculate the coords relative to the rectangle
                rel_coords = [ (x - rect_start[0], y - rect_start[1]) for x,y in hand_coords ]

                # just a quick check to see that no matter where the hand moves, the coords stay relative :o
                # for coord in rel_coords:
                #     cv2.circle(frame, coord, 10, (0, 0, 255), -1)

                # calculating the x and y relative coord maxes
                rel_x_max = max(x for x, y in rel_coords)
                rel_y_max = max(y for x, y in rel_coords)
                # normalizing the relative coordinates for training set
                normalized_rel_coords = [(x / rel_x_max, y / rel_y_max) for x, y in rel_coords]


                if COLLECTING:
                    # adding normalized relative coords to the data
                    data.append(normalized_rel_coords)
                    NUM_TRAINING_EX += 1
                elif model:
                    normalized_rel_coords = np.expand_dims(normalized_rel_coords, axis=0)
                    predictions = model.predict(normalized_rel_coords, verbose=0)[0]
                    predictions_p = softmax(predictions).numpy()
                    max_prediction = np.argmax(predictions_p)
                    target_probability = predictions_p[max_prediction]

                    if gestures[max_prediction] != 'none' and target_probability > .85:
                        text = f"{gestures[max_prediction]}, {(target_probability * 100):.2f}%"
                    else:
                        text = ""
                    cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Hand Landmarks', frame)


        # when q pressed, we end video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Video capture ended.")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    return data

def write_to_csv(training_data):
    """
    Writes the training data to the csv.

    User is prompted to choose a name for the csv. Takes all the calculated normalized coordinates from video capture and writes to previously specified csv file. These csv files are stored in a directory called data.

    Args:
        training_data: data collected from the video capture.
    """
    directory = 'data'
    file_name = ''
    perms = ''
    if not os.path.exists(directory):
        os.makedirs(directory)

    while len(file_name) == 0:
        file_name = str(input('Enter a name for the csv: '))
        if len(file_name) > 0:
            break

    file_path = os.path.join(directory, file_name+'.csv')
    print(file_path)
    if os.path.exists(file_path):
        perms = 'a'
    else:
        perms = 'w'

    print("Writing data to csv file.")

    file = open(file_path, perms)
    if perms == 'w':
        file.write('coords\n')

    for row in training_data:
        file.write(f'"{row}"\n')
    file.close()

    print("Write successful!")

def retrieve_gestures(directory):
    files = os.listdir(directory)
    categories = [file.split('.')[0] for file in files]
    for i in range(len(categories)):
        categories[i] = " ".join(categories[i].split('_'))

    return sorted(categories)


def main():
    """
    Main function.

    Serves as starting point of the program.
    """
    print("Starting program...")
    testing_model = int(input("Would you like to collect data or test the model? (0-collect |1-test): "))
    if testing_model != 0 and testing_model != 1:
        raise ValueError("Please enter 0 or 1.")
    model = None
    gestures = None
    if testing_model:
        model = keras.models.load_model('gesture_model.keras')
        gestures = retrieve_gestures('./data')
    # video capture begins
    training_data = video_capture(model, gestures)

    # writing to csv
    if training_data:
        write_to_csv(training_data)

main()