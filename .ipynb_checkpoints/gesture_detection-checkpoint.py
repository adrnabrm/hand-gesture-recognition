import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

def gesture_detector(model):
    gestures = ["peace","heart","shaka","none"]
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    BOX_MARGIN = 50
    while cap.isOpened():
        # capturing the current frame
        ret, frame = cap.read()
        # getting window dimensions, shape contains height, width, and channels
        height, width, _ = frame.shape

        # if no frame returned, break
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # for each hand found we calculate the coords of the landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                hand_coords = np.array(
                    [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_x_min, hand_x_max = min(x for x, y in hand_coords), max(x for x, y in hand_coords)
            hand_y_min, hand_y_max = min(y for x, y in hand_coords), max(y for x, y in hand_coords)

            rect_start, rect_end = (hand_x_min - BOX_MARGIN, hand_y_min - BOX_MARGIN), (
            hand_x_max + BOX_MARGIN, hand_y_max + BOX_MARGIN)
            rel_coords = np.array([(x - rect_start[0], y - rect_start[1]) for x, y in hand_coords])

            rel_x_max = max(x for x, y in rel_coords)
            rel_y_max = max(y for x, y in rel_coords)

            normalized_rel_coords = np.array([(x / rel_x_max, y / rel_y_max) for x, y in rel_coords])
            normalized_rel_coords = np.expand_dims(normalized_rel_coords, axis=0)

            predictions = model.predict(normalized_rel_coords, verbose=0)[0]
            prediction_p = tf.nn.softmax(predictions).numpy()
            max_prediction = np.argmax(prediction_p)

            if max_prediction != 3:
                text = f"{gestures[max_prediction]}, {(prediction_p[max_prediction] * 100):.2f}%"
            else:
                text = ""

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Hand Landmarks', frame)

        # when q pressed, we end video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    print('prog started')
    model = keras.models.load_model('gesture_model.keras')
    print('model loaded')
    gesture_detector(model)
    print('prog done')
main()