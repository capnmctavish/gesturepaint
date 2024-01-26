import cv2
import mediapipe as mp
import numpy as np

class HandDraw:
    def __init__(self, cap):
        self.cap = cap
        self.pencil_size = 5
        self.draw_active = False
        self.key_pressed = False

        # Increase camera resolution
        cap.set(3, 1280)  # Width
        cap.set(4, 720)   # Height

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.canvas = np.ones((720, 1280, 3), np.uint8) * 255  # White canvas

    def toggle_pencil(self):
        self.draw_active = not self.draw_active

    def on_close(self):
        self.cap.release()

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Mirror the frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Use the first detected hand
            thumb_top = hand_landmarks.landmark[4]
            index_top = hand_landmarks.landmark[8]

            thumb_top_x, thumb_top_y = int(thumb_top.x * frame.shape[1]), int(thumb_top.y * frame.shape[0])
            index_top_x, index_top_y = int(index_top.x * frame.shape[1]), int(index_top.y * frame.shape[0])

            distance_threshold = 30  # Adjust as needed
            distance = np.sqrt((index_top_x - thumb_top_x)**2 + (index_top_y - thumb_top_y)**2)

            if distance < distance_threshold:
                self.draw_on_canvas(hand_landmarks)

        cv2.imshow("Hand Drawing", self.canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.on_close()
        elif key == ord('p') and not self.key_pressed:
            self.toggle_pencil()
            self.key_pressed = True
        elif key not in [ord('p'), 255]:
            self.key_pressed = False

    def draw_on_canvas(self, hand_landmarks):
        h, w, _ = self.canvas.shape

        cv2.circle(self.canvas, (x, y), self.pencil_size, (0, 0, 0), -1)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    app = HandDraw(cap)

    while True:
        app.update()
