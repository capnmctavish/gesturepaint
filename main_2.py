import cv2
import mediapipe as mp
import numpy as np

class HandDraw:
    def __init__(self, cap):
        self.cap = cap
        self.pencil_size = 3
        self.draw_active = False
        self.prev_x, self.prev_y = -1, -1

        # Increase camera resolution
        cap.set(3, 1280)  # Width
        cap.set(4, 720)   # Height
        self.thumb_top_x, self.thumb_top_y = 0, 0
        self.index_top_x, self.index_top_y = 0, 0
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

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

        # Detect hand landmarks in the live video feed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Use the first detected hand

            thumb_top = hand_landmarks.landmark[4]
            index_top = hand_landmarks.landmark[8]

            self.thumb_top_x, self.thumb_top_y = int(thumb_top.x * frame.shape[1]), int(thumb_top.y * frame.shape[0])
            self.index_top_x, self.index_top_y = int(index_top.x * frame.shape[1]), int(index_top.y * frame.shape[0])

            distance_threshold = 40  # Adjust as needed
            distance = np.sqrt((self.index_top_x - self.thumb_top_x)**2 + (self.index_top_y - self.thumb_top_y)**2)

            if distance < distance_threshold:
                self.toggle_pencil()

        # Draw on the canvas window
        self.draw_on_canvas(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.on_close()

    def draw_on_canvas(self, frame):
        h, w, _ = self.canvas.shape

        if self.draw_active:
            # Calculate the midpoint of thumb and index finger top landmarks
            midpoint_x = (self.thumb_top_x + self.index_top_x) // 2
            midpoint_y = (self.thumb_top_y + self.index_top_y) // 2

            # Draw a circle at the calculated midpoint
            cv2.circle(self.canvas, (midpoint_x, midpoint_y), self.pencil_size, (0, 0, 0), -1)

        # Display the canvas window
        cv2.imshow("Hand Drawing", self.canvas)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    app = HandDraw(cap)

    while True:
        app.update()
