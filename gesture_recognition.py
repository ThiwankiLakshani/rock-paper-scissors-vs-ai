import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Choices for Rock Paper Scissors
CHOICES = ["Rock", "Paper", "Scissors"]

def detect_gesture(landmarks):
    # Extract key landmarks (tip and base of each finger)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]  # Base of thumb
    index_tip = landmarks[8]
    index_mcp = landmarks[5]  # Base of index finger
    middle_tip = landmarks[12]
    middle_mcp = landmarks[9]  # Base of middle finger
    ring_tip = landmarks[16]
    ring_mcp = landmarks[13]  # Base of ring finger
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]  # Base of pinky

    # Helper function to check if a finger is extended
    def is_finger_extended(tip, mcp, threshold=0.05):
        return tip.y < mcp.y - threshold  # Lower y means higher in image (extended)

    # Check if each finger is extended
    thumb_extended = is_finger_extended(thumb_tip, thumb_mcp)
    index_extended = is_finger_extended(index_tip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_mcp)

    # Debugging: Print finger states
    print(f"Thumb: {'Extended' if thumb_extended else 'Folded'}, "
          f"Index: {'Extended' if index_extended else 'Folded'}, "
          f"Middle: {'Extended' if middle_extended else 'Folded'}, "
          f"Ring: {'Extended' if ring_extended else 'Folded'}, "
          f"Pinky: {'Extended' if pinky_extended else 'Folded'}")

    # Gesture detection logic
    if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Rock"  # All fingers folded
    elif index_extended and middle_extended and ring_extended and pinky_extended:
        return "Paper"  # All fingers extended
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Scissors"  # Index and middle extended, others folded
    return None

def determine_winner(player_choice, ai_choice):
    if player_choice == ai_choice:
        return "Tie"
    winning_combinations = {
        "Rock": "Scissors",
        "Paper": "Rock",
        "Scissors": "Paper"
    }
    if winning_combinations[player_choice] == ai_choice:
        return "Win"
    return "Lose"

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Game state variables
game_active = False
countdown = 0
countdown_start = 0
player_choice = None
ai_choice = None
result = None
last_round_time = 0
next_round_time = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip and convert the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(frame_rgb)

    # Handle countdown and gesture detection
    if game_active and countdown > 0:
        elapsed = time.time() - countdown_start
        countdown_display = 3 - int(elapsed)
        if countdown_display <= 0:
            countdown = 0
            # Capture gesture at end of countdown
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = detect_gesture(hand_landmarks.landmark)
                    if gesture:
                        player_choice = gesture
                        ai_choice = random.choice(CHOICES)
                        result = determine_winner(player_choice, ai_choice)
                        last_round_time = time.time()
                        next_round_time = time.time() + 10  # Schedule next round
            else:
                player_choice = None
                ai_choice = None
                result = "No hand detected"
                last_round_time = time.time()
                next_round_time = time.time() + 10

    # Start new round every 10 seconds if game is active
    if game_active and countdown == 0 and time.time() >= next_round_time:
        countdown = 3
        countdown_start = time.time()
        player_choice = None
        ai_choice = None
        result = None

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display countdown
    if countdown > 0:
        cv2.putText(frame, f"Countdown: {3 - int(time.time() - countdown_start)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display player choice, AI choice, and result
    if player_choice:
        cv2.putText(frame, f"Player: {player_choice}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if ai_choice:
        cv2.putText(frame, f"AI: {ai_choice}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if result:
        cv2.putText(frame, f"Result: {result}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Clear result after 3 seconds
    if result and time.time() - last_round_time > 3:
        player_choice = None
        ai_choice = None
        result = None

    # Display instructions
    cv2.putText(frame, f"Press 's' to {'stop' if game_active else 'start'} game, 'q' to quit",
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Rock Paper Scissors", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        game_active = not game_active  # Toggle game state
        if game_active:
            countdown = 3
            countdown_start = time.time()
            next_round_time = time.time() + 10
            player_choice = None
            ai_choice = None
            result = None
        else:
            countdown = 0  # Stop countdown when game is off

# Cleanup
cap.release()
cv2.destroyAllWindows()