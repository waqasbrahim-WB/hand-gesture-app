```python
import os
import cv2
import mediapipe as mp
import streamlit as st

from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================
# Streamlit Page Settings
# =========================

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

st.title("✋ AI Hand Gesture Recognition")
st.write("Show your hand to the camera and the app will count your fingers.")


# =========================
# Groq API
# =========================

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    client = Groq(api_key=groq_api_key)
else:
    client = None


# =========================
# MediaPipe Hands
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, hand_label):

    landmarks = hand_landmarks.landmark

    fingers = 0

    # -------------------------
    # Thumb
    # -------------------------

    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers += 1

    # -------------------------
    # Index, Middle, Ring, Pinky
    # -------------------------

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):

        if landmarks[tip].y < landmarks[pip].y:
            fingers += 1

    return fingers


# =========================
# Video Processor
# =========================

class HandGestureProcessor(VideoProcessorBase):

    def __init__(self):

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.finger_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # BGR → RGB
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        self.finger_count = 0

        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand_label = handedness.classification[0].label

                self.finger_count = count_fingers(
                    hand_landmarks,
                    hand_label
                )

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # =========================
        # Display Finger Count
        # =========================

        cv2.putText(
            img,
            f"Fingers: {self.finger_count}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================
# App Layout
# =========================

left, right = st.columns([2, 1])


# =========================
# LEFT SIDE - CAMERA
# =========================

with left:

    st.subheader("📷 Camera")

    ctx = webrtc_streamer(
        key="hand-gesture",

        video_processor_factory=HandGestureProcessor,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


# =========================
# RIGHT SIDE - RESULT
# =========================

with right:

    st.subheader("🤖 Detection Result")

    if ctx.video_processor:

        count = ctx.video_processor.finger_count

        st.metric(
            "Detected Fingers",
            count
        )

        # =========================
        # Gesture Message
        # =========================

        if count == 0:
            message = "✊ No fingers"

        elif count == 1:
            message = "☝️ One Finger"

        elif count == 2:
            message = "✌️ Two Fingers"

        elif count == 3:
            message = "Three Fingers"

        elif count == 4:
            message = "Four Fingers"

        elif count == 5:
            message = "✋ Five Fingers"

        else:
            message = "Unknown Gesture"

        st.success(message)

        # =========================
        # Groq AI Response
        # =========================

        if client:

            try:

                prompt = f"""
                The user is showing {count} finger(s)
                to a hand gesture recognition application.

                Give a very short and friendly response
                about the detected number.

                Keep the response under 15 words.
                """

                response = client.chat.completions.create(

                    model="llama-3.1-8b-instant",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.5,
                    max_tokens=50
                )

                ai_response = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.info(
                    f"🤖 AI: {ai_response}"
                )

            except Exception:

                st.warning(
                    "Unable to get a response from Groq."
                )

        else:

            st.caption(
                "GROQ_API_KEY is not configured. "
                "Finger detection still works."
            )

    else:

        st.info(
            "Click START and allow camera access."
        )


# =========================
# Instructions
# =========================

st.divider()

st.subheader("📖 How to Use")

st.markdown("""
1. Click **START**.
2. Allow camera permission.
3. Show your hand in front of the camera.
4. The app will automatically count your fingers.
5. **1 finger → One Finger**
6. **2 fingers → Two Fingers**
7. **3 fingers → Three Fingers**
8. **4 fingers → Four Fingers**
9. **5 fingers → Five Fingers**
""")
```
