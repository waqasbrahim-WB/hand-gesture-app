```python
import os
import cv2
import mediapipe as mp
import streamlit as st
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .gesture-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    .number {
        font-size: 70px;
        font-weight: bold;
    }

    .gesture {
        font-size: 25px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">✋ AI Hand Gesture Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Show your hand to the camera and let AI count your fingers.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GROQ CLIENT
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# FINGER COUNT FUNCTION
# ============================================================

def count_fingers(hand_landmarks, hand_label):

    landmarks = hand_landmarks.landmark

    count = 0

    # --------------------------------------------------------
    # Thumb
    # --------------------------------------------------------

    if hand_label == "Right":

        if landmarks[4].x < landmarks[3].x:
            count += 1

    else:

        if landmarks[4].x > landmarks[3].x:
            count += 1

    # --------------------------------------------------------
    # Index Finger
    # --------------------------------------------------------

    if landmarks[8].y < landmarks[6].y:
        count += 1

    # --------------------------------------------------------
    # Middle Finger
    # --------------------------------------------------------

    if landmarks[12].y < landmarks[10].y:
        count += 1

    # --------------------------------------------------------
    # Ring Finger
    # --------------------------------------------------------

    if landmarks[16].y < landmarks[14].y:
        count += 1

    # --------------------------------------------------------
    # Pinky Finger
    # --------------------------------------------------------

    if landmarks[20].y < landmarks[18].y:
        count += 1

    return count


# ============================================================
# GESTURE INFORMATION
# ============================================================

def get_gesture(count):

    gestures = {
        0: ("✊", "No Fingers"),
        1: ("☝️", "One Finger"),
        2: ("✌️", "Two Fingers"),
        3: ("🤟", "Three Fingers"),
        4: ("🖖", "Four Fingers"),
        5: ("✋", "Five Fingers")
    }

    return gestures.get(
        count,
        ("❓", "Unknown Gesture")
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class HandGestureProcessor(VideoProcessorBase):

    def __init__(self):

        self.finger_count = 0

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def recv(self, frame):

        # Convert video frame to OpenCV format
        image = frame.to_ndarray(format="bgr24")

        # OpenCV BGR -> RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Process hand
        results = self.hands.process(rgb_image)

        # Default
        self.finger_count = 0

        # ----------------------------------------------------
        # If hand detected
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand_label = (
                    handedness.classification[0].label
                )

                # Count fingers
                self.finger_count = count_fingers(
                    hand_landmarks,
                    hand_label
                )

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # ----------------------------------------------------
        # Display count on camera
        # ----------------------------------------------------

        cv2.putText(
            image,
            f"Fingers: {self.finger_count}",
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3
        )

        # Return processed frame
        return frame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# MAIN LAYOUT
# ============================================================

camera_column, result_column = st.columns(
    [2, 1],
    gap="large"
)


# ============================================================
# CAMERA
# ============================================================

with camera_column:

    st.subheader("📷 Live Camera")

    ctx = webrtc_streamer(
        key="hand-gesture-camera",

        video_processor_factory=HandGestureProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    st.caption(
        "Click START and allow camera permission."
    )


# ============================================================
# RESULT PANEL
# ============================================================

with result_column:

    st.subheader("🤖 Detection")

    if ctx.video_processor:

        count = ctx.video_processor.finger_count

        emoji, gesture_name = get_gesture(count)

        # ----------------------------------------------------
        # Number
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="gesture-box">

                <div class="number">
                    {count}
                </div>

                <div class="gesture">
                    {emoji} {gesture_name}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # ----------------------------------------------------
        # Groq AI
        # ----------------------------------------------------

        if groq_client:

            try:

                prompt = f"""
                A hand gesture recognition application detected
                {count} fingers.

                Gesture:
                {gesture_name}

                Give a short, friendly response to the user.

                Keep it under 12 words.
                """

                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.5,

                    max_tokens=40
                )

                ai_message = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.info(
                    f"🤖 AI: {ai_message}"
                )

            except Exception:

                st.warning(
                    "Groq AI response is currently unavailable."
                )

        else:

            st.warning(
                "Groq API key is not configured."
            )

    else:

        st.info(
            "Start the camera to detect your hand gesture."
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("📖 How It Works")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        ### 1️⃣ Camera

        Your webcam captures the hand gesture.
        """
    )

with col2:

    st.markdown(
        """
        ### 2️⃣ MediaPipe

        MediaPipe detects the hand and counts the fingers.
        """
    )

with col3:

    st.markdown(
        """
        ### 3️⃣ Groq AI

        Groq generates a short AI response based on the gesture.
        """
    )


# ============================================================
# GESTURES
# ============================================================

st.subheader("✋ Supported Gestures")

st.markdown(
    """
    | Fingers | Gesture |
    |---:|---|
    | 0 | ✊ No Fingers |
    | 1 | ☝️ One Finger |
    | 2 | ✌️ Two Fingers |
    | 3 | 🤟 Three Fingers |
    | 4 | 🖖 Four Fingers |
    | 5 | ✋ Five Fingers |
    """
)
```
