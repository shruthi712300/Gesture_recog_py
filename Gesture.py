import streamlit as st
import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import speech_recognition as sr
import requests


# ---------------------- Voice Synthesis ---------------------- #
gesture_voice_settings = {
    "Alert": {"rate": 130, "voice": 0},
    "Call me": {"rate": 170, "voice": 1},
    "Thank you": {"rate": 150, "voice": 2},
}

def speak(text, gesture=None):
    def _speak():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if gesture in gesture_voice_settings:
            settings = gesture_voice_settings[gesture]
            engine.setProperty('rate', settings.get("rate", 150))
            voice_id = settings.get("voice", 0)
            if voice_id < len(voices):
                engine.setProperty('voice', voices[voice_id].id)
        else:
            engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

# ---------------------- Gesture Recognition ---------------------- #
FINGER_COMBO_LABELS = {
    (0, 0, 0, 0, 0): "Alert",
    (0, 0, 0, 0, 1): "Call me",
    (1, 1, 1, 1, 1): "Thank you",
}

def fingers_up(landmarks, hand_label):
    fingers = []
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return tuple(fingers)

# ---------------------- Simple Response ---------------------- #
def gesture_response(input_text):
    if "thank" in input_text.lower():
        return "You're welcome!"
    elif "call" in input_text.lower():
        return "Calling you now... 📞"
    elif "alert" in input_text.lower():
        return "Emergency alert triggered! 🚨"
    else:
        return "Gesture or voice received."

# ---------------------- UI Setup ---------------------- #
st.set_page_config(page_title="🖐️🎙️ HoloSign", layout="wide")
st.markdown("<h1 style='text-align: center;'>🖐️🎙️ HoloSign</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Real-Time Gesture and Voice Assistant</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------- Session State ---------------------- #
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = ""
if "last_detect_time" not in st.session_state:
    st.session_state.last_detect_time = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------- Tabs ---------------------- #
tab1, tab2, tab3 = st.tabs(["📸 Live Camera - Gesture Detection", "🎤 Voice Assistant", "💬 Emergency Chat"])

# ---------------------- Tab 1: Live Camera ---------------------- #
with tab1:
    st.subheader("📸 Real-Time Gesture Detection")
    run = st.toggle("🎥 Start Camera")

    video_placeholder = st.empty()
    gesture_display = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            while run:
                success, frame = cap.read()
                if not success:
                    st.warning("⚠️ Failed to access the camera.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                current_time = time.time()
                recognize_now = current_time - st.session_state.last_detect_time > 30
                gesture_text = ""

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        handedness = hand_handedness.classification[0].label
                        finger_state = fingers_up(hand_landmarks.landmark, handedness)
                        gesture_text = FINGER_COMBO_LABELS.get(finger_state, "")

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if gesture_text:
                            if recognize_now and gesture_text != st.session_state.last_gesture:
                                response = gesture_response(gesture_text)
                                speak(response, gesture_text)
                                st.session_state.last_gesture = gesture_text
                                st.session_state.last_detect_time = current_time

                # Display gesture text on frame
                label = f"Detected: {gesture_text}" if gesture_text else "No recognizable gesture"
                color = (0, 255, 0) if gesture_text else (0, 0, 255)
                cv2.putText(frame, label, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Update video frame in Streamlit
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # Fancy UI for gesture text display
                if gesture_text:
                    gesture_display.markdown(f"""
                        <div style="
                            background-color: #d4edda;
                            color: #155724;
                            padding: 15px 25px;
                            text-align: center;
                            border-radius: 15px;
                            font-size: 24px;
                            font-weight: bold;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            margin-top: 20px;
                        ">
                            ✋ Detected Gesture: <span style="color:#0c5460;">{gesture_text}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    gesture_display.markdown(f"""
                        <div style="
                            background-color: #f8d7da;
                            color: #721c24;
                            padding: 15px 25px;
                            text-align: center;
                            border-radius: 15px;
                            font-size: 20px;
                            font-weight: 600;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            margin-top: 20px;
                        ">
                            No recognizable gesture detected
                        </div>
                    """, unsafe_allow_html=True)

        cap.release()


# ---------------------- Tab 2: Voice Assistant ---------------------- #
with tab2:
    st.subheader("🎤 Voice Assistant")

    with st.container():
        st.markdown("Use your voice to trigger emergency messages or responses.")
        if st.button("🎙️ Tap to Speak"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                with st.spinner("Listening..."):
                    try:
                        audio = recognizer.listen(source, timeout=5)
                        voice_text = recognizer.recognize_google(audio)
                        st.success(f"You said: **{voice_text}**")
                        response = gesture_response(voice_text)
                        st.info(f"Response: {response}")
                        speak(response)
                    except sr.WaitTimeoutError:
                        st.error("Timeout. No voice detected.")
                    except sr.UnknownValueError:
                        st.error("Could not understand audio.")
                    except sr.RequestError:
                        st.error("Check your internet connection.")

# ---------------------- Tab 3: Emergency Chat ---------------------- #
with tab3:
    st.subheader("💬 Emergency Contact Chat")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your emergency message", placeholder="e.g., I need help urgently.")
        submitted = st.form_submit_button("🚨 Send")

        if submitted and user_input:
            st.session_state.chat_history.append(("You", user_input))

            try:
                # Replace this URL with your actual API endpoint
                response = requests.post(
                    "https://your-api.com/emergency",
                    json={"message": user_input}
                )

                if response.status_code == 200:
                    reply = response.json().get("status", "✅ Message delivered.")
                else:
                    reply = f"⚠️ Error: {response.status_code}"

            except Exception as e:
                reply = f"❌ Failed to send message: {e}"

            st.session_state.chat_history.append(("System", reply))

    with st.container():
        for sender, msg in reversed(st.session_state.chat_history[-10:]):
            bubble_color = "#dcf8c6" if sender == "You" else "#f1f0f0"
            align = "right" if sender == "You" else "left"
            st.markdown(
                f"""
                <div style='text-align: {align}; margin: 8px 0;'>
                    <div style='display: inline-block; padding: 10px 14px; background-color: {bubble_color}; border-radius: 12px; max-width: 75%;'>
                        <strong>{sender}:</strong><br>{msg}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
