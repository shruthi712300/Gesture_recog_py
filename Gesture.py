"""HoloSign emergency console.

Python-only replacement for the React prototype.
This app keeps gesture detection, voice handling, emergency chat,
alarm playback, and Bland calls in one Streamlit file.
"""

from __future__ import annotations

import collections
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import speech_recognition as sr
import streamlit as st
import google.generativeai as genai
import ctypes
from PIL import Image
from playsound import playsound
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


st.set_page_config(page_title="HoloSign Emergency Console", layout="wide")


APP_DIR = Path(__file__).resolve().parent
ALARM_FILE = APP_DIR / "Alarm.mp3"
BLAND_CALL_URL = "https://api.bland.ai/v1/calls"

DEFAULT_BLAND_PHONE_NUMBER = "+91 9442637368"
DEFAULT_BLAND_PATHWAY_ID = "b9b44885-dd9e-4264-b895-b9bb74b112f0"
DEFAULT_BLAND_AUTHORIZATION = "org_7c7c9236a6be87fa8375068aca297619c588708f16c433f74e484a400b7f2949a49f08f17926058de26969"

CALL_KEYWORDS = ("emergency", "call", "help", "attack", "pain", "hurt", "guardian", "assistance", "need")
ALARM_KEYWORDS = ("alert", "alarm", "sound", "siren", "danger", "stop")
POSITIVE_KEYWORDS = ("good", "fine", "ok", "okay", "great", "thank")

GESTURE_LABELS = {
    (0, 0, 0, 0, 0): "Alert (Fist)",
    (1, 0, 0, 0, 1): "Call Guardian",
    (0, 0, 0, 0, 1): "Call Guardian",
    (1, 1, 1, 1, 1): "Need Help (Open Palm)",
    (0, 1, 1, 0, 0): "Pain (Two Fingers)",
    (0, 1, 0, 0, 0): "Attention (One Finger)",
    (1, 0, 0, 0, 0): "OK (Thumbs Up)",
    (0, 1, 1, 1, 1): "Danger (Stop Gesture)",
    (0, 1, 1, 1, 0): "Need Assistance",
}

GESTURE_REFERENCE = [
    {"name": "Alert (Fist)", "pattern": "0 0 0 0 0", "note": "All fingers folded. (Alarm triggers)"},
    {"name": "Call Guardian", "pattern": "1 0 0 0 1", "note": "Thumb and pinky extended. (Phone call triggers)"},
    {"name": "Need Help (Open Palm)", "pattern": "1 1 1 1 1", "note": "All fingers extended. (Voice 'Help is on the way' then alarm)"},
    {"name": "Pain (Two Fingers)", "pattern": "0 1 1 0 0", "note": "Index and middle extended. (Voice 'Use Emergency Chat...')"},
    {"name": "Attention (One Finger)", "pattern": "0 1 0 0 0", "note": "Index finger extended. (Voice 'Help is on the way')"},
    {"name": "OK (Thumbs Up)", "pattern": "1 0 0 0 0", "note": "Thumb extended. (Voice 'Sure Be Safe')"},
    {"name": "Danger (Stop Gesture)", "pattern": "0 1 1 1 1", "note": "Four fingers extended. (Alarm triggers)"},
    {"name": "Need Assistance", "pattern": "0 1 1 1 0", "note": "Index, middle, and ring extended. (Voice 'Calling 108....')"},
]

ACTION_COOLDOWNS = {
    "Play alert sound": 8.0,
    "Call guardian": 20.0,
    "Speak response": 4.0,
}


def format_sequence(items: List[Dict[str, object]]) -> str:
    if not items:
        return "Scanning"
    return " -> ".join(str(item["gesture"]) for item in items)


def fingers_up(landmarks, hand_label: str) -> Tuple[int, int, int, int, int]:
    fingers: List[int] = []
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

    return fingers[0], fingers[1], fingers[2], fingers[3], fingers[4]


def build_gesture_plan(gesture: str, sequence: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not gesture:
        return None

    if gesture == "Alert (Fist)":
        return {
            "intent": "Emergency Alert",
            "urgency": "Critical",
            "action": "Play alert sound",
            "reasoning": "Fist gesture detected.",
            "feedback": "Alert triggered.",
            "cooldown": 8.0,
        }

    if gesture == "Call Guardian":
        return {
            "intent": "Call Guardian",
            "urgency": "High",
            "action": "Call guardian",
            "reasoning": "Thumb and pinky extended.",
            "feedback": "Calling emergency contact now.",
            "cooldown": 20.0,
        }

    if gesture == "Need Help (Open Palm)":
        return {
            "intent": "Help Request",
            "urgency": "Critical",
            "action": "Play alert sound",
            "reasoning": "All fingers extended.",
            "feedback": "Help is on the way",
            "cooldown": 10.0,
        }

    if gesture == "Pain (Two Fingers)":
        return {
            "intent": "Pain Reported",
            "urgency": "High",
            "action": "Speak response",
            "reasoning": "Index and middle extended.",
            "feedback": "Use Emergency Chat to clear the doubt regardin the pain",
            "cooldown": 10.0,
        }

    if gesture == "Attention (One Finger)":
        return {
            "intent": "Attention Requested",
            "urgency": "Medium",
            "action": "Speak response",
            "reasoning": "Index finger extended.",
            "feedback": "Help is on the way",
            "cooldown": 5.0,
        }

    if gesture == "OK (Thumbs Up)":
        return {
            "intent": "OK Acknowledged",
            "urgency": "Low",
            "action": "Speak response",
            "reasoning": "Thumb extended.",
            "feedback": "Sure Be Safe",
            "cooldown": 5.0,
        }

    if gesture == "Danger (Stop Gesture)":
        return {
            "intent": "Danger Detected",
            "urgency": "Critical",
            "action": "Play alert sound",
            "reasoning": "Four fingers extended.",
            "feedback": "Danger detected. Alarm triggered.",
            "cooldown": 8.0,
        }

    if gesture == "Need Assistance":
        return {
            "intent": "Assistance Needed",
            "urgency": "High",
            "action": "Speak response",
            "reasoning": "Index, middle, and ring extended.",
            "feedback": "Calling 108....",
            "cooldown": 10.0,
        }

    return None


def build_text_plan(text: str) -> Dict[str, object]:
    normalized = text.lower()

    if any(word in normalized for word in CALL_KEYWORDS):
        return {
            "intent": "Emergency call requested",
            "urgency": "High",
            "action": "Call guardian",
            "reasoning": "Keyword matched for call.",
            "feedback": "Calling emergency contact now.",
            "cooldown": 20.0,
        }

    if any(word in normalized for word in ALARM_KEYWORDS):
        return {
            "intent": "Alarm requested",
            "urgency": "Critical",
            "action": "Play alert sound",
            "reasoning": "Keyword matched for alarm.",
            "feedback": "Alert triggered.",
            "cooldown": 8.0,
        }

    if any(word in normalized for word in POSITIVE_KEYWORDS):
        return {
            "intent": "Acknowledged",
            "urgency": "Low",
            "action": "Speak response",
            "reasoning": "Positive keyword detected.",
            "feedback": "Acknowledged.",
            "cooldown": 4.0,
        }

    return {
        "intent": "Message received",
        "urgency": "Low",
        "action": "Speak response",
        "reasoning": "No emergency keyword detected.",
        "feedback": "Message received. Monitoring situation.",
        "cooldown": 4.0,
    }


class EmergencyState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_action_times: Dict[str, float] = {}
        self.recent_gestures: Deque[Dict[str, object]] = collections.deque(maxlen=12)
        self.telemetry: Deque[Dict[str, object]] = collections.deque(maxlen=20)
        self.chat_history: Deque[Dict[str, object]] = collections.deque(maxlen=30)
        self.voice_buffer: List[bytes] = []
        self.last_voice_command = ""
        self.voice_transcript = ""
        self.camera_status = "Scanning"
        self.voice_status = "Voice idle"
        self.speech_status = "Speech ready"
        self.call_status = "Idle"
        self.alarm_status = "Ready"
        self.reminders: List[Dict[str, object]] = []
        self.latest_event: Dict[str, object] = {
            "time_epoch": 0.0,
            "time": "--",
            "source": "System",
            "gesture": "Awaiting camera",
            "sequence": "Scanning",
            "intent": "Awaiting gesture",
            "urgency": "Idle",
            "action": "None",
            "message": "Start the camera or microphone to begin.",
            "details": "",
        }

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "camera_status": self.camera_status,
                "voice_status": self.voice_status,
                "speech_status": self.speech_status,
                "call_status": self.call_status,
                "alarm_status": self.alarm_status,
                "latest_event": dict(self.latest_event),
                "telemetry": list(self.telemetry),
                "chat_history": list(self.chat_history),
                "voice_transcript": self.voice_transcript,
            }

    def claim_action(self, action: str, cooldown: float) -> bool:
        now = time.time()
        with self.lock:
            last_time = self.last_action_times.get(action, 0.0)
            if now - last_time < cooldown:
                return False
            self.last_action_times[action] = now
            return True

    def register_gesture(self, gesture: str) -> List[Dict[str, object]]:
        now = time.time()
        with self.lock:
            if (
                not self.recent_gestures
                or self.recent_gestures[-1]["gesture"] != gesture
                or now - float(self.recent_gestures[-1]["time"]) > 1.5
            ):
                self.recent_gestures.append({"gesture": gesture, "time": now})

            filtered = [item for item in self.recent_gestures if now - float(item["time"]) <= 10.0]
            self.recent_gestures = collections.deque(filtered, maxlen=12)
            return list(self.recent_gestures)

    def set_status(self, field: str, value: str) -> None:
        with self.lock:
            setattr(self, field, value)

    def append_chat(self, sender: str, message: str) -> None:
        with self.lock:
            self.chat_history.append(
                {"sender": sender, "message": message, "time": datetime.now().strftime("%I:%M:%S %p")}
            )

    def set_voice_transcript(self, text: str) -> None:
        with self.lock:
            self.voice_transcript = text
            self.voice_status = "Listening"

    def record_event(
        self,
        *,
        source: str,
        gesture: str,
        sequence: str,
        intent: str,
        urgency: str,
        action: str,
        message: str,
        details: str = "",
    ) -> Dict[str, object]:
        event = {
            "time_epoch": time.time(),
            "time": datetime.now().strftime("%I:%M:%S %p"),
            "source": source,
            "gesture": gesture,
            "sequence": sequence,
            "intent": intent,
            "urgency": urgency,
            "action": action,
            "message": message,
            "details": details,
        }
        with self.lock:
            self.latest_event = event
            self.telemetry.appendleft(event.copy())
        return event

    def update_event(self, **changes: object) -> None:
        with self.lock:
            self.latest_event.update(changes)
            self.telemetry.appendleft(self.latest_event.copy())


class AppConfig:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.phone_number = DEFAULT_BLAND_PHONE_NUMBER
        self.pathway_id = DEFAULT_BLAND_PATHWAY_ID
        self.authorization = DEFAULT_BLAND_AUTHORIZATION
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    def update(self, phone_number: str, pathway_id: str, authorization: str, gemini_api_key: str = "") -> None:
        with self.lock:
            self.phone_number = phone_number
            self.pathway_id = pathway_id
            self.authorization = authorization
            if gemini_api_key:
                self.gemini_api_key = gemini_api_key

    def snapshot(self) -> Dict[str, str]:
        with self.lock:
            return {
                "phone_number": getattr(self, "phone_number", DEFAULT_BLAND_PHONE_NUMBER),
                "pathway_id": getattr(self, "pathway_id", DEFAULT_BLAND_PATHWAY_ID),
                "authorization": getattr(self, "authorization", DEFAULT_BLAND_AUTHORIZATION),
                "gemini_api_key": getattr(self, "gemini_api_key", ""),
            }


@st.cache_resource
def get_state(v=3) -> EmergencyState:
    return EmergencyState()


@st.cache_resource
def get_config(v=3) -> AppConfig:
    return AppConfig()


@st.cache_resource
def get_recognizer() -> sr.Recognizer:
    return sr.Recognizer()


@st.cache_resource
def get_tts_queue() -> queue.Queue[Tuple[str, int]]:
    speech_queue: queue.Queue[Tuple[str, int]] = queue.Queue()

    def worker() -> None:
        try:
            engine = pyttsx3.init()
        except Exception as exc:
            print(f"pyttsx3 init error: {exc}")
            return

        voices = engine.getProperty("voices") or []
        if voices:
            engine.setProperty("voice", voices[0].id)

        while True:
            item = speech_queue.get()
            if item is None:
                speech_queue.task_done()
                break

            text, rate = item
            state = get_state()
            state.set_status("speech_status", "Speaking")
            try:
                engine.setProperty("rate", rate)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                state.set_status("speech_status", f"Speech failed: {exc}")
                print(f"pyttsx3 speak error: {exc}")
            else:
                state.set_status("speech_status", "Speech ready")
            finally:
                speech_queue.task_done()

    threading.Thread(target=worker, daemon=True).start()
    return speech_queue


def speak(text: str, rate: int = 150) -> None:
    if not text:
        return
    get_tts_queue().put((text, rate))


def get_medical_ai_response(text: str) -> str:
    """Uses Gemini API to provide medical-related assistance."""
    config = get_config().snapshot()
    api_key = config.get("gemini_api_key")
    if not api_key:
        return "Gemini API key is missing. Please configure it in the sidebar."

    try:
        genai.configure(api_key=api_key)
        
        # Disable safety filters to ensure medical advice is not blocked
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Auto-discover working models for this API key to fix 404 errors
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception as e:
            return f"API Key Error: Could not list models. Please check your key in the sidebar. ({e})"

        if not available_models:
            return "No compatible Generative AI models found for this API key. Ensure 'Generative Language API' is enabled in Google Cloud Console."

        # Pick the best available model (Prefer 1.5 Flash, then 1.5 Pro, then 1.0 Pro, then whatever is first)
        target_model = available_models[0]
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro', 'models/gemini-1.0-pro']:
            if preferred in available_models:
                target_model = preferred
                break

        model = genai.GenerativeModel(target_model, safety_settings=safety_settings)
        system_prompt = (
            "You are a helpful and concise medical emergency assistant. "
            "Provide brief, accurate first-aid advice or guidance based on the user's message. "
            "Always remind the user if they should call emergency services. "
            "Keep responses under 2-3 sentences."
        )
        response = model.generate_content(f"{system_prompt}\n\nUser: {text}")
            
        return response.text
    except Exception as exc:
        return f"Medical AI Error: {exc}"


def reminder_monitor():
    """Background thread that speaks reminders when the time matches."""
    state = get_state()
    while True:
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            
            for r in state.reminders:
                if r["time"] == current_time and not r.get("triggered", False):
                    speak(f"Time Alert: It is time to take your medication: {r['medicine']}. Please follow your prescription.")
                    r["triggered"] = True
        except Exception as e:
            print(f"Reminder error: {e}")
        time.sleep(30)


@st.cache_resource
def start_reminder_service():
    """Starts the reminder monitor thread only once."""
    thread = threading.Thread(target=reminder_monitor, daemon=True)
    thread.start()
    return True


def process_medical_document(uploaded_file) -> str:
    """Uses Gemini Vision to analyze medical documents (Images and PDFs)."""
    config = get_config().snapshot()
    api_key = config.get("gemini_api_key")
    if not api_key:
        return "Gemini API key is missing. Please configure it in the sidebar."

    try:
        genai.configure(api_key=api_key)
        
        # Auto-discover working models for this API key to fix 404 errors
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception as e:
            return f"API Key Error: Could not list models. ({e})"

        if not available_models:
            return "No compatible models found for this API key."

        # Pick the best available model
        target_model = available_models[0]
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
            if preferred in available_models:
                target_model = preferred
                break

        model = genai.GenerativeModel(target_model)
        
        # Determine content type
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            content = {
                "mime_type": "application/pdf",
                "data": uploaded_file.getvalue()
            }
        else:
            content = Image.open(uploaded_file)
        
        prompt = (
            "Analyze this medical document and provide a structured summary in an understandable format:\n\n"
            "1. If it is a MEDICAL REPORT: Extract the disease name, whether the result is positive or negative, "
            "and the severity (if mentioned).\n"
            "2. If it is a BILL or RECEIPT: Extract the Hospital Name, Date of Admission/Service, and other important charges or details.\n"
            "3. If it is a PRESCRIPTION: List all medicines. For each, give the Medical Name, its General (common) Name, "
            "what the medicine is used for, and why it was prescribed.\n\n"
            "CRITICAL: If it is a prescription, also create a doses schedule. At the very end of your response, "
            "add a special section starting with '---SCHEDULE_DATA---' and then list doses in 'MedicineName|HH:MM' format, one per line (24-hour time).\n\n"
            "Format the output using clear headers, bold text, and bullet points for high readability."
        )
        
        response_text = model.generate_content([prompt, content]).text
        
        # Parse schedule data and add to state
        if "---SCHEDULE_DATA---" in response_text:
            parts = response_text.split("---SCHEDULE_DATA---")
            main_info = parts[0].strip()
            sched_block = parts[1].strip()
            
            state = get_state()
            for line in sched_block.split("\n"):
                if "|" in line:
                    med, time_str = line.split("|", 1)
                    state.reminders.append({
                        "medicine": med.strip(),
                        "time": time_str.strip(),
                        "triggered": False,
                        "date": datetime.now().strftime("%Y-%m-%d")
                    })
            return main_info
            
        return response_text
    except Exception as exc:
        return f"Document Analysis Error: {exc}"


def trigger_bland_call(phone_number: str, pathway_id: str, authorization: str) -> Tuple[bool, str]:
    payload = {"phone_number": phone_number, "pathway_id": pathway_id}
    headers = {"authorization": authorization, "Content-Type": "application/json"}

    try:
        response = requests.post(BLAND_CALL_URL, json=payload, headers=headers, timeout=15, verify=False)
        if 200 <= response.status_code < 300:
            return True, "Emergency call initiated successfully."
        return False, f"Call failed: {response.text}"
    except requests.exceptions.ConnectionError:
        return False, "Call failed: Network connection error. Please check your internet connection."
    except requests.exceptions.Timeout:
        return False, "Call failed: Request timed out. The emergency call service is not responding."
    except Exception as exc:
        return False, f"Call failed: {exc}"


def play_mp3_mci(path: str) -> None:
    """Plays an MP3 file using Windows MCI."""
    try:
        ctypes.windll.winmm.mciSendStringW("close alarm", None, 0, 0)
        ctypes.windll.winmm.mciSendStringW(f'open "{path}" type mpegvideo alias alarm', None, 0, 0)
        ctypes.windll.winmm.mciSendStringW("play alarm", None, 0, 0)
    except Exception as exc:
        print(f"MCI play error: {exc}")


def stop_mp3_mci() -> None:
    """Stops the MP3 file playback."""
    try:
        ctypes.windll.winmm.mciSendStringW("stop alarm", None, 0, 0)
        ctypes.windll.winmm.mciSendStringW("close alarm", None, 0, 0)
    except Exception as exc:
        print(f"MCI stop error: {exc}")


def play_alarm() -> None:
    state = get_state()

    def _play() -> None:
        state.set_status("alarm_status", "Alarm playing")
        try:
            if ALARM_FILE.exists():
                play_mp3_mci(str(ALARM_FILE))
                time.sleep(2.0)  # User requested 2-second alarm
                stop_mp3_mci()
            else:
                state.append_chat("System", f"Alarm file not found: {ALARM_FILE}")
                state.set_status("alarm_status", "Alarm missing")
        except Exception as exc:
            state.set_status("alarm_status", f"Alarm blocked: {exc}")
            print(f"Siren error: {exc}")
        finally:
            state.set_status("alarm_status", "Ready")

    threading.Thread(target=_play, daemon=True).start()


def dispatch_plan(
    plan: Optional[Dict[str, object]],
    *,
    source: str,
    gesture: str,
    sequence: str,
    bland_phone_number: str,
    bland_pathway_id: str,
    bland_authorization: str,
) -> Optional[Dict[str, object]]:
    if not plan:
        return None

    state = get_state()
    action = str(plan["action"])
    cooldown = float(plan.get("cooldown", ACTION_COOLDOWNS.get(action, 4.0)))

    now = time.time()
    if not state.claim_action(action, cooldown):
        return None

    event = state.record_event(
        source=source,
        gesture=gesture,
        sequence=sequence,
        intent=str(plan["intent"]),
        urgency=str(plan["urgency"]),
        action=action,
        message=str(plan.get("feedback", plan["intent"])),
        details=str(plan.get("reasoning", "")),
    )

    if source != "gesture":
        state.append_chat("System", event["message"])

    def worker() -> None:
        # Voice notification of detection first
        speak(f"Detected Gesture - {gesture}", rate=150)
        time.sleep(1.5)  # Wait for detection voice to finish

        if action == "Play alert sound":
            feedback = str(plan.get("feedback", "Alert triggered."))
            speak(feedback, rate=135)
            if gesture == "Need Help (Open Palm)":
                time.sleep(2.5)  # Let the voice finish before starting alarm
            play_alarm()
        elif action == "Call guardian":
            state.set_status("call_status", "Calling guardian")
            ok, message = trigger_bland_call(bland_phone_number, bland_pathway_id, bland_authorization)
            state.set_status("call_status", "Call sent" if ok else "Call failed")
            final_message = str(plan.get("feedback", message)) if ok else message
            state.update_event(message=final_message)
            speak(final_message, rate=150)
            if not ok:
                state.append_chat("System", final_message)
        elif action == "Speak response":
            speak(str(plan.get("feedback", "Acknowledged.")), rate=150)

    threading.Thread(target=worker, daemon=True).start()
    return event


def handle_text_command(
    text: str,
    *,
    source: str,
    bland_phone_number: str,
    bland_pathway_id: str,
    bland_authorization: str,
) -> Optional[Dict[str, object]]:
    trimmed = text.strip()
    if not trimmed:
        return None

    state = get_state()
    if source == "chat":
        state.append_chat("You", trimmed)
    elif source == "voice":
        state.append_chat("Voice", trimmed)

    plan = build_text_plan(trimmed)
    sequence = format_sequence(list(get_state().recent_gestures))
    
    # If it's an emergency, dispatch the plan (calls/alarms)
    event = None
    if plan["urgency"] in {"High", "Critical"}:
        event = dispatch_plan(
            plan,
            source=source,
            gesture="Voice command" if source == "voice" else "Text command",
            sequence=sequence,
            bland_phone_number=bland_phone_number,
            bland_pathway_id=bland_pathway_id,
            bland_authorization=bland_authorization,
        )

    # Use Gemini for medical context/assistance
    def gemini_worker():
        ai_resp = get_medical_ai_response(trimmed)
        state.append_chat("Medical AI", ai_resp)
        speak(ai_resp)

    threading.Thread(target=gemini_worker, daemon=True).start()

    if source == "voice":
        state.set_voice_transcript(trimmed)

    return event


class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.draw = mp.solutions.drawing_utils
        self.connections = mp.solutions.hands.HAND_CONNECTIONS
        self.frame_queue: queue.Queue[av.VideoFrame] = queue.Queue(maxsize=1)
        self.gesture_history: Deque[str] = collections.deque(maxlen=5)
        self.last_trigger_time: Dict[str, float] = {}
        self.frame_count = 0
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    def _process_loop(self) -> None:
        """Background thread for logic and voice."""
        while True:
            try:
                frame = self.frame_queue.get()
                image = frame.to_ndarray(format="bgr24")
                h, w, _ = image.shape
                scale = 320 / max(h, w)
                small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
                
                results = self.hands.process(rgb)
                state = get_state()
                config = get_config().snapshot()
                now = time.time()
                
                detected_gesture = ""
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        handedness = hand_handedness.classification[0].label
                        finger_state = fingers_up(hand_landmarks.landmark, handedness)
                        detected_gesture = GESTURE_LABELS.get(finger_state, "")

                self.gesture_history.append(detected_gesture)
                stable_gesture = ""
                if len(self.gesture_history) == 5 and all(g == self.gesture_history[0] for g in self.gesture_history):
                    stable_gesture = self.gesture_history[0]

                if stable_gesture:
                    last_time = self.last_trigger_time.get(stable_gesture, 0.0)
                    if now - last_time > 3.0:
                        self.last_trigger_time[stable_gesture] = now
                        sequence_items = state.register_gesture(stable_gesture)
                        state.set_status("camera_status", f"Executed {stable_gesture}")
                        plan = build_gesture_plan(stable_gesture, sequence_items)
                        if plan:
                            dispatch_plan(plan, source="gesture", gesture=stable_gesture, sequence=format_sequence(sequence_items),
                                        bland_phone_number=config["phone_number"], bland_pathway_id=config["pathway_id"], bland_authorization=config["authorization"])
            except Exception:
                pass
            finally:
                time.sleep(0.05)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Put the latest frame in the queue for the AI thread
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # AI is busy, skip this frame to keep video fluid
        
        # Return the frame immediately - NO BLOCKING
        return frame


def render_metric_cards(snapshot: Dict[str, object]) -> None:
    cols = st.columns(4)
    metrics = [
        ("Gesture", snapshot["latest_event"]["gesture"]),
        ("Sequence", snapshot["latest_event"]["sequence"]),
        ("Urgency", snapshot["latest_event"]["urgency"]),
        ("Action", snapshot["latest_event"]["action"]),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); border: 2px solid #000; border-radius: 12px; padding: 12px 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #000; margin-bottom: 4px; font-weight: 900;">{label}</div>
                <div style="font-size: 16px; font-weight: 900; color: #000; word-break: break-word;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_status_row(snapshot: Dict[str, object]) -> None:
    chips = [
        f"Camera: {snapshot['camera_status']}",
        f"Voice: {snapshot['voice_status']}",
        f"Speech: {snapshot['speech_status']}",
        f"Call: {snapshot['call_status']}",
        f"Alarm: {snapshot['alarm_status']}",
    ]
    st.write(" | ".join(chips))


def render_telemetry(entries: List[Dict[str, object]]) -> None:
    if not entries:
        st.info("Waiting for a gesture, voice command, or chat message.")
        return

    for entry in entries[:10]:
        urgency_class = str(entry.get("urgency", "low")).lower()
        color = "#000" 
        if urgency_class == "critical": color = "#dc2626"
        elif urgency_class == "high": color = "#ea580c"

        st.markdown(
            f"""
            <div style="border-left: 8px solid {color}; background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px); border-radius: 12px; padding: 12px 15px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #000; margin-bottom: 5px; font-weight: 900;">
                    <span style="font-weight: 900;">{entry.get("source", "System")}</span>
                    <span>{entry.get("time", "--")}</span>
                </div>
                <div style="font-size: 16px; font-weight: 900; color: {color};">{entry.get("intent", "")}</div>
                <div style="font-size: 13px; color: #000; margin-top: 3px; font-family: monospace; font-weight: 900;">{entry.get("sequence", "")}</div>
                <div style="display: flex; gap: 10px; margin-top: 8px; justify-content: center;">
                    <span style="background: rgba(0,0,0,0.1); color: #000; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 900; border: 1px solid #000;">{entry.get("action", "")}</span>
                    <span style="background: rgba(0,0,0,0.1); color: #000; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 900; border: 1px solid #000;">URGENCY: {urgency_class.upper()}</span>
                </div>
                <div style="font-size: 13px; margin-top: 8px; color: #000; line-height: 1.4; font-weight: 900;">{entry.get("message", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_gesture_map() -> None:
    st.subheader("Gesture Reference")
    selected_gesture = st.selectbox(
        "Select a gesture to see details",
        options=[item['name'] for item in GESTURE_REFERENCE],
        key="gesture_map_dropdown"
    )
    
    item = next(i for i in GESTURE_REFERENCE if i['name'] == selected_gesture)
    st.info(f"**Pattern:** {item['pattern']}\n\n**Note:** {item['note']}")


def render_chat_history(history: List[Dict[str, object]]) -> None:
    if not history:
        st.info("No chat messages yet.")
        return

    for item in history[-10:]:
        sender = item.get("sender", "System")
        message = item.get("message", "")
        time_label = item.get("time", "--")
        is_user = sender == "You"
        bubble_color = "rgba(0, 0, 0, 0.1)" if is_user else "rgba(255, 255, 255, 0.4)"
        text_align = "right" if is_user else "left"
        border_radius = "15px 15px 2px 15px" if is_user else "15px 15px 15px 2px"
        
        st.markdown(
            f"""
            <div style="margin-bottom: 12px; text-align: {text_align};">
                <div style="display: inline-block; max-width: 80%; padding: 12px 16px; background: {bubble_color}; backdrop-filter: blur(10px); border-radius: {border_radius}; border: 2px solid #000; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 10px; text-transform: uppercase; font-weight: 900; color: #000; margin-bottom: 4px;">{sender} • {time_label}</div>
                    <div style="font-size: 14px; color: #000; line-height: 1.5; font-weight: 900;">{message}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    if "app_started" not in st.session_state:
        st.session_state.app_started = False

    # Global Premium Styling
    import base64
    bg_img_path = APP_DIR / "holosign_bg.png"
    encoded_bg = ""
    if bg_img_path.exists():
        with open(bg_img_path, "rb") as f:
            encoded_bg = base64.b64encode(f.read()).decode()

    # Global Premium Styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        h1 { font-size: 3rem !important; font-weight: 600 !important; color: #1E293B; }
        h2 { font-size: 2.2rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.6rem !important; font-weight: 500 !important; }
        
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            font-weight: 600;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    if not st.session_state.app_started:
        st.title("🛡️ Welcome to HoloSign")
        st.subheader("Your AI-Powered Emergency Assistant")
        st.write("Experience the next generation of personal safety with gesture-controlled emergency response.")
        
        tab1, tab2, tab3 = st.tabs(["👋 Welcome", "⚙️ Setup", "📚 Tutorial"])
        
        with tab1:
            st.markdown(
                """
                <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 20px; border: 2px solid #000; text-align: center;">
                    <h3 style="margin-top:0; font-weight: 900; color: #000;">What is HoloSign?</h3>
                    <p style="font-weight: 900; color: #000;">HoloSign is an advanced, touch-free emergency console designed to keep you safe. 
                    It constantly monitors for emergencies using:</p>
                    <ul style="list-style-type: none; padding-left: 0; font-weight: 900; color: #000;">
                        <li style="margin-bottom: 10px;"><b>Live Camera Gestures:</b> Recognize hand signals in real-time.</li>
                        <li style="margin-bottom: 10px;"><b>Voice Assistant:</b> Listen for emergency keywords.</li>
                        <li style="margin-bottom: 10px;"><b>Emergency Chat:</b> Direct text messaging for help.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 ENTER EMERGENCY CONSOLE", key="center_enter_btn"):
                st.session_state.app_started = True
                st.rerun()
            
        with tab2:
            st.markdown("### Emergency Contact Configuration")
            st.write("Set up who should be contacted in case of an emergency.")
            config = get_config()
            
            bland_phone_number = st.text_input("Guardian Phone Number", value=config.snapshot()["phone_number"], key="setup_phone")
            bland_pathway_id = st.text_input("Bland Pathway ID", value=config.snapshot()["pathway_id"], key="setup_pathway")
            bland_authorization = st.text_input("Bland Authorization", value=config.snapshot()["authorization"], type="password", key="setup_auth")
            
            if st.button("Save Configuration", type="secondary"):
                config.update(bland_phone_number, bland_pathway_id, bland_authorization)
                st.success("Configuration saved! Go to the **Tutorial** tab next.")
                
        with tab3:
            st.markdown("### Quick Tutorial")
            st.write("Provides brief, expandable cards that illustrate each gesture’s pattern and the corresponding action, so new users can quickly learn how to control HoloSign with their hands.")
            
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Live Camera Feed"):
                    st.write("This tab streams your webcam at a smooth 60 FPS while the background AI continuously watches for hand gestures. When a recognized gesture appears (e.g., a fist, open palm, thumb‑pinky, or thumbs‑up), the system instantly executes the associated emergency action—playing an alarm, calling your Guardian, or canceling an alert—without any visual lag.")
                with st.expander("Voice Assistant"):
                    st.write("Here you can speak naturally to the console. The built‑in speech recognizer listens for emergency keywords and confirms actions with spoken feedback (“Calling your Guardian now”). The assistant also reads out the status of the system, letting you operate hands‑free when you’re unable to look at the screen.")
                with st.expander("Emergency Chat"):
                    st.write("This is a text‑based chat powered by Gemini. Type or paste a medical query, and the AI returns concise first‑aid advice, symptom triage, or medication information. The chat automatically refreshes when a response arrives, so you always see the latest answer without reloading the page.")
            with col2:
                with st.expander("Medical Docs"):
                    st.write("Upload a scanned image or PDF of any medical document—prescriptions, bills, or diagnostic reports. Gemini’s vision model extracts the essential data (disease name, test results, hospital name, dates, medication list, etc.) and presents it in a clear, readable format.")
                with st.expander("Medication Calendar"):
                    st.write("After a prescription is processed, any dosage schedule found in the document is automatically added to this tab. Each entry shows the medication name, scheduled time, and a status badge. You can mark a dose as Done with a single checkbox, and the background reminder service will speak a reminder at the appropriate time if the dose is still pending.")
            st.markdown("---")
            st.info("Ensure your camera and microphone are ready.", icon="ℹ️")
            if st.button("🚀 Enter Emergency Console", use_container_width=True, type="primary"):
                st.session_state.app_started = True
                st.rerun()
        return

    state = get_state()
    config = get_config()
    start_reminder_service() # Ensure the voice reminder thread is running

    st.title("HoloSign Emergency Console")
    st.caption("Python-only gesture detection, voice response, and emergency chat.")

    with st.sidebar:
        st.subheader("Emergency Config")
        bland_phone_number = st.text_input("Guardian phone number", value=config.snapshot()["phone_number"])
        bland_pathway_id = st.text_input("Bland pathway ID", value=config.snapshot()["pathway_id"])
        bland_authorization = st.text_input("Bland authorization", value=config.snapshot().get("authorization", ""), type="password")
        gemini_api_key = st.text_input("Gemini API Key", value=config.snapshot().get("gemini_api_key", ""), type="password")
        config.update(bland_phone_number, bland_pathway_id, bland_authorization, gemini_api_key)
        
        st.divider()
        st.subheader("Controls")
        if st.button("🔴 STOP ALARM", use_container_width=True, type="primary"):
            stop_mp3_mci()
            st.warning("Alarm stopped manually.")

        st.divider()
        render_gesture_map()

        st.divider()
        st.caption("The Python app handles both the UI and the emergency logic directly.")
        if st.button("Clear logs"):
            with state.lock:
                state.telemetry.clear()
                state.chat_history.clear()
                state.recent_gestures.clear()
                state.voice_buffer.clear()
                state.last_voice_command = ""
                state.voice_transcript = ""
                state.latest_event = {
                    "time_epoch": 0.0,
                    "time": "--",
                    "source": "System",
                    "gesture": "Awaiting camera",
                    "sequence": "Scanning",
                    "intent": "Awaiting gesture",
                    "urgency": "Idle",
                    "action": "None",
                    "message": "Logs cleared.",
                    "details": "",
                }
            st.success("Logs cleared.")

    snapshot = state.snapshot()
    render_status_row(snapshot)

    tab_camera, tab_voice, tab_chat, tab_docs, tab_cal = st.tabs(["Live Camera", "Voice Assistant", "Emergency Chat", "Medical Docs", "Medication Calendar"])

    with tab_camera:
        st.header(f"Gesture: {snapshot['latest_event']['gesture']}")
        
        webrtc_streamer(
            key="gesture-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=GestureVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        snapshot = state.snapshot()
        st.markdown("### Live State")
        render_metric_cards(snapshot)

        st.markdown("### Telemetry")
        render_telemetry(snapshot["telemetry"])

    with tab_voice:
        st.subheader("Voice Assistant")
        st.caption("Speak emergency keywords into the browser microphone.")

        recognizer = get_recognizer()

        def audio_frame_callback(frame):
            local_state = get_state()
            audio = frame.to_ndarray()
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
            audio = np.asarray(audio, dtype=np.int16)

            with local_state.lock:
                local_state.voice_buffer.append(audio.tobytes())
                buffered = list(local_state.voice_buffer)

            if len(buffered) < 50:
                return frame

            with local_state.lock:
                full_audio = b"".join(local_state.voice_buffer)
                local_state.voice_buffer.clear()

            try:
                audio_data = sr.AudioData(full_audio, frame.sample_rate, 2)
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                local_state.set_status("voice_status", "Listening")
                return frame
            except sr.RequestError as exc:
                local_state.set_status("voice_status", f"Voice error: {exc}")
                return frame
            except Exception as exc:
                local_state.set_status("voice_status", f"Voice error: {exc}")
                return frame

            cleaned = text.strip()
            if cleaned:
                local_state.set_voice_transcript(cleaned)
                if cleaned != local_state.last_voice_command:
                    local_state.last_voice_command = cleaned
                    handle_text_command(
                        cleaned,
                        source="voice",
                        bland_phone_number=config.snapshot()["phone_number"],
                        bland_pathway_id=config.snapshot()["pathway_id"],
                        bland_authorization=config.snapshot()["authorization"],
                    )
            return frame

        webrtc_streamer(
            key="voice-stream",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_frame_callback,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        snapshot = state.snapshot()
        st.markdown("### Voice Status")
        st.write(snapshot["voice_status"])
        st.write(f"Transcript: {snapshot['voice_transcript'] or 'Waiting for voice input.'}")

        if st.button("Speak test response"):
            speak("Speech synthesis is working in the Python app.", rate=150)
            st.success("Spoken test phrase queued.")

    with tab_chat:
        st.subheader("Emergency Chat")
        st.caption("Typing here uses the same alarm and call workflow as gestures and voice.")

        with st.form("emergency-chat-form", clear_on_submit=True):
            user_input = st.text_input("Type your emergency message", placeholder="I need help urgently.")
            submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            handle_text_command(
                user_input,
                source="chat",
                bland_phone_number=config.snapshot()["phone_number"],
                bland_pathway_id=config.snapshot()["pathway_id"],
                bland_authorization=config.snapshot()["authorization"],
            )
            st.success("Command processed.")

        snapshot = state.snapshot()
        st.markdown("### Chat History")
        render_chat_history(snapshot["chat_history"])
        
        # Small hack to auto-refresh when AI might be typing
        if any(msg["sender"] == "You" for msg in snapshot["chat_history"][-2:]):
             st.caption("Waiting for AI response... (Page auto-refreshes)")
             time.sleep(1)
             st.rerun()

    with tab_docs:
        st.subheader("Medical Document Assistant")
        st.write("Upload a medical report, bill, or prescription for AI extraction.")
        
        uploaded_file = st.file_uploader("Choose a document...", type=["jpg", "jpeg", "png", "pdf"], key="doc_uploader")
        
        if uploaded_file is not None:
            # Only show st.image if it's a picture. PDFs will crash st.image.
            file_name = uploaded_file.name.lower()
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
            elif file_name.endswith(".pdf"):
                st.info(f"📄 PDF Document Uploaded: **{uploaded_file.name}**")
            
            if st.button("Extract Information", type="primary"):
                with st.spinner("Analyzing document details..."):
                    doc_results = process_medical_document(uploaded_file)
                    st.divider()
                    st.markdown("### Extraction Results")
                    st.markdown(doc_results)
                    
                    # Also speak the summary if it's short
                    if len(doc_results) < 500:
                        speak("Analysis complete. I have added the medication reminders to your schedule.")

    with tab_cal:
        st.header("📅 Medication Calendar")
        st.write("View and manage your scheduled medicine reminders.")
        
        # Show active reminders
        if hasattr(state, "reminders") and state.reminders:
            for i, r in enumerate(state.reminders):
                col1, col2 = st.columns([0.8, 0.2])
                is_done = r.get("triggered", False)
                status = "✅ Completed" if is_done else "🔔 Pending"
                
                with col1:
                    st.info(f"**{r['time']}** — {r['medicine']} — Status: {status}")
                
                with col2:
                    # If user clicks the checkbox, update the 'triggered' state
                    if st.checkbox("Done", value=is_done, key=f"rem_{i}"):
                        r["triggered"] = True
                    else:
                        r["triggered"] = False
            
            st.divider()
            if st.button("Clear All Reminders", type="primary"):
                state.reminders = []
                st.rerun()
        else:
            st.write("No active reminders. Upload a prescription in the **Medical Docs** tab to set a schedule.")



if __name__ == "__main__":
    main()
