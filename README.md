# Gesture_recog_py

HoloSign is now a Python-only Streamlit app for emergency gesture detection, voice commands, TTS, alarm playback, and emergency chat.

## What it does

- Detects the defined hand gestures with MediaPipe.
- Plays `Alarm.mp3` when `Alert (Fist)` or `Danger (Stop Gesture)` is detected.
- Triggers a Bland AI call when `Call Guardian`, `Need Help (Open Palm)`, `Pain (Two Fingers)`, or `Need Assistance` is detected.
- Listens for emergency voice keywords from the browser microphone.
- Lets you type emergency text commands in the chat panel.
- Shows live telemetry with the current gesture, sequence, urgency, and action.

## Gesture Map

- `Alert (Fist)` -> `0,0,0,0,0`
- `Call Guardian` -> `1,0,0,0,1` or `0,0,0,0,1`
- `Need Help (Open Palm)` -> `1,1,1,1,1`
- `Pain (Two Fingers)` -> `0,1,1,0,0`
- `Attention (One Finger)` -> `0,1,0,0,0`
- `OK (Thumbs Up)` -> `1,0,0,0,0`
- `Danger (Stop Gesture)` -> `0,1,1,1,1`
- `Need Assistance` -> `0,1,1,1,0`

## Run the app

```powershell
cd "C:\Users\Shruthi T\Desktop\Gesture_recog_py-main"
pip install -r requirements.txt
streamlit run Gesture.py
```

## Notes

- The React `frontend/` app has been retired.
- `Gesture.py` is the active app and contains the UI and emergency logic in one place.
