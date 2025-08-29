import cv2
import pyttsx3
import easyocr
import speech_recognition as sr
from ultralytics import YOLO
import datetime
from textblob import TextBlob
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ================== Initialization ==================
yolo = YOLO("yolov8n.pt")   # YOLOv8 Nano for speed
ocr_reader = easyocr.Reader(['en'])
tts = pyttsx3.init()

# TrOCR (for handwriting OCR)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# TTS customization
voices = tts.getProperty('voices')
tts.setProperty('voice', voices[0].id)   # change index if you want female/male
tts.setProperty('rate', 170)

# ================== Helper Functions ==================
def speak(text):
    print(f"[AI]: {text}")
    tts.say(text)
    tts.runAndWait()

def detect_objects(frame):
    results = yolo.predict(frame, verbose=False)
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        label = yolo.model.names[int(cls)]
        speak(f"{label} detected")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def read_handwriting(frame):
    """Read handwritten text using TrOCR"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def read_text(frame, mode="printed"):
    if mode == "handwriting":
        try:
            text = read_handwriting(frame)
            speak(f"I think the handwriting says: {text}")
            with open("visionai_notes.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                f.write(f"Handwriting OCR: {text}\n")
        except Exception as e:
            speak("Sorry, handwriting OCR failed.")
            print(e)
    else:
        results = ocr_reader.readtext(frame)
        if not results:
            speak("No readable text found")
            return
        detected_lines = [text for (_, text, prob) in results if prob > 0.5]
        if detected_lines:
            raw_text = " ".join(detected_lines)
            corrected_text = str(TextBlob(raw_text).correct())
            speak(f"I detected the text: {raw_text}")
            speak(f"I think it means: {corrected_text}")
            with open("visionai_notes.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                f.write(f"Raw OCR: {raw_text}\n")
                f.write(f"Corrected: {corrected_text}\n")
        else:
            speak("No clear text detected")

def voice_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Listening for command...]")
        audio = r.listen(source, phrase_time_limit=5)
        try:
            command = r.recognize_google(audio)
            print(f"[User]: {command}")
            return command.lower()
        except:
            return ""

def tell_time():
    now = datetime.datetime.now()
    speak(f"The time is {now.strftime('%I:%M %p')}")

def tell_date():
    today = datetime.date.today()
    speak(f"Today is {today.strftime('%B %d, %Y')}")

def read_notes():
    try:
        with open("visionai_notes.txt", "r", encoding="utf-8") as f:
            notes = f.read().strip()
        if notes:
            speak("Here are your saved notes.")
            speak(notes[-500:])  # read last ~500 characters
        else:
            speak("Your notes file is empty.")
    except FileNotFoundError:
        speak("No notes have been saved yet.")

# ================== Camera Setup ==================
cap = cv2.VideoCapture(2, cv2.CAP_MSMF)  # Adjust index for Iriun webcam

speak("VisionAI is now active. Say a command.")

# ================== Main Loop ==================
while True:
    ret, frame = cap.read()
    if not ret:
        speak("Camera not available.")
        break

    frame = detect_objects(frame)
    cv2.imshow("VisionAI", frame)

    command = voice_command()

    if "read handwriting" in command:
        read_text(frame, mode="handwriting")
    elif "read" in command:
        read_text(frame, mode="printed")
    elif "time" in command:
        tell_time()
    elif "date" in command:
        tell_date()
    elif "open notes" in command or "read notes" in command:
        read_notes()
    elif "stop" in command or "exit" in command or "quit" in command:
        speak("Stopping VisionAI. Goodbye.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== Cleanup ==================
cap.release()
cv2.destroyAllWindows()
