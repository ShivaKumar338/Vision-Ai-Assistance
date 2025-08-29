
# VisionAI – AI Powered Voice & Vision Assistant 👁️🎙️

VisionAI is a real-time **AI assistant** that combines **computer vision** and **voice interaction**.  
It detects objects, reads printed/handwritten text, understands your voice commands, and even speaks back to you.  

---

## ✨ Features
- 🔍 **Object Detection** – Uses YOLOv8 for real-time object detection.  
- 📝 **Text Reading** –  
  - EasyOCR for printed text.  
  - TrOCR (Hugging Face) for handwritten text.  
- 🎤 **Voice Commands** – Control the assistant using natural speech.  
- 🗣️ **Text-to-Speech** – Speaks detected text, objects, time, and notes.  
- 📒 **Notes System** – Saves recognized text into `visionai_notes.txt` and can read it back.  
- 🕒 **Date & Time** – Ask for the current date or time.  
- 🎛️ **Interactive** – Commands like:
  - `"read handwriting"`  
  - `"read"`  
  - `"time"`  
  - `"date"`  
  - `"open notes"` / `"read notes"`  
  - `"stop"` / `"exit"` / `"quit"`  

---

## ⚙️ Requirements

Install dependencies before running:

```bash
pip install opencv-python pyttsx3 easyocr SpeechRecognition ultralytics textblob transformers pillow
