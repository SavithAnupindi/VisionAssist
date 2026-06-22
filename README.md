An AI-powered assistive vision system for visually impaired users, built on a 
Raspberry Pi + Laptop split architecture.

Raspberry Pi 3B+ (camera + streaming)
│
│  UDP stream (JPEG frames)
▼
Laptop (detection + display)
├── YOLOv8m — object detection
├── ByteTrack — multi-object tracking
├── Threat scoring engine
├── OCR (pytesseract)
├── TTS audio alerts (pyttsx3)
└── Tkinter GUI

Features
- Real-time object detection with distance estimation
- Kalman-smoothed distance and approach rate calculation
- Time-to-collision (TTC) alerts
- Adaptive threat threshold based on scene density
- Occlusion detection
- Text recognition (OCR) for signs and labels
- Priority-based audio alerts
