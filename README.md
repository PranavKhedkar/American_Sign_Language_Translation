## Problem Statement

This project aims to translate American Sign Language (ASL) gestures into English text in real-time. By detecting and interpreting hand gestures, this application bridges the communication gap for individuals who use ASL.

## Technologies Used

- **YOLOv8**: Used for real-time object detection and classification of hand gestures.
- **cv2**: OpenCV library for capturing and processing video feed from the webcam.
- **Streamlit**: Web application framework for creating the user interface.

## File Structure
```bash
├── app.py                   # Streamlit application
├── best.pt                  # Trained model
├── main.py                  # Main application logic
├── requirements.txt         # Dependencies
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

## Working

1. Video Capture: The cv2 library captures the video feed from the webcam.
2. Gesture Detection: The captured video frames are fed into the YOLOv8 model, which detects and identifies the hand gestures.
3. Translation: The identified gestures are then mapped to their corresponding ASL signs.
4. Display: The translated English text is displayed in a Streamlit web application, providing an interactive and user-friendly interface.

## Output:

![Screenshot 2024-07-01 131948](https://github.com/PranavKhedkar/American_Sign_Language_Translation/assets/99120112/bcd4ad12-eca0-4af2-a038-9825f5eb15c0)

## Future Scope:

Currently, the model can identify nine hand signs: '**Hello**', '**I Love You**', '**One Second**', '**Peace**', '**Thumbs down**', '**Thumbs up**', '**Time Out**', '**Hope**' and '**Ok**'. In future, more signs can be added for making the model a more powerful tool for communication, accessible to a broader range of users and applications.
