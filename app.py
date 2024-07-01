import cv2
import streamlit as st
from ultralytics import YOLO

# Load the YOLOv8 model (assuming 'best.pt' is in the same directory)
model = YOLO('best.pt')

# Define the class names (adjust according to your model's classes)
class_names = [
    'Hello', 'I Love You', 'One Second', 'Peace', 
    'Thumbs down', 'Thumbs up', 'Time Out', 'Hope', 'Ok'
]

# Streamlit app
st.title("Sign Language Translation")
st.write("This app uses a YOLOv8 model to translate sign language in real-time.")

# Create a placeholder for the video stream
video_placeholder = st.empty()

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize stop button state
if 'stop' not in st.session_state:
    st.session_state.stop = False

def stop_stream():
    st.session_state.stop = True

# Create a stop button
st.button('Stop', on_click=stop_stream)

# Stream the webcam feed
while cap.isOpened() and not st.session_state.stop:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image.")
        break
    
    # Run YOLO model on the frame
    results = model(frame)
    
    # Parse predictions and draw bounding boxes/labels
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].int().tolist()
        conf = result.conf.item()
        cls = int(result.cls.item())
        label = class_names[cls]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        st.write('It means: ',label)

    # Convert BGR to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame in Streamlit app
    video_placeholder.image(frame, channels="RGB")

# Release resources
cap.release()
cv2.destroyAllWindows()
