import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Define the class names (adjust according to your model's classes)
class_names = [ 'Hello'
, 'I Love You'
, 'One Second'
, 'Peace'
, 'Thumbs down'
, 'Thumbs up'
, 'Time Out'
, 'hope'
, 'ok']

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Get the width and height of the frame
    height, width, _ = frame.shape

    # Use the YOLO model to make predictions
    results = model(frame)
    
    # Parse predictions
    for result in results[0].boxes:  # access the boxes attribute
        x1, y1, x2, y2 = result.xyxy[0].int().tolist()
        conf = result.conf.item()
        cls = int(result.cls.item())
        label = class_names[cls]
        
        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Sign Language Translation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
