import torch
import cv2

# Load the model (path to your trained weights)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.25  # confidence threshold (adjust as needed)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 expects RGB
    results = model(frame[..., ::-1])  # convert BGR to RGB

    # Render predictions
    results.render()  # updates results.imgs with boxes and labels

    # Show frame
    cv2.imshow("Traffic Sign Recognition", results.ims[0][..., ::-1])  # convert back to BGR

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
