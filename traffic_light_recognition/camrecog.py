import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("./models/model.h5") # model directory

print("Model input shape:", model.input_shape) # model input size is 64, 64, 3

cap = cv2.VideoCapture(0) # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    resized_frame = cv2.resize(frame, (64, 64))

    normalized_frame = resized_frame.astype(np.float32) / 255.0

    input_data = np.expand_dims(normalized_frame, axis=0)

    predictions = model.predict(input_data)

    predicted_class = np.argmax(predictions)

    cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
