import cv2
import dlib

# Initialize the dlib face detector
detector = dlib.get_frontal_face_detector()

# Connect to the default camera (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frameq
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check the data type of the grayscale image
    if gray.dtype != 'uint8':
        print(f"Unexpected image data type: {gray.dtype}")
        break

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Draw rectangles around detected faces and count them
    for i, face in enumerate(faces):
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, f'Face {i+1}', (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
