import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

# Initialize the webcam
# '1' means the third camera connected to the computer, usually 0 refers to the built-in webcam
cap = cv2.VideoCapture(0)

# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()

    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    # If draw=True, it will draw rectangles around faces in the image.
    # If draw=False, it will only detect faces without drawing.
    img, bboxs = detector.findFaces(img, draw=True)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        # Since multiple faces can be detected, we loop through each face's bounding box data
        for bbox in bboxs:
            # bbox is a dictionary that contains details about a detected face.
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            # The "center" key stores the center of the detected face as (cx, cy).
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # ---- Draw Data  ---- #
            # This draws a circle at the detected face's center.
            # cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            # cv2.putText(img, f'{score}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cvzone.cornerRect(img, (x, y, w, h))

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)
