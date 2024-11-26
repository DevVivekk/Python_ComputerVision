import cv2
import time
import mediapipe as mp;
# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize previous time for FPS calculation
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75);

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    results = faceDetection.process(imgRGB);
    #print(results);
    if results.detections:
        for id,detection in enumerate(results.detections):
            #print(detection.score);
            #mpDraw.draw_detection(img,detection);
            score = int(detection.score[0] * 100)  # Convert to a percentage and round to an integer
            cv2.putText(img, f'{score:.2f}', (int(detection.location_data.relative_bounding_box.xmin * img.shape[1]), int(detection.location_data.relative_bounding_box.ymin * img.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            bboxC = detection.location_data.relative_bounding_box;
            ih,iw,ic = img.shape;
            bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
            int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),1);
    if not success:
        break
    
    # Get the current time
    cTime = time.time()
    
    # Calculate FPS
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    
    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow("Image", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
