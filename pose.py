import cv2
import mediapipe as mp
import time

# Initialize video capture and Mediapipe Pose solution
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
# Timing variables
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Convert image to RGB for Mediapipe processing
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRgb)

    # Optional: Draw Pose landmarks if needed
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS);
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            #print(id,lm);
            cx,cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED);  

    #print(results.pose_landmarks);
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show the image with FPS
    cv2.imshow("Image", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
