import cv2
import mediapipe as mp
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)
pTime = 0  # Previous time for FPS calculation

# Setup MediaPipe FaceMesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawspec = mpDraw.DrawingSpec(thickness=1,circle_radius=2);

while True:
    success, img = cap.read()  # Read frame from webcam
    if not success:
        break  # If frame reading fails, exit

    img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format

    # Process the image to get face landmarks
    results = facemesh.process(imgRGB)

    # If face landmarks are found, draw them
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS,drawspec,drawspec)  # Drawing the face mesh
            for id,lm in enumerate(face_landmarks.landmark):
                #print(lm);    
                ih,iw,ic = img.shape;
                x,y = int(lm.x*iw),int(lm.y*ih);
                print(x,y,id);
    # Calculate FPS (frames per second)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with face mesh
    cv2.imshow("Iage", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
