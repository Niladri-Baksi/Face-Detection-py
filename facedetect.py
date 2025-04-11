import cv2 #imports cv2

# video_cap=cv2.VideoCapture(0) #enables the camera in runtime

# while True: #keeps camera on until a certain key is pressed
#     ret,video_data=video_cap.read() #reads video
#     cv2.imshow("video_live",video_data)
#     if cv2.waitKey(10)==ord(";"):
#         break

# video_cap.release()
# cv2.destroyAllWindows()

face_cap=cv2.CascadeClassifier("C:/Users/nilad/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap=cv2.VideoCapture(0) #enables the camera in runtime
address="https://192.168.0.101:8080/video"
video_cap.open(address)

while True: #keeps camera on until a certain key is pressed
    ret,video_data=video_cap.read() #reads video
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    faces=face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_data)
    if cv2.waitKey(10)==ord(";"):
        break

video_cap.release()
cv2.destroyAllWindows()
