import cv2

video=cv2.VideoCapture("Sample.mp4")

while True:
    ret,frame=video.read()
    if not ret:
        break
    cv2.rectangle(frame,(100,100),(400,300),(0,255,0),2)
    cv2.putText(frame,"Hello",(100,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
    cv2.imshow("Video",frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()

