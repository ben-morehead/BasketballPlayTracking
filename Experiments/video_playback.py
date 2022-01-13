import cv2

if __name__ == "__main__":
    # Functionality for reading a video feed and playing it back
    cap = cv2.VideoCapture("Media\\Video_Samples\\unrelated_sample.mp4")

    if (cap.isOpened() == False):
        print("Error Opening Video File")
        
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
        
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()