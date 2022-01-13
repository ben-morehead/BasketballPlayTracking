import cv2
from frame import get_necessary_models, get_center_of_play

def analyze_video(video_path):
    autoencoder, yolo_model = get_necessary_models()

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        print("Error Opening Video File")
        
    while(cap.isOpened()):
        ret, frame = cap.read()
        x, y = get_center_of_play(frame, autoencoder, yolo_model, show_results=False)
        print(frame.shape)
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Performing Tracking and ")
    video_path = "Media\\Video_Samples\\10s_Sample.mp4"
    analyze_video(video_path)