import cv2
import time
import multiprocessing
from multiprocessing import Pipe, Queue, Process
from frame import get_necessary_models, get_center_of_play

def branch_center_of_play(q, frame, autoencoder, yolo_model, show_results):
    time_start = time.perf_counter()
    print("Starting_Get_Center")
    x, y = get_center_of_play(frame, autoencoder, yolo_model, show_results)
    print("Done Get Center: {}".format(time.perf_counter() - time_start))
    q.put_nowait((x, y))
    print("Additional Time: {}".format(time.perf_counter() - time_start))


def analyze_video(video_path):

    autoencoder, yolo_model = get_necessary_models()

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        print("Error Opening Video File")

    q = Queue()
    p = None
    x, y = 0, 0
    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_copy = frame
        if not q.qsize():
            if p is None:
                
                p = Process(target=branch_center_of_play, args=(q,frame_copy,autoencoder,yolo_model,False))
                time_start = time.perf_counter()
                p.start()
                time_end = time.perf_counter()
                print("Time to Make ANew: {}".format(time_end - time_start))
        #x, y = get_center_of_play(frame, autoencoder, yolo_model, show_results=False)
        #print(frame.shape)
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            break
        
        print("{}. X: {}, Y: {}".format(frame_num, x, y))

        if not q.empty():
            time_start = time.perf_counter()
            x, y = q.get_nowait()
            p.join()
            p = None
            time_end = time.perf_counter()
            print("Time to Grab and Join: {}".format(time_end - time_start))
        frame_num+=1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Performing Tracking and ")
    video_path = "Media\\Video_Samples\\10s_Sample.mp4"
    analyze_video(video_path)