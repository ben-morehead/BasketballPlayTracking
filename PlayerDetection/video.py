import cv2
import time
import multiprocessing
from multiprocessing import Pipe, Queue, Process
from frame import get_necessary_models, get_center_of_play

def branch_center_of_play(q_frame, q_coords, q_exit, autoencoder, yolo_model, show_results):
    run_loop = True
    while(run_loop):
        if not q_frame.empty():
            frame = q_frame.get_nowait()
            time_start = time.perf_counter()
            print("Starting_Get_Center")
            x, y = get_center_of_play(frame, autoencoder, yolo_model, show_results)
            print("Time to Analyze: {}".format(time.perf_counter() - time_start))
            q_coords.put_nowait((x, y))
    
    




def analyze_video(video_path):

    autoencoder, yolo_model = get_necessary_models()

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        print("Error Opening Video File")

    q_frame = Queue()
    q_coords = Queue()
    q_exit = Queue()
    p = None
    x, y = 0, 0
    frame_num = 0

    if p is None:
        p = Process(target=branch_center_of_play, args=(q_frame, q_coords, q_exit, autoencoder, yolo_model, False))
        time_start = time.perf_counter()
        p.start()
        time_end = time.perf_counter()

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_copy = frame
        
        if q_frame.empty():
            q_frame.put_nowait(frame_copy)
            
        #x, y = get_center_of_play(frame, autoencoder, yolo_model, show_results=False)
        #print(frame.shape)
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                q_exit.close()
                q_exit.join_thread()
                q_coords.close()
                q_coords.join_thread()
                q_frame.close()
                q_frame.join_thread()
                p.terminate()
                break
        else:
            q_exit.close()
            q_exit.join_thread()
            q_coords.close()
            q_coords.join_thread()
            q_frame.close()
            q_frame.join_thread()
            p.terminate()
            break
        
        print("{}. X: {}, Y: {}".format(frame_num, x, y))

        if not q_coords.empty():
            time_start = time.perf_counter()
            x, y = q_coords.get_nowait()
            time_end = time.perf_counter()
            print("Time to Grab: {}".format(time_end - time_start))
        frame_num+=1
    
    p = None
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Performing Tracking and ")
    video_path = "Media\\Video_Samples\\10s_Sample.mp4"
    analyze_video(video_path)