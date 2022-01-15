import cv2
import time
import multiprocessing
from multiprocessing import Pipe, Queue, Process
#from frame import get_necessary_models, get_center_of_play
from PlayerDetection.frame import get_necessary_models, get_center_of_play

THRESHOLD_STEP = 6

def branch_center_of_play(q_frame, q_coords, autoencoder, yolo_model, show_results):
    run_loop = True
    while(run_loop):
        if not q_frame.empty():
            frame = q_frame.get_nowait()
            time_start = time.perf_counter()
            x, y = get_center_of_play(frame, autoencoder, yolo_model, show_results)
            #print("Time to Analyze: {}".format(time.perf_counter() - time_start))
            q_coords.put_nowait((x, y))

def analyze_video(video_path):

    autoencoder, yolo_model = get_necessary_models()
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter('Media\\Tracking_Output\\41_Sec_Output.mp4', -1, 30.0, (1920,1080))
    if (cap.isOpened() == False):
        print("Error Opening Video File")

    q_frame = Queue()
    q_coords = Queue()
    p = None
    x_ref, y_ref = 0, 0
    x_ref_prev, y_ref_prev = x_ref, y_ref
    x_cam, y_cam = 0, 0
    delta_x = 0
    frame_num = 0
    frame_gap = 0

    camera_initialized = False

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    if p is None:
        p = Process(target=branch_center_of_play, args=(q_frame, q_coords, autoencoder, yolo_model, False))
        p.start()

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_copy = frame

        if not camera_initialized:
            x_cam = int(frame.shape[1]/2)
            y_cam = int(frame.shape[0]/2)
            #print(x_cam, y_cam)
            camera_initialized = True
        
        if q_frame.empty():
            q_frame.put_nowait(frame_copy)
        
        if ret == True:
            cv2.circle(frame,(x_ref, y_ref), 8, (0, 0, 255), -1)
            cv2.circle(frame,(x_cam, y_cam), 8, (255, 0, 0), -1)
            out.write(frame)
            cv2.resizeWindow('Frame', 1280, 720)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                q_coords.close()
                q_coords.join_thread()
                q_frame.close()
                q_frame.join_thread()
                p.terminate()
                break
        else:
            q_coords.close()
            q_coords.join_thread()
            q_frame.close()
            q_frame.join_thread()
            p.terminate()
            break
        
        #print("{}. Reference X: {}, Reference Y: {}, Delta Value: {}".format(frame_num, x_ref, y_ref, delta_x))

        #Obtaining new Reference coordinates
        if not q_coords.empty():
            time_start = time.perf_counter()
            x_ref_prev, y_ref_prev = x_ref, y_ref
            x_ref, y_ref = q_coords.get_nowait()
            if x_ref == -1:
                x_ref = x_ref_prev
            if y_ref == -1:
                y_ref = y_ref_prev
            time_end = time.perf_counter()
            frame_gap = frame_num - frame_gap

            #Calculate change to be applied
            delta_x = int(2.5 * (x_ref - x_cam) / frame_gap) #Linear

            #Dampening Function
            if abs(delta_x) > THRESHOLD_STEP:
                delta_x = int(delta_x / 3)

            
        #Apply change to camera point
        x_cam = x_cam + delta_x

        #Incrementing frame number
        frame_num+=1
    
    p = None
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Performing Tracking and Movement")
    video_path = "Media\\Video_Samples\\41s_Sample.mp4"
    analyze_video(video_path)