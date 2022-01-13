import time
from multiprocessing import Pipe, Queue, Process

def f(q, num):
    x_sq = num * num
    time.sleep(1)
    q.put_nowait(x_sq)

if __name__ == "__main__":
    # Simulates 2 processes needing different amounts of time to complete via 
    # multiprocessing and queues
    list_of_numbers = list(range(0, 100))
    x_sq = 0
    parent_conn, child_conn = Pipe()
    q = Queue()
    p = None
    total_time_start = time.perf_counter()
    for x in list_of_numbers:
        loop_start = time.perf_counter()
        if not q.qsize():
            if p is None:
                p = Process(target=f, args=(q,x))
                p.start()
        print("X: {}, X2: {}".format(x, x_sq))
        #print(q.qsize())
        time.sleep(0.1)
        if not q.empty():
            x_sq = q.get_nowait()
            p.join()
            p = None

        loop_end = time.perf_counter()
        print("Iter Time: {}".format(loop_end - loop_start))
    total_time_end = time.perf_counter()
    print("Full Time: {}".format(total_time_end - total_time_start))
    #p.join()