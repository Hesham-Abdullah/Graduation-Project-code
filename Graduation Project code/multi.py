import time
import multiprocessing


def do_something(seconds):
    print(f'Sleeping for {seconds} sec\n')
    time.sleep(seconds)
    print('Done Sleeping\n')


if __name__ == "__main__":
    
    start = time.perf_counter()
    processes = multiprocessing.JoinableQueue()


    for i in range(10):
        p = multiprocessing.Process(target=do_something, args=(10-i,))
        p.start()
        processes.put(p)

    processes.get()

    finish = time.perf_counter()

    print(f'time = {round((finish - start), 2)}')
    