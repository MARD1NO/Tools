import os 
import psutil 

"""
Usage: 

Call this function in your python script, and you can get the current memory info. 

"""

def get_current_memory_gb(info): 
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()

    print("{} ===== Now Memory use: {} =====".format(info, info.rss / 1024 / 1024 / 1024))


