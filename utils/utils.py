"""
"""
import time

class BestRecoder(object):
    """
    """
    def __init__(self, init=0, descending=False):
        """
        """
        self.best = init
        self.descending = descending

    def update(self, data):
        update_flag = (data > self.best) if not self.descending else (data < self.best)
        if update_flag:
            self.best = data


class Timer(object):
    """
    return the seconds from 1970, type, float
    """
    def __init__(self):
        self.cur_time = time.time()

    def reset(self):
        self.cur_time = time.time()

    def elapse(self):
        cur_time = time.time()
        delta_t = cur_time - self.cur_time
        self.cur_time = cur_time
        return delta_t
        