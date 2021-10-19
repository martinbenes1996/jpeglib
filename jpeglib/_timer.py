
import logging
import time

class Timer:
    def __init__(self, message, *args):
        self.message = message
        self._args = args
        self.end = None
        self.start = time.time()
        
    def __del__(self):
        if self.end is None:
            self.stop()
    def stop(self, f = logging.debug):
        self.end = time.time()
        self.time = self.end - self.start
        f(f'{self.message % self._args} took {self.time} s')
        return self.time