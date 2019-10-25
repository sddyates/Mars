
import datetime as dt
class Timer:

    def __init__(self, active=False):

        if active:
            self.start_timer = self.start()
        else:
            None

    def start(self):
        return dt.now()

    def stop(self):
        return dt.now()

    def diff(self, start, stop):
        return stop - start
