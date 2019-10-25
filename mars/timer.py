
from datetime import datetime


class Timer:

    def __init__(self, p):

        self.active = p['profiling']

        if p['Dimensions'] == '1D':
            self.resolution = p['resolution x1']
        elif p['Dimensions'] == '2D':
            self.resolution = p['resolution x1']*p['resolution x2']
        else:
            self.resolution = p['resolution x1']*p['resolution x2']*p['resolution x3']

        if self.active:
            self.Mcell_av = 0.0
            self.step_av = 0.0
            self.total_step = 0.0
            self.total_space_loop = 0.0
            self.total_reconstruction = 0.0
            self.total_riemann = 0.0

            self.start_step = self.start()
            self.stop_step, self.total_step = self.stop(self.start_step, self.total_step)
            self.Mcell = self.resolution*1.0e-6/self.stop_step
            self.Mcell_av += self.Mcell

            self.start_space_loop = self.start()
            self.stop_space_loop, self.total_space_loop = self.stop(self.start_space_loop, self.total_space_loop)

            self.start_reconstruction = self.start()
            self.stop_reconstruction, self.total_reconstruction = self.stop(self.start_reconstruction, self.total_reconstruction)

            self.start_riemann = self.start()
            self.stop_riemann, self.total_riemann = self.stop(self.start_riemann, self.total_riemann)

        else:
            None

    def start(self):
        if self.active:
            return self.get_time()
        else:
            return None

    def stop(self, start, total):
        if self.active:
            stop = self.get_time()
            diff = self.diff(start, stop)
            return diff, total+diff
        else:
            return None

    def diff(self, start, stop):
        if self.active:
            return stop - start
        else:
            return None

    def get_time(self):
        return datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6
