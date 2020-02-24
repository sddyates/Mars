
from datetime import datetime


class Timer:

    def __init__(self, parameter):

        self.active = parameter['profiling']

        if self.active:

            if parameter['Dimensions'] == '1D':
                self._resolution = parameter['resolution x1']
            elif parameter['Dimensions'] == '2D':
                self._resolution = parameter['resolution x1']\
                    *parameter['resolution x2']
            else:
                self._resolution = parameter['resolution x1']\
                    *parameter['resolution x2']\
                    *parameter['resolution x3']

            self.total_sim = 0.0
            self.total_io = 0.0
            self.Mcell_av = 0.0
            self.step_av = 0.0
            self.total_step = 0.0
            self.total_space_loop = 0.0
            self.total_reconstruction = 0.0
            self.total_riemann = 0.0

        else:
            None

    def start_simulation(self):
        self._start_simulation = self.start()
        return

    def stop_simulation(self):
        diff = self.stop(self._start_simulation)
        self.total_sim += diff
        return

    def start_io(self):
        self._start_io = self.start()
        return

    def stop_io(self):
        diff = self.stop(self._start_io)
        self.total_io += diff
        return

    def start_step(self):
        self._start_step = self.start()
        return

    def stop_step(self):
        diff = self.stop(self._start_step)
        self.step_diff = diff
        self.Mcell = self._resolution*1.0e-6/diff
        self.Mcell_av += self.Mcell
        self.step_av += diff
        return

    def start_space_loop(self):
        self._start_space_loop = self.start()
        return

    def stop_space_loop(self):
        diff = self.stop(self._start_space_loop)
        self.total_space_loop += diff
        return

    def start_reconstruction(self):
        self._start_reconstruction = self.start()
        return

    def stop_reconstruction(self):
        diff = self.stop(self._start_reconstruction)
        self.total_reconstruction += diff
        return

    def start_riemann(self):
        self._start_riemann = self.start()
        return

    def stop_riemann(self):
        diff = self.stop(self._start_riemann)
        self.total_riemann += diff
        return

    def start(self):
        if self.active:
            return self.get_time()
        else:
            return None

    def stop(self, start):
        if self.active:
            stop = self.get_time()
            diff = self.diff(start, stop)
            return diff
        else:
            return None

    def diff(self, start, stop):
        if self.active:
            return stop - start
        else:
            return None

    def get_time(self):
        return datetime.now().day*86400.0\
            + datetime.now().hour*3600.0\
            + datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6
