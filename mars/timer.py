
import numpy as np
from datetime import datetime


class Timer:
    """
    Synopsis
    --------
    Timer class for timings of fuction calls and related.

    Args
    ----
    p: dic-like
    Dictionary of user defined ps, e.g.
    maximum simulation time.

    Methods
    -------
    Various

    TODO
    ----
    None
    """

    def __init__(self, p):

        self.active = p['profiling']

        if self.active:

            self._resolution = np.prod(p['resolution'])

            self._resolution *= 1.0e-6

            self.total_sim = 0.0
            self.total_io = 0.0
            self.Mcell = 0.0
            self.Mcell_av = 0.0
            self.step = 0.0
            self.step_av = 0.0
            self.total_step = 0.0
            self.total_boundary = 0.0
            # self.total_space_loop = 0.0
            # self.total_reconstruction = 0.0
            # self.total_riemann = 0.0

        else:
            None


    def start_sim(self):
        if self.active:
            self._start_sim = self.start()
        return


    def stop_sim(self):
        if self.active:
            diff = self.stop(self._start_sim)
            self.total_sim += diff
        return


    def start_io(self):
        if self.active:
            self._start_io = self.start()
        return


    def stop_io(self):
        if self.active:
            diff = self.stop(self._start_io)
            self.total_io += diff
        return


    def start_boundary(self):
        if self.active:
            self._start_boundary = self.start()
        return


    def stop_boundary(self):
        if self.active:
            diff = self.stop(self._start_boundary)
            self.total_boundary += diff
        return


    def start_step(self):
        if self.active:
            self._start_step = self.start()
        return


    def stop_step(self):
        if self.active:
            diff = self.stop(self._start_step)
            self.step_diff = diff
            self.Mcell = self._resolution/diff
            self.Mcell_av += self.Mcell
            self.step = diff
            self.step_av += diff
        return


    # def start_space_loop(self):
    #     if self.active:
    #         self._start_space_loop = self.start()
    #     return
    #
    #
    # def stop_space_loop(self):
    #     if self.active:
    #         diff = self.stop(self._start_space_loop)
    #         self.total_space_loop += diff
    #     return
    #
    #
    # def start_reconstruction(self):
    #     if self.active:
    #         self._start_reconstruction = self.start()
    #     return
    #
    #
    # def stop_reconstruction(self):
    #     if self.active:
    #         diff = self.stop(self._start_reconstruction)
    #         self.total_reconstruction += diff
    #     return
    #
    #
    # def start_riemann(self):
    #     if self.active:
    #         self._start_riemann = self.start()
    #     return
    #
    #
    # def stop_riemann(self):
    #     if self.active:
    #         diff = self.stop(self._start_riemann)
    #         self.total_riemann += diff
    #     return


    def start(self):
        return self.get_time()


    def stop(self, start):
        stop = self.get_time()
        diff = self.diff(start, stop)
        return diff


    def diff(self, start, stop):
        return stop - start


    def get_time(self):
        return datetime.now().day*86400.0\
            + datetime.now().hour*3600.0\
            + datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6
