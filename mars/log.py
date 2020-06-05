
import sys

import datetime as dt

class Log:
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

        self.p = p
        self.iteration = 0

        if self.p['Dimensions'] == '1D':
            self.resolution = f"{self.p['resolution x1']}"
        elif self.p['Dimensions'] == '2D':
            self.resolution = f"{self.p['resolution x1']}"\
                + f" x {self.p['resolution x2']}"
        else:
            self.resolution = f"{self.p['resolution x1']}"\
                + f" x {self.p['resolution x2']}"\
                + f" x {self.p['resolution x3']}"


    def logo(self):
        print("")
        print(r"    -----------------------------------------------")
        print(r"                                                   ")
        print(r"        \\\\\\\\\      /\     |\\\\\ /\\\\\        ")
        print(r"        ||  ||  \\    //\\    ||  // ||            ")
        print(r"        ||  ||  ||   //  \\   ||\\\\ \\\\\\        ")
        print(r"        ||  ||  ||  //\\\\\\  ||  ||     ||        ")
        print(r"        ||  ||  || //      \\ ||  || \\\\\/ 0.2    ")
        print(r"                                                   ")
        print(r"    -----------------------------------------------")
        sys.stdout.flush()
        return


    def options(self):
        print(f"    Problem settings:")
        print(f"        - Name: {self.p['Name']}")
        print(f"        - Dimensions: {self.p['Dimensions']}")
        print(f"        - Max time: {self.p['max time']}")
        print(f"        - CFL: {self.p['cfl']}")
        print(f"        - Resolution: " + self.resolution)
        print(f"        - Riemann: {self.p['riemann']}")
        print(f"        - Reconstruction: {self.p['reconstruction']}")
        print(f"        - Limiter: {self.p['limiter']}")
        print(f"        - Time stepping: {self.p['time stepping']}")
        print(f"        - Physics: {self.p['method']}")
        print(f"        - Gamma: {self.p['gamma']}")
        print("")
        sys.stdout.flush()
        return


    def begin(self):
        print("    Starting time integration loop...")
        sys.stdout.flush()
        return


    def step(self, g, timing):

        percent = g.t*100.0/self.p['max time']

        string = f"    n = {self.iteration}, t = {g.t:.2e}, dt = {g.dt:.2e}, {percent:.1f} %"

        if timing.active & (self.iteration > 0):
            string += f", ({timing.Mcell:.3f} Mcell/s, {timing.step:.3f} s/n)"

        print(string)
        sys.stdout.flush()

        self.iteration += 1
        return


    def end(self, timing):

        tot = timing.total_sim
        io = timing.total_io
        space = timing.total_space_loop
        rec = timing.total_reconstruction
        rie = timing.total_riemann

        space -= rec + rie
        other = tot - io - space - rec - rie

        print("")
        print(f"    Simulation {self.p['Name']} complete...")
        print("")
        print(f"    Timings:")
        print(f"    Total simulation:    {tot:.3f} s")
        print(f"    Spatial integration: {space:.3f} s ({100.0*space/tot:.1f} %)")
        print(f"    Riemann:             {rie:.3f} s ({100.0*rie/tot:.1f} %)")
        print(f"    Reconstruction:      {rec:.3f} s ({100.0*rec/tot:.1f} %)")
        print(f"    IO:                  {io:.3f} s ({100.0*io/tot:.1f} %)")
        print(f"    Other:               {other:.3f} s ({100.0*other/tot:.1f} %)")
        print("")
        print(f"    Average performance: {timing.Mcell_av/self.iteration:.3f} Mcell/s")
        print(f"    Average time per iteration: {timing.step_av/self.iteration:.3f} s")
        print("")
        sys.stdout.flush()
        return
