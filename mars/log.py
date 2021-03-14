
import sys


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

    def logo(self):
        print("")
        print(r"    -------------------------------------------")
        print(r"                                               ")
        print(r"        \\\\\\\\\      /\     |\\\\\ /\\\\\    ")
        print(r"        ||  ||  \\    //\\    ||  // ||        ")
        print(r"        ||  ||  ||   //  \\   ||\\\\ \\\\\\    ")
        print(r"        ||  ||  ||  //\\\\\\  ||  ||     ||    ")
        print(r"        ||  ||  || //      \\ ||  || \\\\\/    ")
        print(r"                                               ")
        print(r"    -------------------------------------------")
        sys.stdout.flush()
        return

    def options(self):
        print("    Problem settings:")
        print(f"        - Name: {self.p['Name']}")
        print(f"        - Dimensions: {self.p['Dimensions']}")
        print(f"        - Max time: {self.p['max time']}")
        print(f"        - CFL: {self.p['cfl']}")
        print(f"        - Resolution: {self.p['resolution']}")
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

        if self.iteration > 0:
            string += f", ({timing.Mcell:.3f} Mcell/s, {timing.step:.3f} s/n)"

        print(string)
        sys.stdout.flush()

        self.iteration += 1
        return

    def end(self, timing):

        print("")
        print(f"    Simulation {self.p['Name']} complete...")
        print("")
        print("    Timings:")
        print(f"    Total simulation: {timing.total_sim:.3f} s")
        print(f"    Boundaries:       {timing.total_boundary:.3f} s ({100.0*timing.total_boundary/timing.total_sim:.1f} %)")
        print(f"    IO:               {timing.total_io:.3f} s ({100.0*timing.total_io/timing.total_sim:.1f} %)")
        print("")
        print(f"    Average performance: {timing.Mcell_av/self.iteration:.3f} Mcell/s")
        print(f"    Average time per iteration: {timing.step_av/self.iteration:.3f} s")
        print("")
        sys.stdout.flush()
        return
