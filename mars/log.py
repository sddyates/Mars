
import datetime as dt


class Log:

    def __init__(self, p):

        self.p = p

        if self.p['Dimensions'] == '1D':
            self.resolution = f"{self.p['resolution x1']}"
        elif self.p['Dimensions'] == '2D':
            self.resolution = f"{self.p['resolution x1']}"\
                + "x{self.p['resolution x2']}"
        else:
            self.resolution = f"{self.p['resolution x1']}"\
                + f"x{self.p['resolution x2']}"\
                + f"x{self.p['resolution x3']}"

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
        return



    def begin(self):
        return print("    Starting time integration loop...")

    def step(self, i, t, dt, timing):
        percent = t*100.0/self.p['max time']
        string = f"    n = {i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}%"
        if timing.active:
            string += f", {timing.Mcell:.3f} Mcell/s ({timing.step_diff:.3f} s)"

        return print(string)

    def end(self, i, timing):
        print("")
        print(f"    Simulation {self.p['Name']} complete...")
        print("")
        print(f"    Total simulation time: {timing.total_sim:.3f} s")
        print(f"    Total space integration time: {timing.total_reconstruction:.3f} s")
        print(f"    Total reconstruction time: {timing.total_reconstruction:.3f} s")
        print(f"    Total Riemann time: {timing.total_riemann:.3f} s")
        print("")
        print(f"    Average performance: {timing.Mcell_av/i:.3f} Mcell/s")
        print(f"    Average time per iteration: {timing.step_av/i:.3f} s")
        print("")
        return
