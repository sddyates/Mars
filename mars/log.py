
import datetime as dt


class Log:

    def __init__(self, p):

        self.p = p
        self.iteration = 0

        if self.p['Dimensions'] == '1D':
            self.resolution = f"{self.p['resolution x1']}"
        elif self.p['Dimensions'] == '2D':
            self.resolution = f"{self.p['resolution x1']}"\
                + f"x{self.p['resolution x2']}"
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


    def step(self, g, timing):

        percent = g.t*100.0/self.p['max time']

        string = f"    n = {self.iteration}, t = {g.t:.2e}, dt = {g.dt:.2e}, {percent:.1f}%"

        if timing.active & (self.iteration > 0):
            string += f", {timing.Mcell_av/self.iteration:.3f} Mcell/s ({timing.step_av/self.iteration:.3f} s,"

        if self.iteration > 0:
            total_time_at_current_dt = g.t_max/g.dt*timing.step_av/self.iteration
            time_done = timing.get_time() - timing._start_sim

            string += f" {total_time_at_current_dt - time_done:.3f} s)"

        print(string)

        self.iteration += 1
        return


    def end(self, timing):
        print("")
        print(f"    Simulation {self.p['Name']} complete...")
        print("")
        print(f"    Total simulation time: {timing.total_sim:.3f} s")
        print(f"    Total io time: {timing.total_io:.3f} s ({100.0*timing.total_io/timing.total_sim:.1f}) %")
        print(f"    Total space integration time: {timing.total_space_loop:.3f} s ({100.0*timing.total_space_loop/timing.total_sim:.1f}) %")
        print(f"    Total reconstruction time: {timing.total_reconstruction:.3f} s ({100.0*timing.total_reconstruction/timing.total_sim:.1f}) %")
        print(f"    Total Riemann time: {timing.total_riemann:.3f} s ({100.0*timing.total_riemann/timing.total_sim:.1f}) %")
        print("")
        print(f"    Average performance: {timing.Mcell_av/self.iteration:.3f} Mcell/s")
        print(f"    Average time per iteration: {timing.step_av/self.iteration:.3f} s")
        print("")
        return
