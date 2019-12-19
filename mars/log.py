
import datetime as dt


class Log:

    def __init__(self, parameter):

        self.iteration = 0

        if parameter['Dimensions'] == '1D':
            self.resolution = f"{parameter['resolution x1']}"
        elif parameter['Dimensions'] == '2D':
            self.resolution = f"{parameter['resolution x1']}"\
                + f"x{parameter['resolution x2']}"
        else:
            self.resolution = f"{parameter['resolution x1']}"\
                + f"x{parameter['resolution x2']}"\
                + f"x{parameter['resolution x3']}"

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
        print(f"        - Name: {parameter['Name']}")
        print(f"        - Dimensions: {parameter['Dimensions']}")
        print(f"        - Max time: {parameter['max time']}")
        print(f"        - CFL: {parameter['cfl']}")
        print(f"        - Resolution: " + self.resolution)
        print(f"        - Riemann: {parameter['riemann']}")
        print(f"        - Reconstruction: {parameter['reconstruction']}")
        print(f"        - Limiter: {parameter['limiter']}")
        print(f"        - Time stepping: {parameter['time stepping']}")
        print(f"        - Physics: {parameter['method']}")
        print(f"        - Gamma: {parameter['gamma']}")
        print("")
        return


    def begin(self):
        return print("    Starting time integration loop...")


    def step(self, g, timer):

        percent = grid.t*100.0/parameter['max time']

        string = f"    n = {self.iteration}, t = {grid.t:.2e}, dt = {grid.dt:.2e}, {percent:.1f}%"

        if timer.active & (self.iteration > 0):
            string += f", {timer.Mcell_av/self.iteration:.3f} Mcell/s ({timer.step_av/self.iteration:.3f} s,"

        if self.iteration > 0:
            total_time_at_current_dt = gridt_max/griddt*timer.step_av/self.iteration
            time_done = timer.get_time() - timer._start_sim

            string += f" {total_time_at_current_dt - time_done:.3f} s)"

        print(string)

        self.iteration += 1
        return


    def end(self, timer):
        print("")
        print(f"    Simulation {parameter['Name']} complete...")
        print("")
        print(f"    Total simulation time: {timer.total_sim:.3f} s")
        print(f"    Total io time: {timer.total_io:.3f} s ({100.0*timer.total_io/timer.total_sim:.1f}) %")
        print(f"    Total space integration time: {timer.total_space_loop:.3f} s ({100.0*timer.total_space_loop/timer.total_sim:.1f}) %")
        print(f"    Total reconstruction time: {timer.total_reconstruction:.3f} s ({100.0*timer.total_reconstruction/timer.total_sim:.1f}) %")
        print(f"    Total Riemann time: {timer.total_riemann:.3f} s ({100.0*timer.total_riemann/timer.total_sim:.1f}) %")
        print("")
        print(f"    Average performance: {timer.Mcell_av/self.iteration:.3f} Mcell/s")
        print(f"    Average time per iteration: {timer.step_av/self.iteration:.3f} s")
        print("")
        return
