

class RomTimeStepper():
    """Base class for ROM time steppers.
    
    Child classes provide utilities for advancing the solution forward in time.
    
    This is provided separately as there are many nuances in how ROMs advance the solutions.
    E.g. both projection-based ROMs and OpInf require a numerical time integrator, discrete Koopman and ANN time steppers
    require a recurrent time stepper, and continuous Koopman requires continuous time advancing
    """

    def __init__(self):
        pass