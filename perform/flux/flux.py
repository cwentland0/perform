class Flux:
    """Base class for any flux scheme, either viscous or inviscid.

    Child classes must implement the following member functions:
    * calc_flux()
    * calc_jacob_prim()
    """

    # NOTE: not really sure this is necessary

    def __init__(self):

        pass
