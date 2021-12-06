from perform.rom.ml_library.ml_library import MLLibrary


class PyTorchLibrary(MLLibrary):
    """Class for accessing PyTorch functionalities."""

    def __init__(self, rom_domain):
        raise ValueError("PyTorch MLLibrary not implemented yet")
        super().__init__(rom_domain)
