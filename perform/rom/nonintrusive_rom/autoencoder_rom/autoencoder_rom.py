from perform.rom.nonintrusive_rom.nonintrusive_rom import NonIntrusiveRom


class AutoencoderRom(NonIntrusiveRom):
    """Base class for all autoencoder-based ROMs.

    This class provides functionality for all non-intrusive ROMs which are constructed around autoencoders.
    Child classes must provide a means for mapping from the full state space to the latent space, and for mapping from
    the latent space back to the full state space.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)

    def decode_sol():
        """
        """

    def init_from_sol():
        """
        """
