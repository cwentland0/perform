from perform.rom.nonintrusive_rom.autoencoder_rom.autoencoder_rom import AutoencoderRom


class AutoencoderKoopman(AutoencoderRom):
    """Base class for autoencoder ROMs which use a linear Koopman operator to advance the latent variables in time."""

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)
