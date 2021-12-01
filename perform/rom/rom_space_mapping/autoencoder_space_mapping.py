from perform.rom.rom_space_mapping.rom_space_mapping import RomSpaceMapping


class AutoencoderSpaceMapping(RomSpaceMapping):
    """Autoencoder mapping to/from the state/latent spaces"""

    def __init__(self):

        super().__init__()
