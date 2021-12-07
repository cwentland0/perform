class RomMethod:
    """Base class for ROM methods."""

    def __init__(self, sol_domain, rom_domain):

        rom_dict = rom_domain.rom_dict

        # Should have as many centering and normalization profiles as there are models
        try:
            assert len(rom_dict["cent_profs"]) == rom_domain.num_models
            assert len(rom_dict["norm_fac_profs"]) == rom_domain.num_models
            assert len(rom_dict["norm_sub_profs"]) == rom_domain.num_models
        except AssertionError:
            raise AssertionError("Feature scaling profiles must have as many files as there are models")
