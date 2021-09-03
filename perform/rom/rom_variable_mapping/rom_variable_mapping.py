

class RomVariableMapping():
    """Base class for mapping ROM target variables to primitive/conservative.
    
    For ROMs targetting primitive or conservative variables, this is trivial.
    
    Others may require analytical expressions which utilize gas model functions e.g. lifting to the 
    set [p, u, xi, c_0, c_1, ...] where xi is specific volume and c is molar concentration.
    """

    def __init__(self, rom_domain):
        
        # check model_var_idxs
        model_var_sum = 0
        for model_idx in range(rom_domain.num_models):
            model_var_sum += len(rom_domain.model_var_idxs[model_idx])
            for model_var_idx in rom_domain.model_var_idxs[model_idx]:
                assert model_var_idx >= 0, "model_var_idxs must be non-negative integers"
                assert (
                    model_var_idx < self.num_vars
                ), "model_var_idxs must less than the number of governing equations"
        assert model_var_sum == self.num_vars, (
            "Must specify as many model_var_idxs entries as governing equations ("
            + str(model_var_sum)
            + " != "
            + str(self.num_vars)
            + ")"
        )
        model_var_idxs_one_list = sum(rom_domain.model_var_idxs, [])
        assert len(model_var_idxs_one_list) == len(
            set(model_var_idxs_one_list)
        ), "All entries in model_var_idxs must be unique"
        self.model_var_idxs = rom_domain.model_var_idxs
