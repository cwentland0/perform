import os

from perform.constants import FD_STEP_DEFAULT
from perform.rom.projectionROM.projectionROM import ProjectionROM
from perform.inputFuncs import catch_input


class AutoencoderProjROM(ProjectionROM):
	"""
	Base class for any non-linear manifold ROM using autoencoders

	Child classes simply supply library-dependent functions
	(e.g. for TensorFlow, PyTorch)
	"""

	def __init__(self, model_idx, rom_domain, solver, sol_domain):

		super().__init__(model_idx, rom_domain, solver, sol_domain)

		rom_dict = rom_domain.rom_dict

		# Load decoder
		decoder_path = os.path.join(rom_domain.model_dir,
									rom_domain.model_files[model_idx])
		assert(os.path.isfile(decoder_path)), \
			"Invalid decoder file path"
		self.decoder = self.load_model_obj(decoder_path)
		self.decoder_io_dtypes = self.check_model(decoder=True)

		# If required, load encoder
		# Encoder is required for encoder Jacobian
		# 	or initializing from projection of full ICs
		self.encoder_jacob = catch_input(rom_dict, "encoder_jacob", False)
		self.encoder = None
		if (self.encoder_jacob
				or (not rom_domain.init_rom_from_file[model_idx])):

			encoder_files = rom_dict["encoder_files"]
			assert (len(encoder_files) == rom_domain.num_models), \
				"Must provide encoder_files for each model"
			encoder_path = os.path.join(rom_domain.model_dir,
										encoder_files[model_idx])
			assert (os.path.isfile(encoder_path)), \
				"Could not find encoder file at " + encoder_path
			self.encoder = self.load_model_obj(encoder_path)
			self.encoder_io_dtypes = self.check_model(decoder=False)

		# numerical Jacobian params
		self.numerical_jacob = catch_input(rom_dict, "numerical_jacob", False)
		self.fd_step = catch_input(rom_dict, "fd_step", FD_STEP_DEFAULT)

	def encode_sol(self, solIn):
		"""
		Compute encoding of full-dimensional state,
		including centering and normalization
		"""

		if self.target_cons:
			sol = \
				self.standardize_data(
					solIn,
					normalize=True,
					norm_fac_prof=self.norm_fac_prof_cons,
					norm_sub_prof=self.norm_sub_prof_cons,
					center=True, cent_prof=self.cent_prof_cons,
					inverse=False
				)

		else:
			sol = \
				self.standardize_data(
					solIn,
					normalize=True,
					norm_fac_prof=self.norm_fac_prof_prim,
					norm_sub_prof=self.norm_sub_prof_prim,
					center=True, cent_prof=self.cent_prof_prim,
					inverse=False
				)

		code = self.apply_encoder(sol)

		return code

	def init_from_sol(self, sol_domain):
		"""
		Initialize full-order solution from projection of
		loaded full-order initial conditions
		"""

		sol_int = sol_domain.sol_int

		if self.target_cons:
			self.code = self.encode_sol(sol_int.sol_cons[self.var_idxs, :])
			sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)
		else:
			self.code = self.encode_sol(sol_int.sol_prim[self.var_idxs, :])
			sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)
