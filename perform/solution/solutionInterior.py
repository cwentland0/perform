import os

import numpy as np

from perform.constants import REAL_TYPE, RES_NORM_PRIM_DEFAULT
from perform.solution.solutionPhys import SolutionPhys


class SolutionInterior(SolutionPhys):
	"""
	Solution of interior domain
	"""

	def __init__(self, gas, sol_prim_in, solver, time_int):
		super().__init__(gas, solver.mesh.num_cells, sol_prim_in=sol_prim_in)

		gas = self.gas_model
		num_cells = solver.mesh.num_cells

		self.source = np.zeros((gas.num_species, num_cells), dtype=REAL_TYPE)
		self.rhs = np.zeros((gas.num_eqs, num_cells), dtype=REAL_TYPE)

		# add bulk velocity and update state if requested
		if solver.vel_add != 0.0:
			self.sol_prim[1, :] += solver.vel_add
			self.update_state(from_cons=False)

		# initializing time history
		self.sol_hist_cons = [self.sol_cons.copy()] * (time_int.time_order + 1)
		self.sol_hist_prim = [self.sol_prim.copy()] * (time_int.time_order + 1)

		# RHS storage for multi-stage schemes
		self.rhs_hist = [self.rhs.copy()] * (time_int.time_order + 1)

		# snapshot storage matrices, store initial condition
		if solver.prim_out:
			self.prim_snap = \
				np.zeros((gas.num_eqs, num_cells, solver.num_snaps + 1), dtype=REAL_TYPE)
			self.prim_snap[:, :, 0] = self.sol_prim.copy()
		if solver.cons_out:
			self.cons_snap = \
				np.zeros((gas.num_eqs, num_cells, solver.num_snaps + 1), dtype=REAL_TYPE)
			self.cons_snap[:, :, 0] = self.sol_cons.copy()

		# these don't include the source/RHS associated with the final solution
		# TODO: calculate at final solution for DEIM stuff
		if solver.source_out:
			self.source_snap = \
				np.zeros((gas.num_species, num_cells, solver.num_snaps), dtype=REAL_TYPE)
		if solver.rhs_out:
			self.rhs_snap = \
				np.zeros((gas.num_eqs, num_cells, solver.num_snaps), dtype=REAL_TYPE)

		if (time_int.time_type == "implicit") or (solver.run_steady):
			# norm normalization constants
			# TODO: will need a normalization constant for
			# 	conservative residual when it's implemented
			if (len(solver.res_norm_prim) == 1) and (solver.res_norm_prim[0] is None):
				solver.res_norm_prim = [None] * gas.num_eqs
			else:
				assert(len(solver.res_norm_prim) == gas.num_eqs)
			for var_idx in range(gas.num_eqs):
				if (solver.res_norm_prim[var_idx] is None):
					# 0: pressure, 1: velocity, 2: temperature, >=3: species
					solver.res_norm_prim[var_idx] = RES_NORM_PRIM_DEFAULT[min(var_idx, 3)]

			# residual norm storage
			if time_int.time_type == "implicit":

				self.res = np.zeros((gas.num_eqs, num_cells), dtype=REAL_TYPE)
				self.res_norm_l2 = 0.0
				self.res_norm_l1 = 0.0
				self.res_norm_history = np.zeros((solver.num_steps, 2), dtype=REAL_TYPE)

				if (time_int.dual_time) and (time_int.adapt_dtau):
					self.srf = np.zeros(num_cells, dtype=REAL_TYPE)

				# CSR matrix indices
				num_elements = gas.num_eqs**2 * num_cells
				self.jacob_dim = gas.num_eqs * num_cells

				row_idxs_center = np.zeros(num_elements, dtype=np.int32)
				col_idxs_center = np.zeros(num_elements, dtype=np.int32)
				row_idxs_upper = np.zeros(num_elements - gas.num_eqs**2, dtype=np.int32)
				col_idxs_upper = np.zeros(num_elements - gas.num_eqs**2, dtype=np.int32)
				row_idxs_lower = np.zeros(num_elements - gas.num_eqs**2, dtype=np.int32)
				col_idxs_lower = np.zeros(num_elements - gas.num_eqs**2, dtype=np.int32)

				# TODO: definitely a faster way to do this
				lin_idx_A = 0
				lin_idx_B = 0
				lin_idx_C = 0
				for i in range(gas.num_eqs):
					for j in range(gas.num_eqs):
						for k in range(num_cells):

							row_idxs_center[lin_idx_A] = i * num_cells + k
							col_idxs_center[lin_idx_A] = j * num_cells + k
							lin_idx_A += 1

							if k < (num_cells - 1):
								row_idxs_upper[lin_idx_B] = i * num_cells + k
								col_idxs_upper[lin_idx_B] = j * num_cells + k + 1
								lin_idx_B += 1

							if k > 0:
								row_idxs_lower[lin_idx_C] = i * num_cells + k
								col_idxs_lower[lin_idx_C] = j * num_cells + k - 1
								lin_idx_C += 1

				self.jacob_row_idxs = \
					np.concatenate((row_idxs_center, row_idxs_lower, row_idxs_upper))
				self.jacob_col_idxs = \
					np.concatenate((col_idxs_center, col_idxs_lower, col_idxs_upper))

			# "steady" convergence measures
			if solver.run_steady:
				self.d_sol_norm_l2 = 0.0
				self.d_sol_norm_l1 = 0.0
				self.d_sol_norm_history = np.zeros((solver.num_steps, 2), dtype=REAL_TYPE)

	def update_sol_hist(self):
		"""
		Update time history of solution and RHS function
		for multi-stage time integrators
		"""

		# primitive and conservative state history
		self.sol_hist_cons[1:] = self.sol_hist_cons[:-1]
		self.sol_hist_prim[1:] = self.sol_hist_prim[:-1]
		self.sol_hist_cons[0] = self.sol_cons.copy()
		self.sol_hist_prim[0] = self.sol_prim.copy()

		# TODO: RHS update should occur at the FIRST subiteration
		# 	right after the RHS is calculated
		# RHS function history
		self.rhs_hist[1:] = self.rhs_hist[:-1]
		self.rhs_hist[0] = self.rhs.copy()

	def update_snapshots(self, solver):

		store_idx = int((solver.iter - 1) / solver.out_interval) + 1

		if solver.prim_out:
			self.prim_snap[:, :, store_idx] = self.sol_prim
		if solver.cons_out:
			self.cons_snap[:, :, store_idx] = self.sol_cons
		if solver.source_out:
			self.source_snap[:, :, store_idx - 1] = self.source
		if solver.rhs_out:
			self.rhs_snap[:, :, store_idx - 1] = self.rhs

	def write_snapshots(self, solver, failed):
		"""
		Save snapshot matrices to disk
		"""

		unsteady_output_dir = solver.unsteady_output_dir

		# account for failed simulation dump
		# TODO: need to account for non-unity out_interval
		if failed:
			offset = 1
		else:
			offset = 2
		final_idx = int((solver.iter - 1) / solver.out_interval) + offset

		if solver.prim_out:
			sol_prim_file = os.path.join(unsteady_output_dir,
										"solPrim_" + solver.sim_type + ".npy")
			np.save(sol_prim_file, self.prim_snap[:, :, :final_idx])
		if solver.cons_out:
			sol_cons_file = os.path.join(unsteady_output_dir,
										"solCons_" + solver.sim_type + ".npy")
			np.save(sol_cons_file, self.cons_snap[:, :, :final_idx])
		if solver.source_out:
			source_file = os.path.join(unsteady_output_dir,
										"source_" + solver.sim_type + ".npy")
			np.save(source_file, self.source_snap[:, :, :final_idx - 1])
		if solver.rhs_out:
			sol_rhs_file = os.path.join(unsteady_output_dir,
										"solRHS_" + solver.sim_type + ".npy")
			np.save(sol_rhs_file, self.rhs_snap[:, :, :final_idx - 1])

	def write_restart_file(self, solver):
		"""
		Write restart files containing primitive and conservative fields,
		and associated physical time
		"""

		# TODO: write previous time step(s) for multi-step methods
		# 	to preserve time accuracy at restart

		# write restart file to zipped file
		restart_file = os.path.join(solver.restart_output_dir,
									"restartFile_" + str(solver.restart_iter) + ".npz")
		np.savez(restart_file,
					sol_time=solver.sol_time,
					sol_prim=self.sol_prim,
					sol_cons=self.sol_cons)

		# write iteration number files
		restartIterFile = os.path.join(solver.restart_output_dir, "restart_iter.dat")
		with open(restartIterFile, "w") as f:
			f.write(str(solver.restart_iter) + "\n")

		restart_phys_iter_file = os.path.join(solver.restart_output_dir,
											"restart_iter_" + str(solver.restart_iter) + ".dat")
		with open(restart_phys_iter_file, "w") as f:
			f.write(str(solver.iter) + "\n")

		# iterate file count
		if solver.restart_iter < solver.num_restarts:
			solver.restart_iter += 1
		else:
			solver.restart_iter = 1

	def write_steady_data(self, solver):

		# write norm data to ASCII file
		steady_file = os.path.join(unsteady_output_dir, "steady_convergence.dat")
		if (solver.iter == 1):
			f = open(steady_file, "w")
		else:
			f = open(steady_file, "a")
		out_string = (("%8i %18.14f %18.14f\n")
						% (solver.time_iter - 1, self.d_sol_norm_l2, self.d_sol_norm_l1))
		f.write(out_string)
		f.close()

		# write field data
		sol_prim_file = os.path.join(unsteady_output_dir, "sol_prim_steady.npy")
		np.save(sol_prim_file, self.sol_prim)
		sol_cons_file = os.path.join(unsteady_output_dir, "sol_cons_steady.npy")
		np.save(sol_cons_file, self.sol_cons)

	def calc_d_sol_norms(self, solver, time_type):
		"""
		Calculate and print solution change norms
		Note that output is ORDER OF MAGNITUDE of residual norm
		(i.e. 1e-X, where X is the order of magnitude)
		"""

		if time_type == "implicit":
			d_sol = self.sol_hist_prim[0] - self.sol_hist_prim[1]
		else:
			# TODO: only valid for single-stage explicit schemes
			d_sol = self.sol_prim - self.sol_hist_prim[0]

		norm_l2, norm_l1 = self.calc_norms(d_sol, solver.res_norm_prim)

		norm_out_l2 = np.log10(norm_l2)
		norm_out_l1 = np.log10(norm_l1)
		out_string = (("%8i:   L2: %18.14f,   L1: %18.14f")
						% (solver.time_iter, norm_out_l2, norm_out_l1))
		print(out_string)

		self.d_sol_norm_l2 = norm_l2
		self.d_sol_norm_l1 = norm_l1
		self.d_sol_norm_history[solver.iter - 1, :] = [norm_l2, norm_l1]

	def calc_res_norms(self, solver, subiter):
		"""
		Calculate and print linear solve residual norms
		Note that output is ORDER OF MAGNITUDE of residual norm
		(i.e. 1e-X, where X is the order of magnitude)
		"""

		# TODO: pass conservative norm factors if running cons implicit solve
		norm_l2, norm_l1 = self.calc_norms(self.res, solver.res_norm_prim)

		# don't print for "steady" solve
		if not solver.run_steady:
			norm_out_l2 = np.log10(norm_l2)
			norm_out_l1 = np.log10(norm_l1)
			out_string = ((str(subiter + 1) + ":\tL2: %18.14f, \tL1: %18.14f")
							% (norm_out_l2, norm_out_l1))
			print(out_string)

		self.res_norm_l2 = norm_l2
		self.res_norm_l1 = norm_l1
		self.res_norm_history[solver.iter - 1, :] = [norm_l2, norm_l1]

	def calc_norms(self, arr_in, norm_facs):
		"""
		Compute L1 and L2 norms of arr_in
		arr_in assumed to be in [numVars, num_cells] order
		"""

		arr_abs = np.abs(arr_in)

		# L2 norm
		arr_norm_l2 = np.sum(np.square(arr_abs), axis=1)
		arr_norm_l2[:] /= arr_in.shape[1]
		arr_norm_l2 /= np.square(norm_facs)
		arr_norm_l2 = np.sqrt(arr_norm_l2)
		arr_norm_l2 = np.mean(arr_norm_l2)

		# L1 norm
		arr_norm_l1 = np.sum(arr_abs, axis=1)
		arr_norm_l1[:] /= arr_in.shape[1]
		arr_norm_l1 /= norm_facs
		arr_norm_l1 = np.mean(arr_norm_l1)

		return arr_norm_l2, arr_norm_l1
