import os
import numpy as np
import scipy.linalg as LA
import copy
import sys

from perform.misc_funcs import deim_helper

class AdaptROM():
    def __init__(self, solver, rom_domain, sol_domain):

        # attributes needed:
        # window of high-dim RHS
        
        # methods needed: update_residualSampling_window
        # adeim
        # initialize_window
        
        # this assumes vector construction of ROM
        # these initializations need to be changed for the scalar ROM case
        
        assert rom_domain.fom_sol_init_window is not None, "Missing input files for adaptive ROM basis"
        
        # initialize F in the adaptive ROM window with the FOM solution.
        self.fcn_window = rom_domain.fom_sol_init_window[:, -rom_domain.adaptive_rom_window_size:]
        
        self.residual_samplepts = np.zeros(rom_domain.num_residual_comp)
        self.residual_samplepts_comp = np.zeros(sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells - rom_domain.num_residual_comp) # this is the complement
        self.fom_snapshots = np.array([])
        self.fom_snapshots_prim = np.array([])
        self.fom_snapshots_scaled = np.array([])
        self.basis_inc = np.array([])
        self.rom_sol_change = []
                
    def load_fom(self, rom_domain, model):
        # this has to be done for every model in model list if the code is modified to work with multiple models.
        # initializes the window
        model_dir = rom_domain.model_dir
        
        # conservative variables
        try:
            fom_snap = np.load(os.path.join(model_dir, rom_domain.adaptive_rom_fom_file))
            self.fom_snapshots = np.reshape(fom_snap, (-1, fom_snap.shape[-1]), order="C")
            fom_snap_scaled = np.zeros_like(fom_snap)
            n_snap = fom_snap_scaled.shape[-1]

            # scale snapshots
            for i in range(n_snap):
                if sol_domain.time_integrator.dual_time:
                    fom_snap_scaled[:, :, i] = model.scale_profile(
                                                    fom_snap[:, :, i],
                                                    normalize=True,
                                                    norm_fac_prof=model.norm_fac_prof_prim,
                                                    norm_sub_prof=model.norm_sub_prof_prim,
                                                    center=True,
                                                    cent_prof=model.cent_prof_prim,
                                                    inverse=False,
                                                    )
                else:
                    fom_snap_scaled[:, :, i] = model.scale_profile(
                                                    fom_snap[:, :, i],
                                                    normalize=True,
                                                    norm_fac_prof=model.norm_fac_prof_cons,
                                                    norm_sub_prof=model.norm_sub_prof_cons,
                                                    center=True,
                                                    cent_prof=model.cent_prof_cons,
                                                    inverse=False,
                                                    )

            self.fom_snapshots_scaled = np.reshape(fom_snap_scaled, (-1, fom_snap_scaled.shape[-1]), order="C")

        except:
            raise Exception("File for snapshots in conservative variables not found.")
        
        # primitive variables
        try:
            fom_snap_prim = np.load(os.path.join(model_dir, rom_domain.adaptive_rom_fom_file))
            self.fom_snapshots_prim = np.reshape(fom_snap_prim, (-1, fom_snap_prim.shape[-1]), order="C")
            
        except:
            raise Exception("File for snapshots in primitive variables not found.")
    
    # this function simply updates the values of F in the adaptation window as new values come in.
    def cycle_fwindow(self, new_state):
        """ Cycle the window and add new state"""
    
        temp_fcn_window = self.fcn_window.copy()
        temp_fcn_window = temp_fcn_window[:,1:]
        
        self.fcn_window = np.concatenate((temp_fcn_window, new_state), axis=1)
        
    # Update the window and find the sampling set of points (and its complement) for the residual
    def update_res_sampling_window(self, rom_domain, solver, sol_domain, trial_basis, deim_idx_flat, decoded_rom, model, use_fom):
            
        if solver.time_iter >= rom_domain.adaptive_rom_init_time + 1:

            # this is ROM reconstruction
            q_k_temp = decoded_rom
            q_k = q_k_temp.reshape((-1,1))

            if solver.time_iter == rom_domain.adaptive_rom_init_time + 1 or solver.time_iter % rom_domain.sampling_update_freq  == 0:
                
                # compute F[:,k]
                # use_fom=1 and 2 are only for testing.
                if solver.time_iter > rom_domain.adaptive_rom_init_time + 1:
                    if use_fom == 1:                
                        # directly uses the FOM solution for F
                        f_k = self.fom_snapshots[:,solver.time_iter-1:solver.time_iter].copy()
                    elif use_fom == 0:
                        # evaluates the fully discrete rhs function with the ROM solution (line 11, Algorithm 1 in Ben's paper)
                        f_k = rom_domain.time_stepper.calc_fullydiscrhs(sol_domain, q_k, solver, rom_domain)
                    elif use_fom == 2:
                        # evaluates the fully discrete rhs function with the FOM solution
                        fom_qk = self.fom_snapshots[:,solver.time_iter:solver.time_iter+1].copy()
                        f_k = rom_domain.time_stepper.calc_fullydiscrhs(sol_domain, fom_qk, solver, rom_domain)
                    
                    # scale snapshot
                    f_k = f_k.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
                    #since in dual time stepping we are solving for primitive variables.
                    if sol_domain.time_integrator.dual_time:
                        f_k = model.scale_profile(f_k, normalize=True,
                                                norm_fac_prof=model.norm_fac_prof_prim,
                                                norm_sub_prof=model.norm_sub_prof_prim,
                                                center=True,
                                                cent_prof=model.cent_prof_prim,
                                                inverse=False,
                                                )
                    else:
                        f_k = model.scale_profile(f_k, normalize=True,
                                              norm_fac_prof=model.norm_fac_prof_cons,
                                              norm_sub_prof=model.norm_sub_prof_cons,
                                              center=True,
                                              cent_prof=model.cent_prof_cons,
                                              inverse=False,
                                              )
                    f_k = f_k.reshape((-1,1))
      
                    # update F inside the window (self.fcn_window) by appending the newly computed f_k
                    if self.fcn_window.shape[1] == rom_domain.adaptive_rom_window_size:
                        self.cycle_fwindow(f_k)
                    else:
                        self.fcn_window = np.concatenate((self.fcn_window, f_k), axis=1)
                
                # compute R_k (line 12 in Algorithm 1 of Ben's paper)
                r_k = self.fcn_window - (trial_basis @ np.linalg.pinv(trial_basis[deim_idx_flat, :]) @ self.fcn_window[deim_idx_flat , :])    
    
                # find s_k and its complement \breve{s}_k (line 13 in Algorithm 1 of Ben's paper)
                sorted_idx = np.argsort(-np.sum(r_k**2,axis=1))
                all_idx = list(range(sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells))
                all_idx = np.array(all_idx)
                sorted_residual_idx = all_idx[sorted_idx]

                # line 14 in Algorithm 1 of Ben's paper
                self.residual_samplepts = sorted_residual_idx[:rom_domain.num_residual_comp].astype(int)
                self.residual_samplepts_comp = sorted_residual_idx[rom_domain.num_residual_comp:].astype(int)
    
            else:
    
                f_k = np.zeros((sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells, 1))
                
                # take the union of s_k and p_k
                idx_union = np.concatenate((self.residual_samplepts, deim_idx_flat))
                idx_union = np.unique(idx_union)
                idx_union = np.sort(idx_union)
                
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Needs to be modified to accelerate the code>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # first evaluate the fully discrete RHS
                # note that the computationally efficient approach would be to only evaluate the right hand side at select components
                # inefficient. first evaluates right hand side at all components and only selects those components needed
                if use_fom == 1:
                    temp_F_k = self.fom_snapshots[:,solver.time_iter-1:solver.time_iter] 
                elif use_fom == 0:
                    temp_f_k = rom_domain.time_stepper.calc_fullydiscrhs(sol_domain, q_k, solver, rom_domain)
                elif use_fom == 2:
                    fom_qk = self.fom_snapshots[:,solver.time_iter:solver.time_iter+1].copy()
                    temp_f_k = rom_domain.time_stepper.calc_fullydiscrhs(sol_domain, fom_qk, solver, rom_domain)
                
                # scale snapshot
                temp_f_k = temp_f_k.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
                if sol_domain.time_integrator.dual_time:
                    temp_f_k = model.scale_profile(temp_f_k, normalize=True,
                                            norm_fac_prof=model.norm_fac_prof_prim,
                                            norm_sub_prof=model.norm_sub_prof_prim,
                                            center=True,
                                            cent_prof=model.cent_prof_prim,
                                            inverse=False,
                                            )
                else:
                    temp_f_k = model.scale_profile(temp_f_k, normalize=True,
                                            norm_fac_prof=model.norm_fac_prof_cons,
                                            norm_sub_prof=model.norm_sub_prof_cons,
                                            center=True,
                                            cent_prof=model.cent_prof_cons,
                                            inverse=False,
                                            )
                temp_f_k = temp_f_k.reshape((-1,1))
                f_k[idx_union, :] = temp_f_k[idx_union, :]            
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                #note this is not the same as Ben's algorithm (line 18, Algorithm 1 in his paper) that only uses deim samples to reconstruct.
                #here it uses a combination of deim samples and residual samples to reconstruct
                f_k[self.residual_samplepts_comp, :] = trial_basis[self.residual_samplepts_comp, :] @ np.linalg.pinv(trial_basis[idx_union, :]) @ f_k[idx_union, :]
    
                # Update F inside the window (self.fcn_window) by appending the newly computed f_k
                if self.fcn_window.shape[1] == rom_domain.adaptive_rom_window_size:
                    self.cycle_fwindow(f_k)
                else:
                    self.fcn_window = np.concatenate((self.fcn_window, f_k), axis=1)

    # line 21 in Algorithm 1 of Ben's paper, Algorithm 3
    def adeim(self, rom_domain, curr_trial_basis, deim_idx_flat, deim_dim, n_mesh, solver, code):
        
        old_basis = curr_trial_basis.copy()
        trial_basis = curr_trial_basis.copy()
        
        r = copy.copy(rom_domain.update_rank)
        
        # rhs function evaluated at the residual sampling points
        f_s = self.fcn_window[self.residual_samplepts, :]

        #for onestep with nonlocal sampling
        # if solver.time_iter % rom_domain.sampling_update_freq  == 0:
        #     #AFDEIM
        #     C, _, _, _ = np.linalg.lstsq(trial_basis, self.fcn_window, rcond=None)
        #     res = trial_basis @ C - self.fcn_window
            
        #     idx_eval_basis = np.concatenate((self.residual_samplepts, self.residual_samplepts_comp))
        #     idx_eval_basis = np.unique(idx_eval_basis)
        #     idx_eval_basis = np.sort(idx_eval_basis)
        # else:
        #     #ADEIM
        #     f_p = self.fcn_window[deim_idx_flat, :]
        #     C, _, _, _ = np.linalg.lstsq(trial_basis[deim_idx_flat, :], f_p, rcond=None) # not sure if it should be solve or lstsq
        #     res = trial_basis[self.residual_samplepts, :] @ C - f_s
            
        #     idx_eval_basis = self.residual_samplepts
        
        # Ben's paper uses ADEIM, but Wayne's code was unstable with DEIM, it only worked with AFDEIM
        if rom_domain.adeim_update == "ADEIM":
            # rhs function evaluated at the deim sampling points
            f_p = self.fcn_window[deim_idx_flat, :]
            # line 2 in algorithm 3 of Ben's paper
            C, _, _, _ = np.linalg.lstsq(trial_basis[deim_idx_flat, :], f_p, rcond=None) 
            res = trial_basis[self.residual_samplepts, :] @ C - f_s
            
            idx_eval_basis = self.residual_samplepts
            
        elif rom_domain.adeim_update == "AODEIM":
            idx_union = np.concatenate((self.residual_samplepts, deim_idx_flat))
            idx_union = np.unique(idx_union)
            idx_union = np.sort(idx_union)
            f_p = self.fcn_window[idx_union, :]
            C, _, _, _ = np.linalg.lstsq(trial_basis[idx_union, :], f_p, rcond=None)
            res = trial_basis[idx_union, :] @ C - self.fcn_window[idx_union, :]
            
            idx_eval_basis = idx_union
            
        elif rom_domain.adeim_update == "AFDEIM":
            # C and res are computed everywhere, not just at the sampling points
            C, _, _, _ = np.linalg.lstsq(trial_basis, self.fcn_window, rcond=None)
            res = trial_basis @ C - self.fcn_window
            
            # this way basis will be updated at all points
            idx_eval_basis = np.concatenate((self.residual_samplepts, self.residual_samplepts_comp))
            idx_eval_basis = np.unique(idx_eval_basis)
            idx_eval_basis = np.sort(idx_eval_basis)

        idx_eval_basis = idx_eval_basis.astype(int)

        _, s_v, s_rh = np.linalg.svd(res)
        s_r = s_rh.T
        
        ct_pinv = np.linalg.pinv(C.T)

        # r = min(r, length(s_v))
        # why i:i+1? isn't it just i?
        for i in range(r):
            alfa = -res @ s_r[:, i:i+1]
            beta = ct_pinv @ s_r[:, i:i+1]
            trial_basis[idx_eval_basis, :] = trial_basis[idx_eval_basis, :] + alfa @ beta.T
            
        #one-step basis adaptation
        # q_hat = model.scale_profile(self.fcn_window[:,-1], normalize=True,
        #                                 norm_fac_prof=model.norm_fac_prof_prim,
        #                                 norm_sub_prof=model.norm_sub_prof_prim,
        #                                 center=False,
        #                                 cent_prof=model.cent_prof_prim,
        #                                 inverse=True,
        #                                 )
        # q_tilde = model.scale_profile(trial_basis @ code, normalize=True,
        #                                 norm_fac_prof=model.norm_fac_prof_prim,
        #                                 norm_sub_prof=model.norm_sub_prof_prim,
        #                                 center=False,
        #                                 cent_prof=model.cent_prof_prim,
        #                                 inverse=True,
        #                                 )
        # delta_basis = np.zeros(trial_basis.shape)
        # delta = (self.fcn_window[:,-1] - trial_basis @ code)
        # #delta = q_hat - q_tilde
        # #delta = self.fcn_window[:,-1] - (trial_basis[idx_eval_basis, :] @ np.linalg.pinv(trial_basis[idx_eval_basis, :]) @ self.fcn_window[:,-1])
        # delta_basis[idx_eval_basis, :] = (np.expand_dims(delta, axis=1)  @ np.expand_dims(code, axis=1).T) / (np.linalg.norm(code)**2)
        # #delta[idx_eval_basis, :] = (f_s[:,-1] @ code.T) / np.linalg.norm(code)
        # trial_basis[idx_eval_basis, :] = trial_basis[idx_eval_basis, :] + delta_basis[idx_eval_basis, :]

        # orthogonalize basis (line 13 in Algorithm 3 of Ben's paper)
        trial_basis = LA.orth(trial_basis)   
        
        # Update QDEIM points (since basis is updated, hyper-reduction sampling points should also be updated)
        sampling_id = deim_helper(trial_basis, deim_dim, n_mesh)
            
        return trial_basis, sampling_id
    
    # this seems to be for testing purposes to compare Ben's method to repeatedly applying POD
    # def pod_basis(self, deim_dim, nmesh, old_basis, solver):
        
    #     # compute basis using POD from snapshots
    #     u, sv, _ = np.linalg.svd(self.fcn_window)
    #     trial_basis = u[:, :deim_dim]   

    #     # Update QDEIM points
    #     sampling_id = deim_helper(trial_basis, deim_dim, n_mesh)
        
    #     return trial_basis, sampling_id
