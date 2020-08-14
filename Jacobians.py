# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:39:55 2020

@author: ashis
"""

import numpy as np
from stateFuncs import calcCpMixture, calcGasConstantMixture, calcStateFromPrim
import constants
from spaceSchemes import calcSource, reconstruct_2nd, calcRoeDissipation




class def_cellidx:
    
    def __init__(self,geom):
        
        self.l2edge = np.linspace(1,geom.numCells,geom.numCells)
        self.r2edge = np.linspace(1,geom.numCells,geom.numCells)
        
        self.evaluate = np.linspace(1,geom.numCells,geom.numCells)
        
        self.l22edge = np.linspaece(1,geom.numCells-1,geom.numCells-1)
        self.r22edge = np.linspace(2,geom.numCells,geom.numCells-1)
        
        self.nevaluate = geom.numCells
        
        self.sample_in_evaluate = np.linspace(1,geom.numCells,geom.numCells)
        
        self.l2edge_in_evaluate = np.linspace(1,geom.numCells,geom.numCells)
        self.r2edge_in_evaluate = np.linspace(1,geom.numCells,geom.numCells)
        
        self.l22edge_in_evaluate = np.linspaece(1,geom.numCells-1,geom.numCells-1)
        self.r22edge_in_evaluate = np.linspace(2,geom.numCells,geom.numCells-1)
        
class def_edgeidx:
    
    def __init__(self,geom):
        
        self.l2sample = np.linspace(1,geom.numNodes-1,geom.numNodes-1)
        self.r2sample = np.linspace(2,geom.numNodes,geom.numNodes-1)
        
        self.evaluate = np.linspace(1,geom.numNodes,geom.numNodes)
        
        self.l2sample_in_evaluate = np.linspace(1,geom.numNodes-1,geom.numNodes-1)
        self.r2sample_in_evaluate = np.linspace(2,geom.numNodes,geom.numNodes-1)
        
        self.nedge = geom.numNodes
        
        self.isl2sample = np.ones(geom.numNodes)
        self.isl2sample[-1] = 0
        
        self.isr2sample = np.ones(geom.numNodes)
        self.isr2sample[0] = 0
        
        self.cell_l2_edge_in_eval = np.zeros(geom.numNodes)
        self.cell_l2_edge_in_eval[1:] = np.linspace(1,geom.numNodes-1,geom.numNodes-1)
        self.cell_l2_edge_in_eval[0] = np.nan
        
        self.cell_r2_edge_in_eval = np.zeros(geom.numNodes)
        self.cell_r2_edge_in_eval[:geom.numNodes-1] = np.linspace(1,geom.numNodes-1,geom.numNodes-1)
        self.cell_r2_edge_in_eval[geom.numNodes] = np.nan
        
        self.cell_l22_edge_in_eval = np.zeros(geom.numNodes)
        self.cell_l22_edge_in_eval[2:] = np.linspace(1,geom.numNodes-2,geom.numNodes-2)
        self.cell_l22_edge_in_eval[0] = np.nan
        self.cell_l22_edge_in_eval[1] = np.nan
        
        self.cell_r22_edge_in_eval = np.zeros(geom.numNodes)
        self.cell_r22_edge_in_eval[:geom.numNodes-2] = np.linspace(1,geom.numNodes-2,geom.numNodes-2)
        self.cell_r22_edge_in_eval[geom.numNodes] = np.nan
        self.cell_r22_edge_in_eval[geom.numNodes-1] = np.nan
        
        

def calc_RAE(truth,pred):
    
    RAE = np.mean(np.abs(truth-pred))/np.max(np.abs(truth))
    
    return RAE


def calc_dsolPrim(sol,gas):
    
    
    rhs = sol.RHS.copy()
    Jac = calc_dsolPrimdsolCons(sol.solCons, sol.solPrim, gas)
    
    for i in range(sol.solPrim.shape[0]):
        
        rhs[i,:] = Jac[:,:,i] @ rhs[i,:]
    
    return rhs

def calc_dsolPrimdsolCons(solCons,solPrim,gas):
    
    RUniv = 8314.0
    gamma_matrix_inv = np.zeros((gas.numEqs,gas.numEqs,solPrim.shape[0]))
    
    rho = solCons[:,0]
    p = solPrim[:,0]
    u = solPrim[:,1]
    T = solPrim[:,2]
    
    if (gas.numSpecies > 1):
        Y = solPrim[:,3:]
        massFracs = solPrim[:,3:]
    else:
        Y = solPrim[:,3]
        massFracs = solPrim[:,3]
        
    Ri = calcGasConstantMixture(massFracs, gas)
    Cpi = calcCpMixture(massFracs, gas)
    
    rhop = 1/(Ri*T)
    rhoT = -rho/T
    hT = Cpi
    d = rho*rhop*hT + rhoT
    h0 = (solCons[:,2]+p)/rho
    
    if (gas.numSpecies == 0):
        gamma11 = (rho*hT + rhoT*(h0-(u*u)))/d
        
    else:
        rhoY = -(rho*rho)*(RUniv*T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
        hY = gas.enthRefDiffs + (T-gas.tempRef)*(gas.Cp[0]-gas.Cp[gas.numSpecies_full-1])
        gamma11 = (rho*hT + rhoT*(h0-(u*u))+ (Y*(rhoY*hT - rhoT*hY)))/d #s
        
    gamma_matrix_inv[0,0,:] = gamma11
    gamma_matrix_inv[0,1,:] = u*rhoT/d
    gamma_matrix_inv[0,2,:] = -rhoT/d
    
    if (gas.numSpecies > 0):
        gamma_matrix_inv[0,3:,:] = (rhoT*hY - rhoY*hT)/d
        
    gamma_matrix_inv[1,0,:] = -u/rho
    gamma_matrix_inv[1,1,:] = 1/rho
    
    if (gas.numSpecies == 0):
        gamma_matrix_inv[2,0,:] = (-rhop*(h0-(u*u))+1)/d
        
    else:
        gamma_matrix_inv[2,0,:] = (-rhop*(h0-(u*u)) + 1 + (Y * (rho*rhop*hY + rhoY))/rho)/d #s
        gamma_matrix_inv[2,3:,:] = -(rho*rhop*hY + rhoY)/(rho*d)
        
    gamma_matrix_inv[2,1,:] = -u*rhop/d
    gamma_matrix_inv[2,2,:] = rhop/d
    
    if (gas.numSpecies > 0):
        gamma_matrix_inv[3:,0,:] = -Y/rho
        
        for i in range(3,gas.numEqs):
            gamma_matrix_inv[i,i,:] = 1/rho
            
            
    return gamma_matrix_inv

def calc_dsolConsdsolPrim(solCons,solPrim,gas):
    
    gamma_matrix = np.zeros((gas.numEqs,gas.numEqs,solPrim.shape[0]))
    rho = solCons[:,0]
    p = solPrim[:,0]
    u = solPrim[:,1]
    T = solPrim[:,2]

    if (gas.numSpecies > 1):
        Y = solPrim[:,3:]
        massFracs = solPrim[:,3:]
    else:
        Y = solPrim[:,3]
        massFracs = solPrim[:,3]
        
    Ri = calcGasConstantMixture(massFracs, gas)
    Cpi = calcCpMixture(massFracs, gas)
    
    rhop = 1/(Ri*T)
    rhoT = -rho/T
    hT = Cpi
    d = rho*rhop*hT + rhoT
    h0 = (solCons[:,2]+p)/rho
    
    if (gas.numSpecies > 0):
        rhoY = -(rho**2)*(constants.RUniv * T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
        hY = gas.enthRefDiffs + (T-gas.tempRef)*(gas.Cp[0]-gas.Cp[gas.numSpecies_full-1])
        
    gamma_matrix[0,0,:] = rhop
    gamma_matrix[0,2,:] = rhoT
    
    if (gas.numSpecies > 0):
        gamma_matrix[0,3:,:] = rhoY
        
    gamma_matrix[1,0,:] = u*rhop
    gamma_matrix[1,1,:] = rho
    gamma_matrix[1,2,:] = u*rhoT
    
    if (gas.numSpecies > 0):
        gamma_matrix[1,3:,:] = u*rhoY
        
    gamma_matrix[2,0,:] = rhop*h0 + rho*hp - 1
    gamma_matrix[2,1,:] = rho*u
    gamma_matrix[2,2,:] = rhoT*h0 + rho*hT
    
    if (gas.numSpecies > 0):
        gamma_matrix[2,3:,:] = rhoY*h0 + rho*hY
        
        for i in range(3,gas.numEqs):
            gamma_matrix[i,0,:] = Y[:,i-3] * rhop
            gamma_matrix[i,2,:] = Y[:,i-3] * rhoT
            
            for j in range(3,gas.numEqs):
                gamma_matrix[i,j,:] = (i==j)*rho + Y[:,i-3]*rhoY[:,j-3]
                
    return gamma_matrix

    

def calc_dSourcedsolPrim(sol,gas,geom,dt):
    
    dSdQp = np.zeros((gas.numEqs,gas.numEqs,geom.numCells))

    rho = sol.solCons[:,0]
    p = sol.solPrim[:,0]
    T = sol.solPrim[:,2]

    
    if (gas.numSpecies > 1):
        Y = sol.solPrim[:,3:]
        massFracs = sol.solPrim[:,3:]
    else:
        Y = (sol.solPrim[:,3]).reshape((sol.solPrim[:,3].shape[0],1))
        massFracs = sol.solPrim[:,3]
        
    Ri = calcGasConstantMixture(massFracs, gas)
    
    rhop = 1/(Ri*T)
    rhoT = -rho/T
    
    rhoY = -(rho*rho) * (constants.RUniv*T/p) * (1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
    
    wf_rho = 0
    
    A = gas.preExpFact
    wf = A * np.exp(gas.actEnergy/T)
    
    for i in range(gas.numSpecies):
        
        if (gas.nuArr[i] != 0):
            wf = wf*((Y[:,i]*rho/gas.molWeights[i])**gas.nuArr[i])
            wf[Y[:,i] <= 0] = 0
            
    for i in range(gas.numSpecies):
        
        if (gas.nuArr[i] != 0):
            wf = np.minimum(wf,Y[:,i]/dt*rho)
    
    for i in range(gas.numSpecies):
        
        if (gas.nuArr[i] != 0):       
            wf_rho = wf_rho + wf*gas.nuArr[i]/rho
            
    wf_T = wf_rho * rhoT - wf * gas.actEnergy/T**2
    wf_p = wf_rho * rhop
    wf_Y = wf_rho * rhoY
    
    for i in range(gas.numSpecies):
        
        s = wf_Y[(Y[:,i]>0)].shape[0]
        wf_Y[(Y[:,i]>0)] = wf_Y[(Y[:,i]>0)] + wf[(Y[:,i]>0)] * (gas.nuArr[i]/Y[(Y[:,i]>0),:]).reshape(s) 
        dSdQp[3+i,0,:] = -gas.molWeightNu[i] * wf_p
        dSdQp[3+i,2,:] = -gas.molWeightNu[i] * wf_T
        dSdQp[3+i,3+i,:] = -gas.molWeightNu[i] * wf_Y
        
    
    return dSdQp
    
def calc_dSourcedsolPrim_FD(sol,gas,geom,params,dt,dx):
    
    dSdQp = np.zeros((gas.numEqs,gas.numEqs,geom.numCells))
    dSdQp_an = calc_dSourcedsolPrim(sol,gas,geom,dt)
    
    dx = dx*np.maximum(1e-14,np.amax(np.abs(sol.solPrim)))
    
    for i in range(geom.numCells):
        
        for j in range(gas.numEqs):
            
            solprim_curr = sol.solPrim.copy()
            
            #adding positive perturbations
            solprim_curr[i,j] = solprim_curr[i,j] + 1*dx
            [solcons_curr, RMix, enthRefMix, CpMix] = calcStateFromPrim(solprim_curr,gas)
            S1 = calcSource(solprim_curr, solcons_curr[:,0], params, gas)
            
            #adding positive perturbations
            solprim_curr[i,j] = solprim_curr[i,j] - 1*dx
            [solcons_curr, RMix, enthRefMix, CpMix] = calcStateFromPrim(solprim_curr,gas)
            S2 = calcSource(solprim_curr, solcons_curr[:,0], params, gas)

            #calculating the jacobians
            Jac = (S1-S2)/(dx)
            dSdQp[:,j,i] = Jac[i,:]
            
            #Unperturbing
            solprim_curr[i,j] = solprim_curr[i,j] 
            [solcons_curr, RMix, enthRefMix, CpMix] = calcStateFromPrim(solprim_curr,gas)
    
    diff= calc_RAE(dSdQp_an.ravel(), dSdQp.ravel())
    
    return diff

def calc_dSourcedsolPrim_imag(sol,gas,geom,params,dt,dx):
    
    dSdQp = np.zeros((gas.numEqs,gas.numEqs,geom.numCells))
    dSdQp_an = calc_dSourcedsolPrim(sol,gas,geom,dt)
    
    for i in range(geom.numCells):
        for j in range(gas.numEqs):
            
            solprim_curr = sol.solPrim.copy()
            solprim_curr = solprim_curr.astype(dtype=np.complex64)
            #adding complex perturbations
            solprim_curr[i,j] = solprim_curr[i,j] + complex(0,dx)
            [solcons_curr, RMix, enthRefMix, CpMix] = calcStateFromPrim(solprim_curr,gas)
            S1 = calcSource(solprim_curr, solcons_curr[:,0], params, gas)
            
            #calculating jacobian
            Jac = S1.imag/dx
            
            dSdQp[:,j,i] = Jac[i,:]
            
            #Unperturbing
            solprim_curr[i,j] = solprim_curr[i,j] - complex(0,dx)
            
    diff= calc_RAE(dSdQp_an.ravel(), dSdQp.ravel())
    
    return diff



def calc_Ap(solPrim, rho, h0, gas):
    
    Ap = np.zeros((gas.numEqs, gas.numEqs, solPrim.shape[0]))
    
    p = solPrim[:,0]
    u = solPrim[:,1]
    T = solPrim[:,2]
    
    if (gas.numSpecies > 1):
        Y = solPrim[:,3:]
        massFracs = solPrim[:,3:]
    else:
        Y = solPrim[:,3]
        massFracs = solPrim[:,3]
        
    Ri = calcGasConstantMixture(massFracs, gas)
    Cpi = calcCpMixture(massFracs, gas)
    rhoY = -(rho*rho)*(constants.RUniv*T/p)*(1/gas.molWeights[0] - 1/gas.molWeights[gas.numSpecies_full-1])
    hY = gas.enthRefDiffs + (T-gas.tempRef)*(gas.Cp[0]-gas.Cp[gas.numSpecies_full-1])
    
    rhop = 1/(Ri*T)
    rhoT = -rho/T
    hT = Cpi
    hp = 0
    
    Ap[0,0,:] = rhop*u
    Ap[0,1,:] = rho
    Ap[0,2,:] = rhoT*u
    
    if (gas.numSpecies > 0):
        Ap[0,3:,:] = u * rhoY
        
    Ap[1,0,:] = rhop * (u**2) + 1
    Ap[1,1,:] = 2*rho*u
    Ap[1,2,:] = rhoT*(u**2)
    
    if (gas.numSpecies > 0):
        Ap[1,3:,:] = u**2 * rhoY
        
    Ap[2,0,:] = u*(rhop*h0 + rho*hp)
    Ap[2,1,:] = rho*(u**2+h0)
    Ap[2,2,:] = u*(rhoT*h0 + rho*hT)
    
    if (gas.numSpecies > 0):
        Ap[2,3:,:] = u*(rhoY*h0 + rho*hY)
        
        for i in range(3,gas.numEqs):
            
            Ap[i,0,:] = Y[:,i-3] * rhop * u
            Ap[i,1,:] = Y[:,i-3] * rho
            Ap[i,2,:] = rhoT * u * Y[:,i-3]
            
            for j in range(3,gas.numEqs):
                Ap[i,j,:] = u*((i==j)*rho + Y[:,i-3]*rhoY[:,j-3])
            
    return Ap

def calc_dFluxdsolPrim(solConsL, solPrimL, solConsR, solPrimR, geoms, gas):
    
    neq = gas.numEqs
    
    rHL = solConsL[:,2] + solPrimL[:,0]
    HL = rHL/solConsL[:,0]
    
    rHR = solConsR[:,2] + solPrimR[:,0]
    HR = rHR/solConsR[:,0]
    
    # Roe Average
    rhoi = np.sqrt(solConsR[:,0]*solConsL[:,0])
    di = np.sqrt(solConsR[:,0]/solConsL[:,0])
    dl = 1/(1+di)
    
    Qp_i = solPrimL
    Qp_i[:,:neq] = (solPrimR[:,:neq]*di + solPrimL[:,:neq]*dl)
    
    Hi = (di*HR + HL) * dl
    
    if (gas.numSpecies > 1):
        Y = Qp_i[:,3:]
        massFracs = Qp_i[:,3:]
    else:
        Y = Qp_i[:,3]
        massFracs = Qp_i[:,3]
        
    Ri = calcGasConstantMixture(massFracs, gas)
    Cpi = calcCpMixture(massFracs, gas)
    gammai = Cpi/(Cpi-Ri)
    
    ci = np.sqrt(gammai*Ri*Qp_i[:,2])
    
    M_ROE = calcRoeDissipation(Qp_i, rhoi, Hi, ci, Ri, Cpi, gas)
    dFluxdQp_l = 0.5 * (calc_Ap(solPrimL, solConsL[:,0], HL, gas) + M_ROE)
    dFluxdQp_r = 0.5 * (calc_Ap(solPrimR, solConsR[:,0], HR, gas) - M_ROE)
    
    Ck = gas.muRef*Cpi/gas.Pr
    dFluxdQp_l[1,1,:] = dFluxdQp_l[1,1,:] + (4/3)*gas.muRef[0]/geoms.dx
    dFluxdQp_r[1,1,:] = dFluxdQp_r[1,1,:] - (4/3)*gas.muRef[0]/geoms.dx
    
    dFluxdQp_l[2,2,:] = dFluxdQp_l[2,2,:] + Ck[0]/geoms.dx
    dFluxdQp_r[2,2,:] = dFluxdQp_r[2,2,:] - Ck[0]/geoms.dx
    
    if (gas.numSpecies > 0):
        Cd = gas.muRef[0]*rhoi/gas.Sc[0]
        rhoCd_dx = rhoi*Cd/geoms.dx
        hY = gas.enthRefDiffs + (Qp_i[:,2]-gas.tempRef)*(gas.Cp[0]-gas.Cp[gas.numSpecies_full-1])
        
        for i in range(3,gas.numEqs):
            
            dFluxdQp_l[2,i,:] = dFluxdQp_l[2,i,:] + rhoCd_dx*hY[:,i-3]
            dFluxdQp_r[2,i,:] = dFluxdQp_r[2,i,:] - rhoCd_dx*hY[:,i-3]
            
            dFluxdQp_l[i,i,:] = dFluxdQp_l[i,i,:] + rhoCd_dx
            dFluxdQp_r[i,i,:] = dFluxdQp_r[i,i,:] - rhoCd_dx
    
    return dFluxdQp_l,dFluxdQp_r
    
def calc_dRHSdsolPrim(sol, gas, geom, params, bounds, dt_inv):
    
    
    cellidx = def_cellidx(geom)
    edgeidx = def_edgeidx(geom)
    
    
    nsamp = geom.numCells
    neq = geom.numEqs
    
    dRdQp = np.zeros((neq*nsamp,neq*nsamp))
    
    dSdQp = calc_dSourcedsolPrim(sol, gas, geom, params.dt)
    gamma_matrix = calc_dsolConsdsolPrim(sol.solCons, sol.solPrim, gas)
    
    for j in range(nsamp):
        
        idx0 = (j+1)*neq - neq
        idx1 = (cellidx.sample_in_evaluate(j))*neq - neq
        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = gamma_matrix[:,:,cellidx.sample_in_evaluate(j)-1]*dt_inv
        
        if (gas.numSpecies > 0 and hasattr('gas','nuArr') and np. count_nonzero(gas.nuArr)):
            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx0+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx0+neq-1)] - dSdQp[:,:,j]
            
        if (params.spaceOrder > 2):
            raise ValueError("Higher-Order fluxes not implemented yet")
        elif(params.spaceOrder == 2):
            [solPrimL,solConsL,solPrimR,solConsR,phi] = reconstruct_2nd(sol,bounds,geom,gas)
        else:
            solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
            solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
            solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
            solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)
            
        [dFdQp_l,dFdQp_r] = calc_dFluxdsolPrim(solConsL, solPrimL, solConsR, solPrimR, geom, gas)
        
        for j in range(geom.numNodes):
            
            iedge = edgeidx.evaluate(j)
            
            if (iedge != 1):
                
                if (params.spaceOrder == 2):
                    phi_l = np.diag(phi[edgeidx.cell_l2_edge_in_eval(j)-1,:]/4)
                    
                    if (iedge != 2):
                        idx1 = edgeidx.cell_l22_edge_in_eval(j)*neq - neq
                        
                        if (edgeidx.isr2sample(j)):
                            idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_l[:,:,j] * phi_l/geom.dx
                            
                        if (edgeidx.isl2sample(j)):
                            idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] +  dFdQp_l[:,:,j] * phi_l/geom.dx
                            
                    #left to left edge
                    idx1 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                    
                    if (edgeidx.isr2sample(j)):
                        idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_l[:,:,j] / geom.dx
                        
                    if (edgeidx.isl2sample(j)):
                        idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_l[:,:,j] / geom.dx
                        
                    #right to left edge
                    if (iedge != geom.numNodes):
                        idx1 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                        
                        if edgeidx.isr2sample(j):
                            idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_l[:,:,j] * phi_l / geom.dx
                            
                        if edgeidx.isl2sample(j):
                            idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_l[:,:,j] * phi_l / geom.dx
                            
                else:
                    
                    idx1 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                    
                    if (edgeidx.isr2sample(j)):
                        idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_l[:,:,j] / geom.dx
                        
                    if (edgeidx.isl2sample(j)):
                        idx0 = edgeidx.cell_r2_edge_in_eval(j)
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_l[:,:,j] / geom.dx
                        
            if (iedge !=geom.numNodes):
                
                if (params.spaceOrder == 2):
                    phi_r2e = np.diag(phi[edgeidx.cell_r2_edge_in_eval(j)-1,:]/4)
                    
                    #left to right edge
                    if (iedge != 1):
                        idx1 = edgeidx.cell_l2_edge_in_eval(j) * neq - neq
                        
                        if edgeidx.isr2sample(j):
                            idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_r[:,:,j] * phi_r2e / geom.dx
                            
                        if edgeidx.isl2sample(j):
                            idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_r[:,:,j] * phi_r2e/ geom.dx
                            
                        
                    #right to right edge
                    idx1 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                    if edgeidx.isr2sample(j):
                        idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_r[:,:,j] / geom.dx
                        
                    if edgeidx.isl2sample(j):
                        idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_r[:,:,j] / geom.dx
                        
                    #right right to right edge
                    if (iedge != geom.numNodes-1):
                        idx1 = edgeidx.cell_r22edge_in_evaluate(j)*neq - neq
                        
                        if (edgeidx.isr2sample(j)):
                            idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_r[:,:,j]*phi_r2e / geom.dx
                        
                        if (edgeidx.isl2sample(j)):
                            idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                            dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_r[:,:,j]*phi_r2e / geom.dx
                            
                        
                else:
                    
                    idx1 = edgeidx.cell_r2_edge_in_eval(j) * neq - neq
                    
                    if edgeidx.isr2sample(j):
                        idx0 = edgeidx.cell_l2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] + dFdQp_r[:,:,j] / geom.dx
                        
                    if edgeidx.isl2sample(j):
                        idx0 = edgeidx.cell_r2_edge_in_eval(j)*neq - neq
                        dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] = dRdQp[(idx0):(idx0+neq-1),(idx1):(idx1+neq-1)] - dFdQp_r[:,:,j] / geom.dx
                        
    return dRdQp
                    
            
                
                
                
                
                    
                        
                        
                        
                            
            
                    
                
                
                    
                    
        
            
       
            
        
        
            
            
        
    
    
            
    
    

    


