import numpy as np
from scipy.linalg import orth
import copy
import pdb
import matplotlib.pyplot as plt


class adaptROM():
    def __init__(self, model, solver, romDomain):

        self.adaptiveROMMethod = romDomain.adaptiveROMMethod
        self.adaptsubIteration   = False

        if self.adaptiveROMMethod == "OSAB":
            """
            Method developed by Prof. Karthik Duraisamy (University of Michigan)
            """
            #TODO: Implement residual sampling step

            self.adaptsubIteration              =   True
            self.trueStandardizedState          =   np.zeros((model.numVars, solver.mesh.numCells))
            self.adaptionResidual               =   np.zeros((model.numVars*solver.mesh.numCells))
            self.basisUpdate                    =   np.zeros(model.trialBasis.shape)

            self.adaptROMResidualSampStep       =   romDomain.adaptROMResidualSampStep
            self.adaptROMnumResSamp             =   romDomain.adaptROMnumResSamp

        elif self.adaptiveROMMethod == "AADEIM":
            """
            Method developed by Prof. Benjamin Peherstorfer (NYU)
            """

            self.adaptROMResidualSampStep   =   romDomain.adaptROMResidualSampStep
            self.adaptROMnumResSamp         =   romDomain.adaptROMnumResSamp
            self.adaptROMWindowSize         =   romDomain.adaptROMWindowSize
            self.adaptROMUpdateRank         =   romDomain.adaptROMUpdateRank

            self.FWindow                 =   []
            self.QWindow                 =   []
            self.residualSampleIdx       =   []


        else:
            raise ValueError("Invalid selection for adaptive ROM type")

    def gatherStaleCons(self, romDomain, solDomain, solver, model):

        assert(romDomain.staleSnapshots.shape[1] == solver.mesh.numCells), 'resolution mis-match between stale states and current simulation'

        romDomain.timeIntegrator.staleStatetimeOrder = min(romDomain.timeIntegrator.timeOrder, romDomain.staleSnapshots.shape[-1])
        solDomain.timeIntegrator.staleStatetimeOrder =  romDomain.timeIntegrator.staleStatetimeOrder

        self.FWindow = np.zeros((model.numVars, solver.mesh.numCells, self.adaptROMWindowSize))
        self.QWindow = np.zeros((model.numVars, solver.mesh.numCells, romDomain.timeIntegrator.timeOrder+1))

        if self.adaptROMWindowSize != 1 :
            self.FWindow[:, :, :self.adaptROMWindowSize-1] = romDomain.staleSnapshots[model.varIdxs, :, -self.adaptROMWindowSize+1 :]

        self.QWindow[:, :, 1:romDomain.timeIntegrator.staleStatetimeOrder+1] = np.flip(romDomain.staleSnapshots[model.varIdxs, :, -romDomain.timeIntegrator.staleStatetimeOrder:], axis = 2)
        self.QWindow[:, :, 0] = self.QWindow[:, :, 1].copy()


    def HistoryInitialization(self, romDomain, solDomain, solver, model):
    #     #TODO: currently solHistPrim is not being updated. Required if solving based on primitive variables...

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder) #time order update if stale states are available


        for timeIdx in range(1, timeOrder + 1):
            # reading available solution
            solCons = self.QWindow[:, :, timeIdx].copy()

            # standardizing solution
            solCons = model.standardizeData(solCons, normalize=True, normFacProf=model.normFacProfCons,
                                            normSubProf=model.normSubProfCons,
                                            center=True, centProf=model.centProfCons, inverse=False)

            # coded solution History update
            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True).copy()

            # reconstruction update
            reconSolution = solDomain.solInt.solHistCons[timeIdx].copy()
            reconSolution[model.varIdxs, :] = model.decodeSol(model.codeHist[timeIdx])
            solDomain.solInt.solHistCons[timeIdx] = reconSolution.copy()

        # updating current coded solution guess
        model.codeHist[0] = model.codeHist[1].copy()
        model.code = model.codeHist[0].copy()

        # updating current solution guess
        solDomain.solInt.solHistCons[0] = solDomain.solInt.solHistCons[1].copy()
        solDomain.solInt.solCons = solDomain.solInt.solHistCons[0].copy()



    def adaptModel(self, romDomain, solDomain, solver, model):

        if romDomain.adaptiveROMMethod == "OSAB":
            if romDomain.timeIntegrator.timeType == "implicit" : raise ValueError('One step adaptive basis not implemented for implicit framework')

            self.adaptionResidual = (self.trueStandardizedState - model.applyTrialBasis(model.code)).flatten(order = "C").reshape(-1, 1)
            self.basisUpdate = np.dot(self.adaptionResidual, model.code.reshape(1, -1)) / np.linalg.norm(model.code)**2

            model.trialBasis = model.trialBasis + self.basisUpdate

            model.updateSol(solDomain)

            solDomain.solInt.updateState(fromCons=True)

        elif romDomain.adaptiveROMMethod == "AADEIM":

            if not romDomain.timeIntegrator.timeType == "implicit" : raise ValueError ('AADEIM not implemented for explicit framework')
            if romDomain.hyperReduc: raise ValueError('AADEIM not currently hyper-reduction friendly (Check back soon...)')

            solInt = solDomain.solInt

            self.QWindow[:, :, 0] = solInt.solCons[model.varIdxs, :]

            if (solver.timeIter == 1 or solver.timeIter%self.adaptROMResidualSampStep ==0):

                #Previous state estimate
                self.FWindow[:, :, -1] = self.previousStateEstimate(solInt.RHS, romDomain, solver, model)

                #adaption window
                adaptionWindow = self.FWindow.reshape(-1, self.adaptROMWindowSize, order = 'C').copy()
                scaledAdaptionWindow = (adaptionWindow - model.centProfCons.ravel(order = 'C')[:, None] - model.normSubProfCons.ravel(order = 'C')[:, None]) / model.normFacProfCons.ravel(order = 'C')[:, None]

                #residual
                reconstructedWindow = model.trialBasis @ np.linalg.pinv(model.trialBasis) @ scaledAdaptionWindow
                residual = scaledAdaptionWindow - reconstructedWindow


            else:
                raise ValueError('Reduced frequency residual update not implemented (AADEIM)')

            self.ADEIM(model, scaledAdaptionWindow)

            self.QWindow[:, :, 1:] = self.QWindow[:, :, :-1]
            self.FWindow[:, :, :-1] = self.FWindow[:, :, 1:]

        else:
            raise ValueError("Invalid selection for adaptive ROM type")


    def ADEIM(self, model, scaledAdaptionWindow):
        #Computing C matrix
        C = np.linalg.pinv(model.trialBasis) @ scaledAdaptionWindow

        #Computing residual
        R = model.trialBasis @ C - scaledAdaptionWindow

        #Computing SVD
        _, singValues, rightBasis_h = np.linalg.svd(R, full_matrices=False)
        rightBasis = rightBasis_h.T

        #Extras
        pinvCtranspose = np.linalg.pinv(C.T)
        rank = min(self.adaptROMUpdateRank, len(singValues))

        #Basis Update
        for irank in range(rank):
            alpha = -R @ rightBasis[:, irank]
            beta  = pinvCtranspose @ rightBasis[:, irank]

            model.trialBasis = model.trialBasis + alpha[:, None] @ beta[None, :]

        model.trialBasis = orth(model.trialBasis)



    def previousStateEstimate(self, rhs, romDomain, solver, model):

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder) #time order update if stale states are available

        coeffs = romDomain.timeIntegrator.coeffs[timeOrder - 1]

        state = coeffs[0] * self.QWindow[:, :, 0]

        for iterIdx in range(2, timeOrder + 1):
            state += coeffs[iterIdx] * self.QWindow[:, :, iterIdx]

        state = (- state + romDomain.timeIntegrator.dt*rhs[model.varIdxs, :]) / coeffs[1]

        return state

