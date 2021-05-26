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
            self.residualSampleIdx       =   []

            self.trueSolution = []
            self.trueCount = 50

        else:
            raise ValueError("Invalid selection for adaptive ROM type")

    def gatherStaleCons(self, romDomain, solDomain, solver, model):
        '''Gathers the conservative state from the stale file'''

        #solDomain.solHistCons is analogous to Q

        assert(romDomain.staleSnapshots.shape[1] == solver.mesh.numCells), 'resolution mis-match between stale states and current simulation'

        romDomain.timeIntegrator.staleStatetimeOrder = min(romDomain.timeIntegrator.timeOrder, romDomain.staleSnapshots.shape[-1])
        solDomain.timeIntegrator.staleStatetimeOrder =  romDomain.timeIntegrator.staleStatetimeOrder

        self.FWindow = np.zeros((model.numVars, solver.mesh.numCells, self.adaptROMWindowSize))

        if self.adaptROMWindowSize != 1 :
            self.FWindow[:, :, :-1] = romDomain.staleSnapshots[model.varIdxs, :, -(self.adaptROMWindowSize-1):]

        for timeIdx in range(1, romDomain.timeIntegrator.staleStatetimeOrder+1):
            solDomain.solInt.solHistCons[timeIdx] = romDomain.staleSnapshots[:, :, -timeIdx].copy()

        solDomain.solInt.solHistCons[0] = solDomain.solInt.solHistCons[1].copy()
        self.trueSolution = np.load('/Users/sahilbhola/Documents/CASLAB/perform/examples/testCase/UnsteadyFieldResults/True_cons.npy')


    def HistoryInitialization(self, romDomain, solDomain, solver, model):
        '''Computes the coded state and the reconstructed state for initializing the sub-iteration'''

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder)  # updating time order if stale states are available

        for timeIdx in range(timeOrder+1):
            #Updating coded history
            solCons = solDomain.solInt.solHistCons[timeIdx].copy()

            solCons = model.standardizeData(solCons[model.varIdxs, :], normalize=True, normFacProf=model.normFacProfCons, normSubProf=model.normSubProfCons,
									   center=True, centProf=model.centProfCons, inverse=False)

            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True)

        model.code = model.codeHist[0].copy()   #Coded guess
        model.updateSol(solDomain)              #Update reconstructed state


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

            if (solver.timeIter == 1 or solver.timeIter % self.adaptROMResidualSampStep == 0):

                self.trueCount += 1     #Index of current predict

                self.FWindow[:, :, -1] = self.previousStateEstimate(romDomain, solDomain, solver, model)
                # self.FWindow[:, :, -1] = self.trueSolution[:, :, self.trueCount-1]      #Using FOM

                #adaption window
                adaptionWindow          = self.FWindow.reshape(-1, self.adaptROMWindowSize, order = 'C').copy()
                scaledAdaptionWindow    = (adaptionWindow - model.centProfCons.ravel(order = 'C')[:, None] - model.normSubProfCons.ravel(order = 'C')[:, None]) / model.normFacProfCons.ravel(order = 'C')[:, None]

                #residual
                reconstructedWindow = model.trialBasis @ np.linalg.pinv(model.trialBasis) @ scaledAdaptionWindow
                residual = scaledAdaptionWindow - reconstructedWindow

            else:
                raise ValueError('Reduced frequency residual update not implemented (AADEIM)')

            self.ADEIM(model, scaledAdaptionWindow)

            self.FWindow[:, :, :-1] = self.FWindow[:, :, 1:]

    def ADEIM(self, model, scaledAdaptionWindow):
        #Computing C matrix
        C = np.linalg.pinv(model.trialBasis) @ scaledAdaptionWindow
        pinvCtranspose = np.linalg.pinv(C.T)

        #Computing residual
        R = model.trialBasis @ C - scaledAdaptionWindow

        #Computing SVD
        _, singValues, rightBasis_h = np.linalg.svd(R)
        rightBasis = rightBasis_h.T

        rank = min(self.adaptROMUpdateRank, len(singValues))

        OldBasis = model.trialBasis.copy()

        #Basis Update
        for irank in range(rank):
            alpha = -R @ rightBasis[:, irank]
            beta  = pinvCtranspose @ rightBasis[:, irank]
            update = alpha[:, None] @ beta[None, :]
            OldBasis = OldBasis + update

        model.trialBasis = orth(OldBasis)


    def previousStateEstimate(self, romDomain, solDomain, solver, model):

        solInt = solDomain.solInt

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder)  # updating time order if stale states are available

        coeffs = romDomain.timeIntegrator.coeffs[timeOrder - 1]

        state = coeffs[0] * solInt.solHistCons[0][model.varIdxs, :].copy()

        for iterIdx in range(2, timeOrder + 1):
            state += coeffs[iterIdx] * solInt.solHistCons[iterIdx][model.varIdxs, :].copy()

        state = (- state + romDomain.timeIntegrator.dt*solInt.RHS[model.varIdxs, :]) / coeffs[1]

        return state

