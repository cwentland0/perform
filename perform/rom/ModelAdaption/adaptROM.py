import numpy as np
from scipy.linalg import orth
import pdb


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

            self.adaptROMResidualSampStep   =   romDomain.adaptROMResidualSampStep  #specifies the number of step after which full residual is computed
            self.adaptROMnumResSamp         =   romDomain.adaptROMnumResSamp        #specifies the number of samples of the residual
            self.adaptROMWindowSize         =   romDomain.adaptROMWindowSize        #look-back widow size
            self.adaptROMUpdateRank         =   romDomain.adaptROMUpdateRank        #basis upadate rank
            self.adaptROMInitialSnap        =   romDomain.adaptROMInitialSnap       #specifies the "number" of samples available (same as the number of samples used for basis computation)

            self.FWindow                    =   np.zeros((model.numVars, solver.mesh.numCells, self.adaptROMWindowSize)) #look-back window
            self.residualSampleIdx          =   [] #residual sample indexes (indexes fron all the states will be pooled together)

            self.interPolAdaptionWindow     =   []
            self.interPolBasis              =   []
            self.scaledAdaptionWindow       =   np.zeros((model.numVars*solver.mesh.numCells, self.adaptROMWindowSize))


            assert(self.adaptROMWindowSize-1<=self.adaptROMInitialSnap), 'Look back window size minus 1 should be less than equal to the available stale states'

        else:
            raise ValueError("Invalid selection for adaptive ROM type")

    def initializeLookBackWindow(self, romDomain, model):

        self.FWindow[:, :, :-1] = romDomain.staleConsSnapshots[model.varIdxs, :, -(self.adaptROMWindowSize-1):]

    def initializeHistory(self, romDomain, solDomain, solver, model):
        '''Computes the coded state and the reconstructed state for initializing the sub-iteration'''

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder)  # updating time order if stale states are available

        for timeIdx in range(timeOrder + 1):
            #update the coded history
            solCons = solDomain.solInt.solHistCons[timeIdx].copy()
            solCons = model.standardizeData(solCons[model.varIdxs, :], normalize=True,
                                            normFacProf=model.normFacProfCons, normSubProf=model.normSubProfCons,
            								   center=True, centProf=model.centProfCons, inverse=False)
            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True)

        model.code = model.codeHist[0].copy()
        model.updateSol(solDomain)


    def gatherSamplePoints(self, romDomain, solDomain, solver, model):

        if not romDomain.timeIntegrator.timeType == "implicit": raise ValueError('AADEIM not implemented for explicit framework')

        if (solver.timeIter == 1 or solver.timeIter % self.adaptROMResidualSampStep == 0):

            self.FWindow[:, :, -1] = self.previousStateEstimate(romDomain, solDomain, solver, model)

            # adaption window
            adaptionWindow = self.FWindow.reshape(-1, self.adaptROMWindowSize, order='C')
            self.scaledAdaptionWindow = (adaptionWindow - model.centProfCons.ravel(order='C')[:,None] - model.normSubProfCons.ravel(order='C')[:,None]) / model.normFacProfCons.ravel(order='C')[:, None]

            self.interPolAdaptionWindow = (self.scaledAdaptionWindow.reshape(model.numVars, solver.mesh.numCells, -1, order = "C")[:,solDomain.directSampIdxs,:]).reshape(-1, self.adaptROMWindowSize, order = 'C')
            self.interPolBasis          = (model.trialBasis.reshape(model.numVars, -1, model.latentDim)[:,solDomain.directSampIdxs,:]).reshape(-1, model.latentDim, order = "C")

            # residual
            reconstructedWindow = model.trialBasis @ np.linalg.pinv(self.interPolBasis) @ self.interPolAdaptionWindow
            residual = (self.scaledAdaptionWindow - reconstructedWindow).reshape((model.numVars, solver.mesh.numCells, -1), order='C')
            SampleIdx = np.unique((np.argsort(np.sum(residual ** 2, axis=2), axis=1)[:, -self.adaptROMnumResSamp:]).ravel())

            romDomain.residualCombinedSampleIdx = np.unique(np.append(romDomain.residualCombinedSampleIdx, SampleIdx).astype(int))

        else:
            raise ValueError('Reduced frequency residual update not implemented (modified - AADEIM)')

    def adaptModel(self, romDomain, solDomain, solver, model):

        if romDomain.adaptiveROMMethod == "OSAB":
            if romDomain.timeIntegrator.timeType == "implicit" : raise ValueError('One step adaptive basis not implemented for implicit framework')

            self.adaptionResidual = (self.trueStandardizedState - model.applyTrialBasis(model.code)).flatten(order = "C").reshape(-1, 1)
            self.basisUpdate = np.dot(self.adaptionResidual, model.code.reshape(1, -1)) / np.linalg.norm(model.code)**2

            model.trialBasis = model.trialBasis + self.basisUpdate

            model.updateSol(solDomain)

            solDomain.solInt.updateState(fromCons=True)

        elif romDomain.adaptiveROMMethod == "AADEIM":
            self.residualSampleIdx = romDomain.residualCombinedSampleIdx
            reshapedWindow = (self.scaledAdaptionWindow).reshape(model.numVars, -1,  self.adaptROMWindowSize, order = "C")
            sampledAdaptionWindow         = (reshapedWindow[:,self.residualSampleIdx,:]).reshape(-1, self.adaptROMWindowSize, order = "C")
            sampledBasis                  = (model.trialBasis.reshape(model.numVars, -1, model.latentDim)[:,self.residualSampleIdx,:]).reshape(-1, model.latentDim, order = "C")

            #Computing coefficient matrix
            CMat = np.linalg.pinv(self.interPolBasis) @ self.interPolAdaptionWindow
            pinvCtranspose = np.linalg.pinv(CMat.T)

            #Computing residual
            R = sampledBasis @ CMat - sampledAdaptionWindow

            # Computing SVD
            _, singValues, rightBasis_h = np.linalg.svd(R)
            rightBasis = rightBasis_h.T

            rank = min(self.adaptROMUpdateRank, len(singValues))

            # Basis Update
            idx = (np.arange(model.numVars*solver.mesh.numCells).reshape(model.numVars, -1, order = "C")[:, self.residualSampleIdx]).ravel(order = "C")
            for irank in range(rank):
                alpha = -R @ rightBasis[:, irank]
                beta = pinvCtranspose @ rightBasis[:, irank]
                update = alpha[:, None] @ beta[None, :]
                # model.trialBasis[idx, :] +=  update

            # model.trialBasis = orth( model.trialBasis)

            self.FWindow[:, :, :-1] = self.FWindow[:, :, 1:]

    def previousStateEstimate(self, romDomain, solDomain, solver, model):

        solInt = solDomain.solInt

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder)  # updating time order if stale states are available

        coeffs = romDomain.timeIntegrator.coeffs[timeOrder - 1]

        state = coeffs[0] * solInt.solHistCons[0][model.varIdxs, :].copy()

        for iterIdx in range(2, timeOrder + 1):
            state += coeffs[iterIdx] * solInt.solHistCons[iterIdx][model.varIdxs, :].copy()

        PreviousState = (- state + (romDomain.timeIntegrator.dt*solInt.RHS[model.varIdxs, :])) / coeffs[1]

        return PreviousState

