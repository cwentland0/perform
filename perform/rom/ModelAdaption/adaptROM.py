import numpy as np
from scipy.linalg import orth
import copy


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
            raise ValueError('AADEIM is broken')
            self.adaptROMResidualSampStep   =   romDomain.adaptROMResidualSampStep
            self.adaptROMnumResSamp         =   romDomain.adaptROMnumResSamp
            self.adaptROMWindowSize         =   romDomain.adaptROMWindowSize
            self.adaptROMUpdateRank         =   romDomain.adaptROMUpdateRank

            self.solConsWindow      =   []  #For adaption window    -   relates to AADEIM 'averaging' (stupid name)
            self.predConsWindow     =   []  #For time keeping states (used for previous state prediction)   - relates to the 'projected solution (without any effect from Basis update)'
            self.residualSampleIdx  =   []

        else:
            raise ValueError("Invalid selection for adaptive ROM type")

    def gatherStaleCons(self, romDomain, solDomain, solver, model):

        assert(solver.mesh.numCells == romDomain.staleCons.shape[1]), 'resolution mis-match between stale state and current run...'

        self.solConsWindow  =   np.zeros((model.numVars, solver.mesh.numCells, self.adaptROMWindowSize+1))
        self.predConsWindow =   np.zeros((model.numVars, solver.mesh.numCells, romDomain.timeIntegrator.timeOrder+1))

        romDomain.timeIntegrator.timeOrderStaleState = min(romDomain.timeIntegrator.timeOrder, romDomain.staleCons.shape[-1])

        self.solConsWindow[:, :, :self.adaptROMWindowSize]  =   romDomain.staleCons[model.varIdxs, :, -self.adaptROMWindowSize:]

        self.predConsWindow[:, :, 1:] = np.flip(romDomain.staleCons[model.varIdxs, :, -romDomain.timeIntegrator.timeOrderStaleState:], axis = -1)
        self.predConsWindow[:, :, 0] = self.predConsWindow[:, :, 1]


        for timeIdx in range(1, romDomain.timeIntegrator.timeOrderStaleState+1):

            model.codeHist[timeIdx] = model.projectedState(romDomain.staleCons[:, :, -timeIdx], romDomain)
            reconState              = solDomain.solInt.solHistCons[timeIdx].copy()

            reconState[model.varIdxs, :]             = model.decodeSol(model.codeHist[timeIdx])
            solDomain.solInt.solHistCons[timeIdx]    = reconState.copy()

        model.codeHist[0] = model.codeHist[1].copy()
        solDomain.solInt.solHistCons[0] = solDomain.solInt.solHistCons[1].copy()
        model.code = model.codeHist[0].copy()


    def adaptModel(self, romDomain, solDomain, solver, model):

        if romDomain.adaptiveROMMethod == "OSAB":
            if romDomain.timeIntegrator.timeType == "implicit" : raise ValueError('One step adaptive basis not implemented for implicit framework')

            # raise ValueError('Ordering was recently changed by Chris, check OSAB')
            self.adaptionResidual = (self.trueStandardizedState - model.applyTrialBasis(model.code)).flatten(order = "C").reshape(-1, 1)
            self.basisUpdate = np.dot(self.adaptionResidual, model.code.reshape(1, -1)) / np.linalg.norm(model.code)**2

            model.trialBasis = model.trialBasis + self.basisUpdate

            model.updateSol(solDomain)

            solDomain.solInt.updateState(fromCons=True)

        elif romDomain.adaptiveROMMethod == "AADEIM":

            if not romDomain.timeIntegrator.timeType == "implicit" : raise ValueError ('AADEIM not implemented for explicit framework')
            if romDomain.hyperReduc: raise ValueError('AADEIM not currently hyper-reduction friendly (Check back soon...)')
            solInt = solDomain.solInt

            self.predConsWindow[:, :, 0] = solInt.solCons[model.varIdxs, :]

            if (solver.timeIter == 1 or solver.timeIter%self.adaptROMResidualSampStep ==0):

                #Previous state estimate
                self.solConsWindow[:, :, -1] = self.previousStateEstimate(solInt.RHS, romDomain, solver, model)

                #adaption window
                adaptionWindow             = self.solConsWindow[:, :, -self.adaptROMWindowSize:]
                scaledAdaptionWindow       = (adaptionWindow - model.centProfCons[:, :, np.newaxis] - model.normSubProfCons[:, :, np.newaxis]) / model.normFacProfCons[:, :, np.newaxis]
                InterpolatedAdaptionWindow = scaledAdaptionWindow[:, solDomain.directSampIdxs, :]

                #Interpolated Basis
                InterpolatedBasis = (model.trialBasis.reshape(model.numVars, -1, model.latentDim, order = 'C')[:, solDomain.directSampIdxs, :]).reshape(-1, model.latentDim, order = 'C')

                #residual window
                reconstructedWindow  = model.trialBasis @ np.linalg.pinv(InterpolatedBasis) @ InterpolatedAdaptionWindow.reshape(-1, self.adaptROMWindowSize, order = 'C')
                residualWindow       = scaledAdaptionWindow - reconstructedWindow.reshape(model.numVars, -1, self.adaptROMWindowSize, order = 'C')

                #sampling points
                self.residualSampleIdx = np.unique(np.argsort(np.sum(np.square(residualWindow), axis = 2), axis = 1)[:, -self.adaptROMnumResSamp:].ravel())

            else:
                raise ValueError('Reduced frequency residual update not implemented (AADEIM)')

            #Basis update
            sampledAdaptionWindow = scaledAdaptionWindow[:, self.residualSampleIdx, :]
            sampledBasis = (model.trialBasis.reshape((model.numVars, -1, model.latentDim), order='C')[:, self.residualSampleIdx,:]).reshape(-1, model.latentDim, order='C')

            self.ADEIM(model, InterpolatedAdaptionWindow, InterpolatedBasis, sampledAdaptionWindow, sampledBasis)

            #update coded and reconstructed states in (Actual) history and recalibration...
            self.recomputeReconSolnHistory(solDomain, romDomain, model)

            #Window updates
            self.predConsWindow[:, :, 1:] = self.predConsWindow[:, :, :-1]
            self.solConsWindow[:, :, :-1] = self.solConsWindow[:, :, 1:]

        else:
            raise ValueError("Invalid selection for adaptive ROM type")

    def ADEIM(self, model, InterpolatedAdaptionWindow, InterpolatedBasis, sampledAdaptionWindow, sampledBasis):
        #TODO: Hyperreduction points update

        BasisCoefficients   =   np.linalg.pinv(InterpolatedBasis) @ InterpolatedAdaptionWindow.reshape(-1, self.adaptROMWindowSize, order = 'C')
        sampledResidual     =   sampledBasis @  BasisCoefficients  - sampledAdaptionWindow.reshape(-1, self.adaptROMWindowSize, order = 'C')
        BasisCoefficientsPinv       =   np.linalg.pinv(BasisCoefficients.T)

        _, singValues, RightBasis_h =   np.linalg.svd(sampledResidual, full_matrices=False)
        RightBasis = RightBasis_h.T
        rank = min([self.adaptROMUpdateRank, singValues.shape[0]])

        for nRank in range(rank):
            alpha           =   -sampledResidual @ RightBasis[:, nRank]
            beta            =   BasisCoefficientsPinv @ RightBasis[:, nRank]
            basiUpdate      =   (alpha[:, None] @ beta[None, :]).reshape((-1, len(self.residualSampleIdx), model.latentDim), order = 'C')
            UpdatedBasis    =   np.copy(model.trialBasis).reshape(model.numVars, -1, model.latentDim, order = 'C')
            UpdatedBasis[:, self.residualSampleIdx, :] = UpdatedBasis[:, self.residualSampleIdx, :] + basiUpdate
            model.trialBasis = orth(UpdatedBasis.reshape(-1, model.latentDim, order = 'C'))

    def previousStateEstimate(self, rhs, romDomain, solver, model):

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.timeOrderStaleState, timeOrder) #time order update if stale states are available

        coeffs = romDomain.timeIntegrator.coeffs[timeOrder - 1]

        state = coeffs[0] * self.predConsWindow[:, :, 0]
        for iterIdx in range(2, timeOrder + 1):
            state += coeffs[iterIdx] * self.predConsWindow[:, :, iterIdx]

        state = (- state + romDomain.timeIntegrator.dt*rhs[model.varIdxs, :]) / coeffs[1]

        return state

    def recomputeReconSolnHistory(self, solDomain, romDomain, model):

        for timeIdx in range(romDomain.timeIntegrator.timeOrder):
            solCons = self.predConsWindow[:, :, timeIdx].copy()
            solCons = model.standardizeData(solCons, normalize=True,
                                           normFacProf=model.normFacProfCons, normSubProf=model.normSubProfCons,
                                           center=True, centProf=model.centProfCons, inverse=False)

            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True)
            reconState = solDomain.solInt.solHistCons[timeIdx].copy()

            reconState[model.varIdxs, :]             = model.decodeSol(model.codeHist[timeIdx])
            solDomain.solInt.solHistCons[timeIdx]    = reconState.copy()

        model.code = model.codeHist[0].copy()
