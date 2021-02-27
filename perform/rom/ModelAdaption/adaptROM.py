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

        assert(romDomain.staleSnaphots.shape[1] == solver.mesh.numCells), 'resolution mis-match between stale states and current simulation'
        romDomain.timeIntegrator.staleStatetimeOrder = min(romDomain.timeIntegrator.timeOrder, romDomain.staleSnaphots.shape[-1])
        solDomain.timeIntegrator.staleStatetimeOrder =  romDomain.timeIntegrator.staleStatetimeOrder
        self.FWindow = np.zeros((model.numVars, solver.mesh.numCells, self.adaptROMWindowSize+1))
        self.QWindow = np.zeros((model.numVars, solver.mesh.numCells, romDomain.timeIntegrator.timeOrder+1))

        self.FWindow[:, :, :-1]   = romDomain.staleSnaphots[model.varIdxs, :, -self.adaptROMWindowSize:]
        self.QWindow[:, :, 1:]    = np.flip(romDomain.staleSnaphots[model.varIdxs, :, -romDomain.timeIntegrator.staleStatetimeOrder:], axis = 2)
        self.QWindow[:, :, 0]     = self.QWindow[:, :, 1]

        #updating history of coded solution and reconstructed solution
        for timeIdx in range(romDomain.timeIntegrator.staleStatetimeOrder+1):
            solCons = self.QWindow[:, :, timeIdx].copy()
            solCons = model.standardizeData(solCons, normalize=True, normFacProf=model.normFacProfCons, normSubProf=model.normSubProfCons,
									   center=True, centProf=model.centProfCons, inverse=False)

            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True)

            reconstructedSolution   = solDomain.solInt.solHistCons[timeIdx].copy()
            reconstructedSolution[model.varIdxs, :] = model.decodeSol(model.codeHist[timeIdx])
            solDomain.solInt.solHistCons[timeIdx]   = reconstructedSolution.copy()

        model.code = model.codeHist[0].copy()



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
                adaptionWindow             = self.FWindow[:, :, -self.adaptROMWindowSize:]
                scaledAdaptionWindow       = (adaptionWindow - model.centProfCons[:, :, np.newaxis] - model.normSubProfCons[:, :, np.newaxis]) / model.normFacProfCons[:, :, np.newaxis]
                interpolatedAdaptionWindow = scaledAdaptionWindow[:, solDomain.directSampIdxs, :]

                #Interpolated Basis
                interpolatedBasis = (model.trialBasis.reshape(model.numVars, -1, model.latentDim, order = 'C')[:, solDomain.directSampIdxs, :]).reshape(-1, model.latentDim, order = 'C')

                #residual window
                reconstructedWindow  = model.trialBasis @ np.linalg.pinv(interpolatedBasis) @ interpolatedAdaptionWindow.reshape(-1, self.adaptROMWindowSize, order = 'C')
                residualWindow       = scaledAdaptionWindow - reconstructedWindow.reshape(model.numVars, -1, self.adaptROMWindowSize, order = 'C')

                #sampling points
                self.residualSampleIdx = np.unique(np.argsort(np.sum(np.square(residualWindow), axis = 2), axis = 1)[:, -self.adaptROMnumResSamp:].ravel())

            else:
                raise ValueError('Reduced frequency residual update not implemented (AADEIM)')

            #Basis update
            sampledAdaptionWindow = scaledAdaptionWindow[:, self.residualSampleIdx, :]
            sampledBasis = (model.trialBasis.reshape((model.numVars, -1, model.latentDim), order='C')[:, self.residualSampleIdx,:]).reshape(-1, model.latentDim, order='C')

            self.ADEIM(model, interpolatedAdaptionWindow, interpolatedBasis, sampledAdaptionWindow, sampledBasis)

            self.recomputeReconSolnHistory(solDomain, romDomain, model)

            #Window updates
            self.QWindow[:, :, 1:] = self.QWindow[:, :, :-1]
            self.FWindow[:, :, :-1] = self.FWindow[:, :, 1:]

        else:
            raise ValueError("Invalid selection for adaptive ROM type")

    def ADEIM(self, model, interpolatedAdaptionWindow, interpolatedBasis, sampledAdaptionWindow, sampledBasis):
        #TODO: Hyperreduction points update

        basisCoefficients   =   np.linalg.pinv(interpolatedBasis) @ interpolatedAdaptionWindow.reshape(-1, self.adaptROMWindowSize, order = 'C')
        sampledResidual     =   sampledBasis @  basisCoefficients  - sampledAdaptionWindow.reshape(-1, self.adaptROMWindowSize, order = 'C')
        basisCoefficientsPinv       =   np.linalg.pinv(basisCoefficients.T)

        _, singValues, rightBasis_h =   np.linalg.svd(sampledResidual, full_matrices=False)
        rightBasis = rightBasis_h.T
        rank = min([self.adaptROMUpdateRank, singValues.shape[0]])

        UpdatedBasis = np.copy(model.trialBasis).reshape(model.numVars, -1, model.latentDim, order='C')

        for nRank in range(rank):
            alpha            =   -sampledResidual @ rightBasis[:, nRank]
            beta             =   basisCoefficientsPinv @ rightBasis[:, nRank]
            basisUpdate      =   (alpha[:, None] @ beta[None, :]).reshape((-1, len(self.residualSampleIdx), model.latentDim), order = 'C')
            UpdatedBasis[:, self.residualSampleIdx, :] = UpdatedBasis[:, self.residualSampleIdx, :] + basisUpdate

        model.trialBasis = orth(UpdatedBasis.reshape(-1, model.latentDim, order = 'C'))


    def previousStateEstimate(self, rhs, romDomain, solver, model):

        timeOrder = min(solver.iter, romDomain.timeIntegrator.timeOrder)  # cold start
        timeOrder = max(romDomain.timeIntegrator.staleStatetimeOrder, timeOrder) #time order update if stale states are available

        coeffs = romDomain.timeIntegrator.coeffs[timeOrder - 1]

        state = coeffs[0] * self.QWindow[:, :, 0]
        for iterIdx in range(2, timeOrder + 1):
            state += coeffs[iterIdx] * self.QWindow[:, :, iterIdx]

        state = (- state + romDomain.timeIntegrator.dt*rhs[model.varIdxs, :]) / coeffs[1]

        return state

    def recomputeReconSolnHistory(self, solDomain, romDomain, model):


        for timeIdx in range(romDomain.timeIntegrator.timeOrder):
            solCons = self.QWindow[:, :, timeIdx].copy()
            solCons = model.standardizeData(solCons, normalize=True,
                                           normFacProf=model.normFacProfCons, normSubProf=model.normSubProfCons,
                                           center=True, centProf=model.centProfCons, inverse=False)

            model.codeHist[timeIdx] = model.projectToLowDim(model.trialBasis, solCons, transpose=True)

            reconstructedSolution   = solDomain.solInt.solHistCons[timeIdx].copy()
            reconstructedSolution[model.varIdxs, :] = model.decodeSol(model.codeHist[timeIdx])
            solDomain.solInt.solHistCons[timeIdx]   = reconstructedSolution.copy()

        model.code = model.codeHist[0].copy()

