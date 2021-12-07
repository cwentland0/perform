import numpy as np


class ReactionModel:
    """Base class for all reaction models.

    All child classes must implement the following member functions:

    * calc_source()
    * calc_jacob_prim()

    Refer to finite_rate_irrev_reaction.py for examples.
    """

    def __init__(self):

        pass

    def calc_reaction(self, sol, dt, samp_idxs=np.s_[:]):

        # calculate source terms
        reaction_source, wf = self.calc_source(sol, dt, samp_idxs=samp_idxs)

        # calculate unsteady heat release
        heat_release = -np.sum(reaction_source * sol.hi[:, samp_idxs], axis=0)

        return reaction_source, wf, heat_release
