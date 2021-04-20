class ReactionModel:
    """Base class for all reaction models.

    All child classes must implement the following member functions:

    * calc_source()
    * calc_jacob_prim()

    Refer to finite_rate_irrev_reaction.py for examples.
    """

    def __init__(self):

        pass
