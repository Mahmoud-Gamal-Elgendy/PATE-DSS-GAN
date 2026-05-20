from .voting import PATEVoteAggregator

def __getattr__(name):
    if name == "PATEEnsembleManager":
        from .ensemble import PATEEnsembleManager
        return PATEEnsembleManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
