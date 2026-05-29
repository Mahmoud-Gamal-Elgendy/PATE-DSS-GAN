from .trainer import PATEDSSGANTrainer
from .evaluation import Evaluator

# TrainConfig now lives in src.config; re-export for backward compatibility
from ..config import TrainConfig
