from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qllm_learner import QLLMLearner
from .qllm_learner_time import QLLMLearnerTime

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qllm_learner"] = QLLMLearner
REGISTRY["qllm_learner_time"] = QLLMLearnerTime