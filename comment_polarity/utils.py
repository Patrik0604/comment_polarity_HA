import os
import warnings

def setup_environment():
    warnings.filterwarnings("ignore")
    os.environ["WANDB_DISABLED"] = "true"
