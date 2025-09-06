import os
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant, PRVAccountant
from loss_traces.config import MODEL_DIR, STORAGE_DIR

torch.serialization.add_safe_globals([PRVAccountant])

def get_epsilon(exp_id = "test_epsilon_0.01"):
    path = os.path.join(MODEL_DIR, exp_id, "accountant.pt")

    new_acc = PRVAccountant()  # must init with same alphas
    new_acc.load_state_dict(torch.load(path, weights_only=False))


    # privacy_engine = PrivacyEngine()
    # privacy_engine.accountant = torch.load(path, weights_only=False)
    # alphas = 
    
    epsilon = new_acc.get_epsilon(delta=1e-5)

    print("Epsilon:", epsilon)


if __name__ == "__main__":
    get_epsilon()