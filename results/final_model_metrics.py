import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import shap
import numpy as np
import pandas as pd

def _loss(model: nn.Module, data_loader: DataLoader):
    device = next(model.parameters()).device

    ret = {
        "loss": [],
        "confidence": []
    }

    with torch.no_grad():
        for inputs, targets, _indices in tqdm(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction="none")
            ret["loss"].extend(loss.tolist())

            probs = F.softmax(outputs, dim=1)
            log_odds = torch.log(probs / (1 - probs))

            top2_values, _ = torch.topk(log_odds, k=2, dim=1)
            confs = top2_values[:, 0] - top2_values[:, 1]
            ret["confidence"].extend(confs.tolist())

    return ret

def _grads(model: nn.Module, data_loader: DataLoader):
    device = next(model.parameters()).device

    ret = {
        "param_grad_norm": [],
        "param_grad_var": [],
        "input_grad_norm": [],
        "input_grad_var": [],
    }


    for inputs, targets, _indices in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad=True
        if inputs.grad is not None:
            inputs.grad.zero_()

        # naively computing per sample gradients
        for i in range(len(inputs)):
            model.zero_grad()
            sample = inputs[i].unsqueeze(0)
            label = targets[i].unsqueeze(0)

            output = model(sample)
            loss = F.cross_entropy(output, label)
            loss.backward()

            param_grads = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
            param_grads = torch.cat(param_grads)

            ret["param_grad_norm"].append(torch.norm(param_grads, p=2).item())
            ret["param_grad_var"].append(torch.var(param_grads).item())

            input_grads = inputs.grad[i].view(-1)

            ret["input_grad_norm"].append(torch.norm(input_grads, p=2).item())
            ret["input_grad_var"].append(torch.var(input_grads).item())

    return ret

def _shap(model: nn.Module, data_loader: DataLoader):

    def get_background(loader, exclude_indices, n_samples=100):
        all_indices = set(range(len(loader.dataset)))
        exclude_indices = set(exclude_indices)
        valid_indices = list(all_indices - exclude_indices)
        
        selected_indices = np.random.choice(valid_indices, n_samples, replace=False)
        
        return torch.stack([loader.dataset[idx][0] for idx in selected_indices])
    
    device = next(model.parameters()).device

    ret = {
        "shap_norm": [],
        "shap_var": [],
    }

    for inputs, targets, indices in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        background = get_background(data_loader, indices).to(device)

        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(inputs)

        target_shap_values = []
        for i in range(len(targets)):
            shap_value = shap_values[i, :, :, :, targets[i]]
            target_shap_values.append(shap_value.reshape(-1))

        ret["shap_var"].extend(np.var(target_shap_values,axis=1).tolist())
        ret["shap_norm"].extend(np.linalg.norm(target_shap_values, axis=1).tolist())
    
    return ret

METRICS = {
    "loss": _loss,
    "grads": _grads,
    "shap": _shap,
}

def get_final_model_metrics(model: nn.Module, data_loader: DataLoader, metrics: list[str] = None):
    if metrics is None:
        metrics = METRICS.keys()
    else:
        for metric in metrics:
            if metric not in METRICS:
                raise ValueError()
    
    ret = {}
    model.eval()

    for metric in metrics:
        fn = METRICS[metric]
        ret.update(fn(model, data_loader))
       
    return pd.DataFrame(ret)

