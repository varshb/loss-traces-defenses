import argparse
from loss_traces.data_processing.data_processing import (
    get_num_classes,
    get_trainset,
    get_testset,
    prepare_transform,
    prepare_loaders)
from loss_traces.models.model import load_model
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from loss_traces.config import MODEL_DIR
import optuna
import time
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    if args.baseline_cifar:
        # Run baseline CIFAR experiment
        config = {
        "dataset": "CIFAR10",
        "arch": "wrn28-2",
        "batchsize": 256,
        "num_workers": 4,
        "augment": True,
        "device": "cuda",
        "verbose": args.verbose,
        "seed": args.seed,
        "save_model": True
        }
        teacher_exp_id = "CIFAR_top5_l0"
        student_exp_id = "CIFAR_top5_l0"
        temp = 2.0
        weight = 0.5
        epochs = 100
        print(f"Running baseline CIFAR experiment with temp={temp} and weight={weight}")

        acc = run_kd_experiment(config, teacher_exp_id, student_exp_id, temp, weight, epochs)
        print(f"Baseline CIFAR experiment completed with accuracy: {acc}")
    
    elif args.optuna_epochs:
        # Run Optuna experiment
        study = optuna.create_study(direction="maximize")

        config = {
        "dataset": "CIFAR10",
        "arch": "wrn28-2",
        "batchsize": 256,
        "num_workers": 4,
        "augment": True,
        "device": "cuda",
        "verbose": args.verbose,
        "seed": args.seed,
        "save_model": False
        }
        teacher_exp_id = "CIFAR_top5_l0"
        student_exp_id = "CIFAR_top5_l0"
        temp = 2.0
        weight = 0.5
        
        print(f"Running Optuna experiment with temp={temp} and weight={weight}")
        results = []
        def objective(trial):
            epochs = trial.suggest_int("epochs", 20, 100, step=5)
            print(f"Running trial with epochs={epochs}")
            acc = run_kd_experiment(config, teacher_exp_id, student_exp_id, temp, weight, epochs)

            results.append((epochs, acc))
            return acc

        study.optimize(objective, n_trials=30)
        print(f"Optuna experiment completed with best accuracy: {study.best_value}")

        # Separate epochs and accuracy
        x = [e for e, a in results]
        y = [a for e, a in results]

        results_dict = {}
        for e, a in results:
            if e not in results_dict:
                results_dict[e] = []
            results_dict[e].append(a)

        avg_y = {e: np.mean(a) for e, a in results_dict.items()}
            
        plt.scatter(x, y)
        plt.plot(list(avg_y.keys()), list(avg_y.values()), color='red', alpha=0.5, label='Average Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Epochs vs Accuracy (temp={temp}, weight={weight})")
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig("epochs_vs_accuracy.png")
        plt.show()

    elif args.ablation:
        # Add your ablation study code here
        temperatures = [1.0, 2.0, 4.0]
        weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        student_ids = ["CIFAR_top5_l2", "CIFAR_top5_l3", "CIFAR_top5_l4", "CIFAR_top5_l5", "CIFAR_top5_l6"]
        # test run
        # temperatures = [1.0]
        # weights = [0.5]
        # student_ids = ["wrn28-2_CIFAR_5_l0"]
        print(f"Running {len(temperatures)} temperature and {len(weights)} weight ablation study with {len(student_ids)} student models")
        total_trials = len(temperatures) * len(weights) * len(student_ids)
        print(f"Total trials: {total_trials}")
        config = {
        "dataset": "CIFAR10",
        "arch": "wrn28-2",
        "batchsize": 256,
        "num_workers": 4,
        "augment": True,
        "device": "cuda",
        "verbose": args.verbose,
        "seed": args.seed,
        "save_model": True
        }
        time_taken = {}
        teacher_exp_id = "wrn28-2_CIFAR_5_l0"
        for temp in temperatures:
            for weight in weights:
                for student_exp_id in student_ids:
                    print(f"Running KD experiment with temp={temp}, weight={weight}, student_exp_id={student_exp_id}")
                    start_time = time.time()
                    acc = run_kd_experiment(config, teacher_exp_id, student_exp_id, temp, weight, epochs=100)
                    print(f"KD experiment completed with accuracy: {acc}")
                    print(f"Time taken: {time.time() - start_time:.2f} seconds")
                    time_taken[(temp, weight, student_exp_id)] = time.time() - start_time
        df = pd.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in time_taken.items()],
            columns=["temp", "weight", "student_id", "time_taken"]
        )
        df.to_csv("time_taken.csv", index=False)

def run_kd_experiment(config, teacher_exp_id, student_exp_id, temp, weight, epochs=10):
    # load model
    teacher = load_model(config["arch"], get_num_classes(config["dataset"])).to(
        config["device"])
    teacher_saves = torch.load(f"{MODEL_DIR}/{teacher_exp_id}/target", weights_only=False)
    teacher.load_state_dict(teacher_saves["model_state_dict"])

    student = load_model(config["arch"], get_num_classes(config["dataset"])).to(
        config["device"])
    student_saves = torch.load(f"{MODEL_DIR}/{student_exp_id}/target", weights_only=False)

    if config['verbose']:
        print(f"Loaded teacher model from {teacher_exp_id} and student model from {student_exp_id}")

    # load data
    train_transform = prepare_transform(config['dataset'], config['arch'], config['augment'])
    plain_transform = prepare_transform(config['dataset'], config['arch'])

    train_superset = get_trainset(config['dataset'], train_transform)
    testset = get_testset(config['dataset'], plain_transform)



    trainset = Subset(train_superset, student_saves['trained_on_indices'])

    train_loader = DataLoader(
        trainset,
        batch_size=config['batchsize'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config['batchsize'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    # load optimizer
    weight_decay = 5e-4
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train model
    for epoch in range(epochs):
        start = time.time()
        stats = train_epoch(
            teacher_model=teacher,
            student_model=student,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config['device'],
            temperature=temp,
            alpha=weight
        )
        results = evaluate_model(
            model=student,
            data_loader=test_loader,
            device=config['device']
        )
        if config['verbose']:
            print(f"Epoch {epoch+1} - Total Loss: {stats['total_loss']:.4f}, CE Loss: {stats['ce_loss']:.4f}, KD Loss: {stats['kd_loss']:.4f}")
            print(f"Epoch {epoch+1} - Test Accuracy: {results:.2f}%")
            print(f"Epoch {epoch+1} - Time: {time.time() - start:.2f} seconds")

    # evaluate model
    acc = evaluate_model(
        model=student,
        data_loader=test_loader,
        device=config['device']
    )

    if config['save_model']:
        save_dict = {
        'model_state_dict': student.state_dict(),
        'trained_on_indices': student_saves['trained_on_indices'],
        'arch': student_saves['arch'],
        'seed' : config.get('seed', 0),
        'hyperparameters': student_saves['hyperparameters'],
        'dataset': student_saves['dataset'],
        'temperature': temp,
        'weight': weight,
        'test_acc': acc
        }

        exp_id = f"{student_exp_id}_kd_temp_{temp}_weight_{weight}"
        dir = f"trained_models/models/kd/"
        os.makedirs(dir, exist_ok=True)
        fullpath = os.path.join(dir, exp_id)
        torch.save(save_dict, fullpath)
        print(f"Saved model to {fullpath}")

    return acc


def load_data(config, saves):
    train_transform = prepare_transform(config['dataset'], config['arch'], config['augment'])
    plain_transform = prepare_transform(config['dataset'], config['arch'])

    train_superset = get_trainset(config['dataset'], train_transform)
    testset = get_testset(config['dataset'], plain_transform)

    num_classes = get_num_classes(config['dataset'])


    trainset = Subset(train_superset, saves['trained_on_indices'])

    train_loader = DataLoader(
        trainset,
        batch_size=config['batchsize'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config['batchsize'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return train_loader, test_loader, num_classes

def train_epoch(teacher_model, student_model, train_loader, 
                                  optimizer, scheduler, device, temperature=3.0, 
                                  alpha=0.7):
    """
    Train student with confidence-masked knowledge distillation
    """
    teacher_model.eval()
    student_model.train()
    
    kd_loss_fn = KnowledgeDistillation(
        temperature=temperature,
        alpha=alpha
    )
    
    epoch_stats = {'total_loss': 0, 'ce_loss': 0, 'kd_loss': 0}
    
    for  data, targets, batch_idx in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher_model(data)
        
        student_logits = student_model(data)
        
        # Calculate loss with confidence masking applied to teacher logits
        total_loss, ce_loss, kd_loss = kd_loss_fn(student_logits, teacher_logits, targets)
        
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_stats['total_loss'] += total_loss.item()
        epoch_stats['ce_loss'] += ce_loss.item()
        epoch_stats['kd_loss'] += kd_loss.item()
        
    scheduler.step()

    for key in epoch_stats:
        epoch_stats[key] /= len(train_loader)
    
    return epoch_stats

def evaluate_model(model, data_loader, device):
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets, _indices in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu()
                correct += correct_idx.sum()

        acc = 100. * float(correct) / float(total)
        print('test acc:', acc)
        return acc
    
class KnowledgeDistillation(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        ce_loss = self.ce_loss(student_logits, targets)
                
        # Knowledge distillation 
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        
        kd_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        
        return total_loss, ce_loss, kd_loss
    
def set_seed(seed=0):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KD experiment")
    parser.add_argument("--baseline_cifar", action="store_true", help="Run baseline CIFAR experiment")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=2546, help="Random seed")
    parser.add_argument("--optuna_epochs", action="store_true", help="Run Optuna experiment with varying epochs")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)