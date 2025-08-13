import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine, GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.data_loader import DPDataLoader
from torch.amp import autocast, GradScaler

import random
from loss_traces.config import MODEL_DIR, STORAGE_DIR


class Trainer:
    def __init__(self, args, dataloaders, device):
        self.args = args
        self.trainloader, self.plainloader, self.testloader, self.augloader, self.vulnloader, self.aug_vulnloader = dataloaders
        self.num_epochs = args.epochs
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def get_optimizer(training_params, lr, momentum, weight_decay):
        optimizer = torch.optim.SGD(
            training_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        return optimizer

    @staticmethod
    def get_training_params(model):
        num_params = 0
        num_training_params = 0
        training_params = []
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                training_params.append(p)
                num_training_params += p.numel()
        print(f'Training {num_training_params / (10 ** 6)}M parameters out of {num_params / (10 ** 6)}M')
        return training_params

    def get_all_losses(self, model, args):
        model.eval()

        all_losses = []
        all_confs = []
        all_norms = []
        for inputs, targets, _indices in self.plainloader:
            self.optimizer.zero_grad()
            model.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = self.loss_func_sample(outputs, targets)
            all_losses.extend(losses.tolist())

            pred_confs, _ = torch.max(outputs, dim=1)
            target_confs = outputs[torch.arange(outputs.shape[0]), targets]
            outputs = outputs.clone()
            outputs[torch.arange(outputs.shape[0]), targets] = float('-inf')
            pred_confs, _ = torch.max(outputs, dim=1)
            m = target_confs - pred_confs
            all_confs.extend(m.tolist())

            # losses.mean().backward()
            # if args.track_grad_norms or args.clip_norm:
            #     batch_grads = [p.grad_sample.view(p.grad_sample.size(0), -1) for p in self.training_params]
            #     batch_norms = torch.cat(batch_grads, dim=1).norm(dim=1)
            #     all_norms.append(batch_norms)

        # all_losses = torch.cat(all_losses, dim=0)
        # if args.clip_norm or args.track_grad_norms:
        #     all_norms = torch.cat(all_norms, dim=0)
        #     if args.clip_norm:
        #         all_norms = all_norms.clamp(max=args.clip_norm)
        #     all_norms = all_norms.tolist()
        return all_losses, all_confs, all_norms


    def train_epoch(self, model, epoch, computed_losses, computed_confidences, grad_norms, args):
        print(f'\n{args.exp_id}-Epoch: %d' % epoch)
        model.train()
        total_samples = 0
        total_correct = 0
        t0 = time.time()

        # collect init losses
        if (args.track_computed_loss or args.track_confidences or args.track_grad_norms) and epoch == 0:
            losses, confs, norms = self.get_all_losses(model, args)
            if args.track_computed_loss:
                computed_losses.append(losses)
            if args.track_confidences:
                computed_confidences.append(confs)
            if args.track_grad_norms:
                grad_norms.append(norms)
        max_physical_batch_size = 256
        if args.augmult:
            with BatchMemoryManager(
            data_loader=self.trainloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=self.optimizer,
            ) as memory_safe_dataloader:
                acc = self.train_augloaders_average_losses(model, memory_safe_dataloader, epoch)
        else:
            for (inputs, targets, _) in self.trainloader:
                correct, total = self.train_batch(model, inputs, targets, _, False,args)
                total_correct += correct
                total_samples += total

        if (args.track_computed_loss or args.track_confidences or args.track_grad_norms):
            losses, confs, norms = self.get_all_losses(model, args)
            model.train()
            if args.track_computed_loss:
                computed_losses.append(losses)
            if args.track_confidences:
                computed_confidences.append(confs)
            if args.track_grad_norms:
                grad_norms.append(norms)

        acc = 100.0 * float(total_correct) / float(total_samples) if total_samples > 0 else 0.0

        t1 = time.time()
        self.scheduler.step()
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc

    def train_epoch_selective_clip(self, model, epoch, computed_losses, computed_confidences, grad_norms, args):
        print(f'\n{args.exp_id}-Epoch: %d' % epoch)
        model.train()
        total_correct, total_samples = 0, 0

        t0 = time.time()

        # collect init losses
        if (args.track_computed_loss or args.track_confidences or args.track_grad_norms) and epoch == 0:
            losses, confs, norms = self.get_all_losses(model, args)
            if args.track_computed_loss:
                computed_losses.append(losses)
            if args.track_confidences:
                computed_confidences.append(confs)
            if args.track_grad_norms:
                grad_norms.append(norms)

        clipped_correct, clipped_total = 0, 0
        unclipped_correct, unclipped_total = 0, 0
        if args.augmult: 
            # not implemented yet
            pass

        else:
            clipped_correct, clipped_total = 0, 0
            unclipped_correct, unclipped_total = 0, 0
            src = "A"
        
            for (inputs, targets, _), src in seed_random_interleave(self.trainloader, self.vulnloader, seed=epoch):  # randomise batches between loaders
                if src == "A":
                    correct, total = self.train_batch_selective_clipping(model, inputs, targets, _, False, args) ## NOTE: passing in True here for selective clipping for debugging
                    unclipped_correct += correct
                    unclipped_total += total
                elif src == "B":
                    correct, total = self.train_batch_selective_clipping(model, inputs, targets, _, True, args)
                    clipped_correct += correct
                    clipped_total += total
                total_correct += correct
                total_samples += total
        print("unclipped acc:", (unclipped_correct / unclipped_total if unclipped_total > 0 else 0) * 100)
        print("clipped acc:", (clipped_correct / clipped_total if clipped_total > 0 else 0) * 100)

        if (args.track_computed_loss or args.track_confidences or args.track_grad_norms):
            losses, confs, norms = self.get_all_losses(model, args)
            model.train()
            if args.track_computed_loss:
                computed_losses.append(losses)
            if args.track_confidences:
                computed_confidences.append(confs)
            if args.track_grad_norms:
                grad_norms.append(norms)

        t1 = time.time()
        acc = 100.0 * float(total_correct) / float(total_samples) if total_samples > 0 else 0.0
        self.scheduler.step()
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc
    

    def train_test(self, model, args, model_id):
        self.training_params = self.get_training_params(model)
        self.optimizer = self.get_optimizer(self.training_params, args.lr, args.momentum, args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)


        if args.private:
            if args.augmult:
                dataloader = self.augloader
            else:
                dataloader = self.trainloader
            self.privacy_engine = PrivacyEngine()
            model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(max_grad_norm=args.clip_norm, module=model,
                                                                                     optimizer=self.optimizer,
                                                                                     data_loader=dataloader,
                                                                                     target_epsilon=args.target_epsilon,
                                                                                     target_delta=args.target_delta,
                                                                                     epochs=args.epochs)

        elif (args.clip_norm or args.noise_multiplier) and not args.selective_clip:
            if args.augmult:
                dataloader = self.augloader
            else:
                dataloader = self.trainloader
            self.privacy_engine = PrivacyEngine()
            model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                                                                                max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                                                module=model,
                                                                                optimizer=self.optimizer,
                                                                                data_loader=dataloader,
                                                                                noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
                                                                                )
        elif args.selective_clip:
            print("Using selective clipping")
            self.privacy_engine = PrivacyEngine()
            model, _, _ = self.privacy_engine.make_private(
                                                                    max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                                    module=model,
                                                                    optimizer=self.optimizer,
                                                                    data_loader=self.vulnloader,
                                                                    noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
                                                                    )
            self.optimizer = SelectiveDPOptimizer(self.optimizer,
                                                       noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
                                                       max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                       expected_batch_size=256)
            self.trainloader = DPDataLoader.from_data_loader(self.trainloader) # wraps loaders so does Poisson sampling
            self.vulnloader = DPDataLoader.from_data_loader(self.vulnloader)

        if args.track_grad_norms and not isinstance(model, GradSampleModule):
            model = GradSampleModule(model)

        # init loss stores
        computed_losses = []
        computed_confidences = []
        grad_norms = []

        self.free_train_losses = pd.DataFrame(np.full((len(self.trainloader.dataset), args.epochs), np.nan))
        self.free_train_confidences = pd.DataFrame(np.full((len(self.trainloader.dataset), args.epochs), np.nan))

        num_training = len(self.trainloader.dataset)
        num_test = len(self.testloader.dataset)
        print('Training on: ', num_training, 'Testing on: ', num_test)

        steps_per_epoch = (num_training // args.batchsize)
        if (num_training % args.batchsize != 0):
            steps_per_epoch += 1

        print(f'-------- Training model {model_id} --------')
        print('\n==> Starting training')

        if args.selective_clip:
            epoch_trainer = self.train_epoch_selective_clip
        else:
            print("Training Normally")
            epoch_trainer = self.train_epoch

        for epoch in range(args.epochs):
            train_acc = epoch_trainer(model, epoch, computed_losses, computed_confidences, grad_norms, args)
            test_acc = self.test(model, args)

            if args.checkpoint and (epoch + 1) % 5 == 0:
                save_model(model, args, self.trainloader, train_acc, test_acc, checkpoint=True, epoch=epoch)

        # print('\n==> Finished training')
        if args.track_free_loss:
            save_free_loss(self.free_train_losses, args)

        if args.track_computed_loss:
            save_tracking_data(computed_losses, args)

        if args.track_confidences:
            save_tracking_data(computed_confidences, args, save_dir="confidences")

        if args.track_grad_norms:
            save_tracking_data(grad_norms, args, save_dir="grad_norms")

        save_model(model, args, self.trainloader, train_acc, test_acc)

    def test(self, model, args):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _indices in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu()
                correct += correct_idx.sum()

        acc = 100. * float(correct) / float(total)
        print('test acc:', acc)
        return acc

    def train_batch(self, model, inputs, targets, indices, selective_clipping,args):
        model.train()
        model.zero_grad()
        self.optimizer.zero_grad()

        if isinstance(self.optimizer, DPOptimizer):
            self.optimizer.expected_batch_size = inputs.shape[0]

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outputs = model(inputs).squeeze(-1).squeeze(-1)

        losses = self.loss_func_sample(outputs, targets)
        if self.args.track_free_loss:
            self.free_train_losses.loc[indices.tolist(), self.epoch] = losses.tolist()

        losses.mean().backward()
        self.optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        minibatch_correct = predicted.eq(targets.data).float().cpu()
        total = targets.size(0)
        correct = minibatch_correct.sum()
        
        return correct, total
    
    def train_augloaders_average_losses(self, model, dataloader, epoch):        
        total_correct = 0
        total_samples = 0
        if epoch == 0:
            print(type(self.optimizer))
            print("training with augmentations")
        scaler = GradScaler('cuda')
        for batch_idx, (inputs, targets, indices) in enumerate(dataloader):
            model.train()
            self.optimizer.zero_grad()

            N, K, C, H, W = inputs.shape  
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if hasattr(self.optimizer, 'expected_batch_size'):
                self.optimizer.expected_batch_size = N

            inputs_flat = inputs.view(N*K, C, H, W)  # [N*K, C, H, W]
            targets_flat = targets.repeat_interleave(K)  # [N*K]
            
            # Single forward pass for all N*K images 
            with autocast('cuda'):
                outputs = model(inputs_flat)  
                
                # Compute per-sample losses 
                losses = self.loss_func_sample(outputs, targets_flat)  # [N*K]
                
                # Reshape to N, K and average over augmentations
                losses = losses.view(N, K)
                per_sample_losses = losses.mean(dim=1)  # average over K augmentations 
                
                # Final loss
                total_loss = per_sample_losses.mean()  
                
                # This is equivalent of taking avg of K gradients
            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            with torch.no_grad():
                model.eval()
                first_aug_outputs = model(inputs[:, 0])  
                preds = first_aug_outputs.argmax(dim=1)
                # print(preds)
                # print(targets)
                correct = (preds == targets).sum().item()
                model.train()

            total_correct += correct
            total_samples += N

        acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
        return acc


    def train_batch_selective_clipping(self, model, inputs, targets, indices, should_clip, args):
        """
        DP optimizer approach with selective clipping
        """
        if should_clip:
            self.optimizer._should_clip_current_batch = True
        else:
            self.optimizer._should_clip_current_batch = False
        model.train()
        model.zero_grad()
        self.optimizer.zero_grad()

        if isinstance(self.optimizer, DPOptimizer):
            self.optimizer.expected_batch_size = inputs.shape[0]
        
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outputs = model(inputs).squeeze(-1).squeeze(-1)

        losses = self.loss_func_sample(outputs, targets)
        
        if hasattr(args, 'track_free_loss') and args.track_free_loss:
            self.free_train_losses.loc[indices.tolist(), self.epoch] = losses.tolist()

        losses.mean().backward()
        self.optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        minibatch_correct = predicted.eq(targets.data).float().cpu()
        total = targets.size(0)
        correct = minibatch_correct.sum()
        
        return correct, total

def save_free_loss(train_loss, args):
    train_dir = os.path.join(STORAGE_DIR, 'free_train_losses')
    os.makedirs(train_dir, exist_ok=True)
    file = args.exp_id

    if args.dual:
        file += f'_dual_{args.dual}'
    file += '.pq'

    train_path = os.path.join(train_dir, file)
    train_loss.to_parquet(train_path)


def save_tracking_data(computed_losses, args, save_dir=None):
    if not save_dir:
        outdir = os.path.join(STORAGE_DIR, 'losses')
    else:
        outdir = os.path.join(STORAGE_DIR, save_dir)

    os.makedirs(outdir, exist_ok=True)

    file = args.exp_id
    if args.dual:
        file += f'_dual_{args.dual}'
    elif args.shadow_count:
        file += f'_shadow_{str(args.shadow_id)}'
    file += '.pq'

    fullpath = os.path.join(outdir, file)
    if os.path.exists(fullpath):
        print("'DUPLICATE' LOSS - OVERWRITING PREVIOUS", file=sys.stderr)

    print(fullpath)
    pd.DataFrame(computed_losses).transpose().to_parquet(fullpath)

def save_model(model, args, trainloader, train_acc, test_acc, checkpoint=False, epoch=None):
    if args.shadow_count:
        save_name = 'shadow_' + str(args.shadow_id)
    elif args.dual:
        save_name = f'dual_{args.dual}'
    elif args.track_computed_loss or args.track_free_loss:
        save_name = 'target'
    else:
        save_name = 'model'

    model_state_dict = model.state_dict()

    dic = {
        'model_state_dict': model_state_dict,
        'hyperparameters': vars(args),
        'trained_on_indices': trainloader.dataset.indices,
        'arch': args.arch,
        'seed': args.seed,
        'dataset': args.dataset,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    dir = os.path.join(MODEL_DIR, args.exp_id)

    if checkpoint:
        dir = os.path.join(dir, f'checkpoint_before_{epoch + 1}')
    os.makedirs(dir, exist_ok=True)
    fullpath = os.path.join(dir, save_name)
    torch.save(dic, fullpath)


def seed_random_interleave(loader_a, loader_b, seed=None):
    order = ["A"] * len(loader_a) + ["B"] * len(loader_b)

    rng = random.Random(seed)
    rng.shuffle(order)

    iter_a = iter(loader_a)
    iter_b = iter(loader_b)

    for src in order:
        if src == "A":
            try:
                yield next(iter_a), "A"
            except StopIteration:
                yield next(iter_b), "B"
        else:
            try:
                yield next(iter_b), "B"
            except StopIteration:
                yield next(iter_a), "A"

class SelectiveDPOptimizer(DPOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._should_clip_current_batch = True
    
    def pre_step(self, closure=None):
        if not self._should_clip_current_batch:
            if self.step_hook:
                self.step_hook(self)
            self._is_last_step_skipped = False
            return True

        # Otherwise use normal DP 
        return super().pre_step(closure)