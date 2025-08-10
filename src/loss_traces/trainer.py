import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine, GradSampleModule
from opacus.optimizers import DPOptimizer

from loss_traces.config import MODEL_DIR, STORAGE_DIR


class Trainer:
    def __init__(self, args, dataloaders, device):
        self.args = args
        self.trainloader, self.plainloader, self.testloader, self.augloader = dataloaders
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
        total = 0
        correct = 0
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

        if args.augmult:
            model.train()
            model.zero_grad()
            self.optimizer.zero_grad()

            N, K, C, H, W = inputs.shape  # unpack
            inputs = inputs.to(self.device)           # [N, K, C, H, W]
            targets = targets.to(self.device)         # [N]

            # Flatten augmentations into batch dimension
            inputs = inputs.view(N*K, C, H, W)         # [N*K, C, H, W]
            targets_flat = targets.repeat_interleave(K)  # [N*K]

            if isinstance(self.optimizer, DPOptimizer):
                self.optimizer.expected_batch_size = inputs.shape[0]  # N*K

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)  # [N*K, num_classes]

            # Per-sample losses
            losses = self.loss_func_sample(outputs, targets_flat)  # [N*K]

            # Reshape to [N, K] and average over augmentations
            losses = losses.view(N, K)
            loss = losses.mean(dim=1).mean()  # avg over K, then over N

            # Backward + step
            loss.backward()
            self.optimizer.step()
        else:
            for batch_idx, (inputs, targets, indices) in enumerate(self.trainloader):
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
                if args.track_free_loss:
                    self.free_train_losses.loc[indices.tolist(), epoch] = losses.tolist()

                losses.mean().backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                minibatch_correct = predicted.eq(targets.data).float().cpu()
                total += targets.size(0)
                correct += minibatch_correct.sum()

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
        self.scheduler.step()
        acc = (100. * float(correct) / float(total)) if total > 0 else 0.0
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc
    
    def train_epoch_clipping(self, model, epoch, computed_losses, computed_confidences, grad_norms, args):
        print(f'\n{args.exp_id}-Epoch: %d' % epoch)
        model.train()
        total = 0
        correct = 0
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


        for batch_idx, (inputs, targets, indices) in enumerate(self.trainloader):
            model.zero_grad()
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, DPOptimizer):
                self.optimizer.expected_batch_size = inputs.shape[0]

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = self.loss_func_sample(outputs, targets)
            if args.track_free_loss:
                self.free_train_losses.loc[indices.tolist(), epoch] = losses.tolist()

            losses.mean().backward()
            print(model.fc.weight.grad_sample) 
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            minibatch_correct = predicted.eq(targets.data).float().cpu()
            total += targets.size(0)
            correct += minibatch_correct.sum()

        if (args.track_computed_loss or args.track_confidences or args.track_grad_norms):
            losses, confs, norms = self.get_all_losses(model, args)
            if args.track_computed_loss:
                computed_losses.append(losses)
            if args.track_confidences:
                computed_confidences.append(confs)
            if args.track_grad_norms:
                grad_norms.append(norms)


        t1 = time.time()
        self.scheduler.step()
        acc = (100. * float(correct) / float(total)) if total > 0 else 0.0
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc


    def train_test(self, model, args, model_id):
        self.training_params = self.get_training_params(model)
        self.optimizer = self.get_optimizer(self.training_params, args.lr, args.momentum, args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)


        if args.private:
            privacy_engine = PrivacyEngine()
            model, self.optimizer, self.trainloader = privacy_engine.make_private_with_epsilon(max_grad_norm=args.clip_norm, module=model,
                                                                                     optimizer=self.optimizer,
                                                                                     data_loader=self.trainloader,
                                                                                     target_epsilon=args.target_epsilon,
                                                                                     target_delta=args.target_delta,
                                                                                     epochs=args.epochs)

        elif args.clip_norm or args.noise_multiplier:
            privacy_engine = PrivacyEngine()
            model, self.optimizer, self.trainloader = privacy_engine.make_private(
                                                                                max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                                                module=model,
                                                                                optimizer=self.optimizer,
                                                                                data_loader=self.trainloader,
                                                                                noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0
                                                                                )

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

        for epoch in range(args.epochs):
            train_acc = self.train_epoch(model, epoch, computed_losses, computed_confidences, grad_norms, args)
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
