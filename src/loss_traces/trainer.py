import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import opacus
import pickle
from opacus import PrivacyEngine, GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.data_loader import DPDataLoader
import loss_traces.augmult_utils
import torchvision.transforms.v2
from torchvision.transforms.v2 import functional as F
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

        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmuls on A100/RTX
        torch.backends.cudnn.allow_tf32 = True

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
        epoch_loss = 0.0
        clipped_samples = 0

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


        max_physical_batch_size = 128

        if args.augmult:
            with BatchMemoryManager(
            data_loader=self.trainloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=self.optimizer,
            ) as memory_safe_dataloader:
                for (inputs, targets, _) in memory_safe_dataloader:
                    correct, total, clipped, clipped_total = self.train_aug_batches(model, inputs, targets, _, False, args)
                    total_correct += correct
                    total_samples += total
        else:
            for (inputs, targets, _) in self.trainloader:
                correct, total, clipped, clipped_total = self.train_batch(model, inputs, targets, _)
                total_correct += correct
                total_samples += total
                clipped_samples += clipped

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
        epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0.0

        self.metrics["trainloader_accs"].append({'total':acc})
        self.metrics["clipped_samples"].append(clipped_samples)

        t1 = time.time()
        self.scheduler.step()
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        print("Clipped samples:", clipped_samples)
        print("Total Samples", total_samples)
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

        src = "A"
        clipped_correct, clipped_total = 0, 0
        unclipped_correct, unclipped_total = 0, 0
        clipped_samples = 0
        if args.augmult > 0: 
            for (inputs, targets, _), src in seed_random_interleave(self.trainloader, self.vulnloader, seed=epoch):  # randomise batches between loaders
                if src == "A":
                    correct, total, _, _ = self.train_batch_selective_clipping(model, inputs, targets, _, False, args) 
                    unclipped_correct += correct
                    unclipped_total += total
                elif src == "B":
                    correct, total, clipped, clipped_total = self.train_aug_batches(model, inputs, targets, _, True, args)
                    clipped_correct += correct
                    clipped_total += total
                    clipped_samples += clipped

                total_correct += correct
                total_samples += total
        else:
            for (inputs, targets, _), src in seed_random_interleave(self.trainloader, self.vulnloader, seed=epoch):  # randomise batches between loaders
                if src == "A":
                    correct, total, _, _ = self.train_batch_selective_clipping(model, inputs, targets, _, False, args) ## NOTE: passing in True here for selective clipping for debugging
                    unclipped_correct += correct
                    unclipped_total += total
                elif src == "B":
                    correct, total, clipped, clipped_total = self.train_batch_selective_clipping(model, inputs, targets, _, True, args)
                    clipped_correct += correct
                    clipped_total += total
                    clipped_samples += clipped

                total_correct += correct
                total_samples += total


        acc = 100.0 * float(total_correct) / float(total_samples) if total_samples > 0 else 0.0
        unclipped_acc = 100.0 * float(unclipped_correct) / float(unclipped_total) if unclipped_total > 0 else 0.0
        clipped_acc = 100.0 * float(clipped_correct) / float(clipped_total) if clipped_total > 0 else 0.0
        self.metrics["trainloader_accs"].append({'total':acc, 'unclipped': unclipped_acc, 'clipped': clipped_acc})
        self.metrics['clipped_samples'].append(clipped_samples)

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
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        print("Clipped samples:", clipped_samples, "Unclipped acc:", unclipped_acc, "Clipped acc:", clipped_acc)
        return acc
    

    def train_test(self, model, args, model_id):
        self.training_params = self.get_training_params(model)
        self.optimizer = self.get_optimizer(self.training_params, args.lr, args.momentum, args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)


        if args.augmult > 0:
            self.augmult_factor = args.augmult
            print("training with augmult:", self.augmult_factor)
            if args.selective_clip:
                print("training with augmult and selective clipping")
                pass # not impelmeneted yet
            elif args.private:
                print("training with augmult and full DP")
                self.privacy_engine = loss_traces.augmult_utils.PrivacyEngineAugmented(opacus.GradSampleModule.GRAD_SAMPLERS)
                model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(max_grad_norm=args.clip_norm, module=model,
                                                                        optimizer=self.optimizer,
                                                                        data_loader=self.trainloader,
                                                                        target_epsilon=args.target_epsilon,
                                                                        target_delta=args.target_delta,
                                                                        epochs=args.epochs)

            elif (args.clip_norm or args.noise_multiplier) and not args.selective_clip:
                print("training with augmult and clipping")
                self.privacy_engine = loss_traces.augmult_utils.PrivacyEngineAugmented(opacus.GradSampleModule.GRAD_SAMPLERS)
                model, self.optimizer, self.trainloader= self.privacy_engine.make_private(max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                            module=model,
                                            optimizer=self.optimizer,
                                            data_loader=self.trainloader,
                                            noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
                                            )

            augmentation = loss_traces.augmult_utils.AugmentationMultiplicity(self.augmult_factor)
            model.GRAD_SAMPLERS[torch.nn.modules.conv.Conv2d] = augmentation.augmented_compute_conv_grad_sample
            model.GRAD_SAMPLERS[torch.nn.modules.linear.Linear] = augmentation.augmented_compute_linear_grad_sample
            model.GRAD_SAMPLERS[torch.nn.GroupNorm] = augmentation.augmented_compute_group_norm_grad_sample


        else:
            if args.selective_clip:
                print("training without augmult and using selective clipping")
                self.privacy_engine = PrivacyEngine()
                model, _, _ = self.privacy_engine.make_private(max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                            module=model,
                                                            optimizer=self.optimizer,
                                                            data_loader=self.vulnloader,
                                                            noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
                                                            )
                self.optimizer = SelectiveDPOptimizer(self.optimizer,
                                                       noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
                                                       max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                       expected_batch_size=256)
                # self.trainloader = DPDataLoader.from_data_loader(self.trainloader) # wraps loaders so does Poisson sampling
                self.vulnloader = DPDataLoader.from_data_loader(self.vulnloader)

            elif args.private:
                print("training without augmult and with full DP")
                dataloader = self.trainloader
                self.privacy_engine = PrivacyEngine()
                model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(max_grad_norm=args.clip_norm, module=model,
                                                                                        optimizer=self.optimizer,
                                                                                        data_loader=dataloader,
                                                                                        target_epsilon=args.target_epsilon,
                                                                                        target_delta=args.target_delta,
                                                                                        epochs=args.epochs)
            elif (args.clip_norm or args.noise_multiplier) and not args.selective_clip:
                print("training without augmult and with clipping")
                dataloader = self.trainloader
                self.privacy_engine = PrivacyEngine()
                model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                                                                                max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                                                module=model,
                                                                                optimizer=self.optimizer,
                                                                                data_loader=dataloader,
                                                                                noise_multiplier=args.noise_multiplier if args.noise_multiplier else 0.0,
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

        self.metrics = {
            "trainloader_accs": [],
            "clipped_samples": []
        }

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

        print('\n==> Finished training')

        save_model(model, args, self.trainloader, train_acc, test_acc)

        if args.noise_multiplier:
            delta = 1e-5
            if noise_mult < 1.0:
                alphas = [1 + x / 1000.0 for x in range(1, 500)] + [1 + x / 10.0 for x in range(1, 100)]
            else:
                alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            epsilon, optimal_alpha = privacy_engine.get_epsilon(delta=1e-5, alphas=alphas)
            print(f"Noise: {noise_mult}, Epsilon: {epsilon:.3f}, Optimal Alpha: {optimal_alpha}")

        if args.track_free_loss:
            save_free_loss(self.free_train_losses, args)

        if args.track_computed_loss:
            save_tracking_data(computed_losses, args)

        if args.track_confidences:
            save_tracking_data(computed_confidences, args, save_dir="confidences")

        if args.track_grad_norms:
            save_tracking_data(grad_norms, args, save_dir="grad_norms")

        if args.clip_norm:
            train_dir = os.path.join(STORAGE_DIR, 'selective_losses')
            os.makedirs(train_dir, exist_ok=True)
            file = args.exp_id
            with open(os.path.join(train_dir, f'{file}_metrics.pkl'), 'wb') as f:
                pickle.dump(self.metrics, f)
        

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

    def train_batch(self, model, inputs, targets, indices):
        model.train()
        # model.zero_grad()
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

        if self.args.clip_norm:
            clipped, clipped_total = get_clipping_stats_from_optimizer(self.optimizer)
        else:
            clipped, clipped_total = 0, 0

        self.optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        minibatch_correct = predicted.eq(targets.data).float().cpu()
        total = targets.size(0)
        correct = minibatch_correct.sum()

        return correct, total,  clipped, clipped_total

    def train_aug_batches(self, model, inputs, targets, indices, should_clip, args):

        if args.selective_clip and should_clip:
            self.optimizer._should_clip_current_batch = True
        elif args.selective_clip and not should_clip:
            self.optimizer._should_clip_current_batch = False

        model.train()
        # model.zero_grad()
        self.optimizer.zero_grad(set_to_none=True)

        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

        original_batch_size = inputs.size(0)
        batch_xs = torch.repeat_interleave(inputs, repeats=self.augmult_factor, dim=0)
        batch_ys = torch.repeat_interleave(targets, repeats=self.augmult_factor, dim=0)
        transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.RandomCrop(32, padding=4),
            torchvision.transforms.v2.RandomHorizontalFlip(),
        ]
        )
        batch_xs = torchvision.transforms.v2.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(batch_xs
        )
        loss_func = torch.compile(self.loss_func_sample)

        assert batch_xs.size(0) == self.augmult_factor * original_batch_size
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(batch_xs)
            loss = loss_func(pred, batch_ys)
            loss.mean().backward()
        clipped, clipped_total = 0, 0

        self.optimizer.step()
        

        correct = (pred.detach().argmax(-1) == batch_ys).int().sum().item()
        total = batch_xs.size(0)

        return correct, total, clipped, clipped_total

    # def train_augloaders_average_losses(self, model, dataloader, epoch):
    #     total_correct = 0
    #     total_samples = 0
    #     if epoch == 0:
    #         print(type(self.optimizer))
    #         print("training with augmentations")
    #     trainloader_metrics = {'clipped_samples': 0, 'clipped_total': 0}
    #     for batch_idx, (inputs, targets, indices) in enumerate(dataloader):
    #         model.train()
    #         self.optimizer.zero_grad()

    #         N, K, C, H, W = inputs.shape  
    #         inputs = inputs.to(self.device)
    #         targets = targets.to(self.device)

    #         # if hasattr(self.optimizer, 'expected_batch_size'):
    #             # self.optimizer.expected_batch_size = N

    #         inputs_flat = inputs.view(N*K, C, H, W)  # [N*K, C, H, W]
    #         targets_flat = targets.repeat_interleave(K)  # [N*K]
        
    #     # Single forward pass for all N*K images 
    #         outputs = model(inputs_flat)  
            
    #         # Compute per-sample losses 
    #         losses = self.loss_func_sample(outputs, targets_flat)  # [N*K]
            
    #         # Reshape to N, K and average over augmentations
    #         losses = losses.view(N, K)
    #         per_sample_losses = losses.mean(dim=1)  # average over K augmentations 
            
    #         # Final loss
    #         total_loss = per_sample_losses.mean()  
            
    #         # This is equivalent of taking avg of K gradients
    #         total_loss.backward()
    #         clipped, clipped_total = get_clipping_stats_from_optimizer(self.optimizer)
    #         trainloader_metrics['clipped_samples'] += clipped
    #         trainloader_metrics['clipped_total'] += clipped_total
    #         self.optimizer.step()

    #         with torch.no_grad():
    #             model.eval()
    #             first_aug_outputs = model(inputs[:, 0])  
    #             preds = first_aug_outputs.argmax(dim=1)
    #             # print(preds)
    #             # print(targets)
    #             correct = (preds == targets).sum().item()
    #             model.train()

    #         total_correct += correct
    #         total_samples += N

    #         if batch_idx % 25 == 0:
    #             torch.cuda.empty_cache()  
    #     print(f"Clipped samples: {trainloader_metrics['clipped_samples']}")
    #     print(f"Total Samples: {trainloader_metrics['clipped_total']}")
    #     acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    #     return acc


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
        batch_losses = losses.mean().item()
        losses.mean().backward()

        if should_clip:
            clipped, clipped_total = get_clipping_stats_from_optimizer(self.optimizer)
        else:
            clipped, clipped_total = 0, 0

        self.optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        minibatch_correct = predicted.eq(targets.data).float().cpu()
        total = targets.size(0)
        correct = minibatch_correct.sum()
        
        return correct, total, clipped, clipped_total

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
    

def get_clipping_stats_from_optimizer(dp_optimizer):
    """Recompute clipping stats from the DP optimizer's state woo"""
    if len(dp_optimizer.grad_samples[0]) == 0:
        return 0, 0
    
    per_param_norms = [
        g.reshape(len(g), -1).norm(2, dim=-1) for g in dp_optimizer.grad_samples
    ]
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
    per_sample_clip_factor = (
        dp_optimizer.max_grad_norm / (per_sample_norms + 1e-6)
    ).clamp(max=1.0)
    
    total_samples = len(per_sample_clip_factor)
    clipped_samples = (per_sample_clip_factor < 1.0).sum().item()
    
    return clipped_samples, total_samples

