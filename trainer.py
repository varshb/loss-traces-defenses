import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, STORAGE_DIR, LOCAL_DIR


class Trainer:
    def __init__(self, args, dataloaders, device):
        self.args = args
        self.trainloader, self.plainloader, self.testloader = dataloaders
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
        for inputs, targets, _indices in self.plainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = self.loss_func_sample(outputs, targets)
            all_losses.extend(losses.tolist())
        # all_losses = torch.cat(all_losses, dim=0)
        model.train()
        return all_losses#.tolist()


    def train_epoch(self, model, epoch, computed_losses, args):
        print('\nEpoch: %d' % epoch)
        model.train()
        total = 0
        correct = 0
        t0 = time.time()

        # collect init losses
        if args.track_computed_loss and epoch == 0:
            computed_losses.append(self.get_all_losses(model, args))

        for batch_idx, (inputs, targets, indices) in enumerate(self.trainloader):
            model.zero_grad()
            self.optimizer.zero_grad()

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

        if args.track_computed_loss:
            computed_losses.append(self.get_all_losses(model, args))

        t1 = time.time()
        self.scheduler.step()
        acc = (100. * float(correct) / float(total)) if total > 0 else 0.0
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc

    def train_test(self, model, args, model_id):
        self.training_params = self.get_training_params(model)
        self.optimizer = self.get_optimizer(self.training_params, args.lr, args.momentum, args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)


        # init loss stores
        computed_losses = []
        self.free_train_losses = pd.DataFrame(np.full((len(self.trainloader.dataset), args.epochs), np.nan))
        self.free_train_losses.index = self.trainloader.dataset.indices

        num_training = len(self.trainloader.dataset)
        num_test = len(self.testloader.dataset)
        print('Training on: ', num_training, 'Testing on: ', num_test)

        steps_per_epoch = (num_training // args.batchsize)
        if (num_training % args.batchsize != 0):
            steps_per_epoch += 1

        print(f'-------- Training model {model_id} --------')
        print('\n==> Starting training')

        for epoch in range(args.epochs):
            train_acc = self.train_epoch(model, epoch, computed_losses, args)
            test_acc = self.test(model, args)

            if args.checkpoint and (epoch + 1) % 5 == 0:
                save_model(model, args, self.trainloader, train_acc, test_acc, checkpoint=True, epoch=epoch)

        print('\n==> Finished training')
        if args.track_free_loss:
            save_free_loss(self.free_train_losses, args)

        if args.track_computed_loss:
            save_tracking_data(computed_losses, args)

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


def save_tracking_data(computed_losses, args):
    outdir = os.path.join(STORAGE_DIR, 'losses')
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
