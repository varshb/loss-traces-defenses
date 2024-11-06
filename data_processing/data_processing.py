import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from config import DATA_DIR, MODEL_DIR
from data_processing.custom_dataset import IndexCIFAR10, IndexCIFAR100, IndexCIFAR100Coarse, \
    IndexRESISC45


def prepare_transform(dataset_name: str, arch: str, augment: bool = False):
    """
    Prepare transforms for a given dataset and architecture.

    Args:
        dataset_name (str): Name of the dataset
        arch (str): Model architecture
        augment (bool, optional): Whether to use data augmentation. Defaults to False.

    Returns:
        transforms.Compose: Composed transform for the dataset

    Raises:
        NotImplementedError: If the dataset is not supported
    """
    # CIFAR10 Transforms
    if dataset_name == "CIFAR10":
        # Specific transforms for ResNet architectures
        if 'rn' in arch or 'resnet' in arch:
            if augment:
                return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        # Basic transform for other architectures
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # CIFAR100 Transforms
    elif dataset_name in ["CIFAR100", "CIFAR100Coarse"]:
        # Specific transforms for ResNet architectures
        if 'rn' in arch or 'resnet' in arch:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

        # Basic transform for other architectures
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    # RESISC45 Transform
    elif dataset_name == "RESISC45":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Unsupported dataset
    raise NotImplementedError(f'Dataset {dataset_name} is not supported')

def get_trainset(dataset_name: str, transform: transforms.Compose) -> Dataset:
    if dataset_name == 'CIFAR10':
        dataset = IndexCIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
    elif dataset_name == 'CIFAR100':
        dataset = IndexCIFAR100(root=DATA_DIR, train=True, transform=transform, download=True)
    elif dataset_name == 'CIFAR100Coarse':
        dataset = IndexCIFAR100Coarse(root=DATA_DIR, train=True, transform=transform, download=True)
    elif dataset_name == 'RESISC45':
        dataset = IndexRESISC45(root=DATA_DIR, split='train_val', transforms=transform)
    else:
        raise NotImplementedError(f"Trainset '{dataset_name}' is not supported")
    return dataset

def get_testset(dataset_name: str, transform: transforms.Compose) -> Dataset:
    if dataset_name == 'CIFAR10':
        testset = IndexCIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    elif dataset_name == 'CIFAR100':
        testset = IndexCIFAR100(root=DATA_DIR, train=False, transform=transform, download=True)
    elif dataset_name == 'CIFAR100Coarse':
        testset = IndexCIFAR100Coarse(root=DATA_DIR, train=False, transform=transform, download=True)
    elif dataset_name == 'RESISC45':
        testset = IndexRESISC45(root=DATA_DIR, split='test', transforms=transform)
    else:
        raise NotImplementedError(f"Testset '{dataset_name}' is not supported")
    return testset

def get_no_shuffle_train_loader(
        dataset: str,
        arch: str,
        batch_size: int = 100,
        num_workers: int = 4,
        ) -> DataLoader:
    transform = prepare_transform(dataset, arch)
    attackset = get_trainset(dataset, transform)
    return DataLoader(attackset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def prepare_loaders(dataset: Dataset, plain_dataset: Dataset, testset: Dataset, num_classes: int, args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    select_indices = get_train_indices(args, dataset, num_classes)
    trainset = Subset(dataset, select_indices)

    workers = 4

    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=workers, pin_memory=True)
    plainloader = DataLoader(plain_dataset, batch_size=args.batchsize, shuffle=False, num_workers=workers, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=workers, pin_memory=True)
    return trainloader, plainloader, testloader


def get_train_indices(args, dataset: Dataset, num_classes: int) -> List[int]:
    ## Some special sampling
    if args.shadow_count is not None:
        all_indices = [i for i in range(len(dataset))]

        select_indices = [[] for _ in range(args.shadow_count)]
        for i in all_indices:
            chosen_shadow_models = np.random.choice(args.shadow_count, args.shadow_count // 2, replace=False)
            for model_idx in chosen_shadow_models:
                select_indices[model_idx].append(i)

        select_indices = select_indices[args.shadow_id]

    ## Copy what's there
    elif args.dual:
        target_trained_on_path = os.path.join(MODEL_DIR, args.exp_id, 'target_trained_on')
        target_path = os.path.join(MODEL_DIR, args.exp_id, 'target')
        if os.path.exists(target_trained_on_path):
            select_indices = torch.load(target_trained_on_path)
        elif os.path.exists(target_path):
            saved = torch.load(target_path, 'cpu')
            select_indices = saved['trained_on_indices']
        else:
            raise FileNotFoundError('Could not find target trainset to train this dual')

    elif args.track_grad_norms:
        if args.balanced_sampling:
            class_indices = {i: [] for i in range(num_classes)}
            for _, label, idx in dataset:
                class_indices[label].append(idx)

            samples_per_class = (len(dataset) // num_classes) // 2
            select_indices = []
            for idxs in class_indices.values():
                select_indices.extend(np.random.choice(idxs, samples_per_class, replace=False))

        else:
            print("Not using balanced sampling")
            select_indices, _other_indices = train_test_split(list(range(len(dataset))), test_size=len(dataset) // 2,
                                                              random_state=args.seed)

        os.makedirs(os.path.join(MODEL_DIR, args.exp_id), exist_ok=True)
        target_trained_on_path = os.path.join(MODEL_DIR, args.exp_id, 'target_trained_on')
        if os.path.exists(target_trained_on_path):
            print("TRAINING NEW TARGET - WILL NOT OVERWRITE TARGET_TRAINED_ON", file=sys.stderr)
        while os.path.exists(target_trained_on_path):
            target_trained_on_path += '_'
        torch.save(select_indices, target_trained_on_path)

    ## Create new set
    elif args.balanced_sampling:  ## TODO: remove option as this is default, but shadow models have their own thing -- make this clear
        class_indices = {i: [] for i in range(num_classes)}
        for _, label, idx in dataset:
            class_indices[label].append(idx)

        samples_per_class = (len(dataset) // num_classes) // 2
        select_indices = []
        for idxs in class_indices.values():
            select_indices.extend(np.random.choice(idxs, samples_per_class, replace=False))

    else:
        raise NotImplementedError('Unknown training mode.')

    return select_indices

def get_num_classes(dataset_name: str) -> int:
    if dataset_name == 'CIFAR10':
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        num_classes = 100
    elif dataset_name == 'CIFAR100Coarse':
        num_classes = 20
    elif dataset_name == 'RESISC45':
        num_classes = 45
    else:
        raise NotImplemented(f'Dataset {dataset_name} is not supported.')

    return num_classes
