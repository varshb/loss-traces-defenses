import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from loss_traces.config import DATA_DIR, MODEL_DIR
from loss_traces.data_processing.custom_dataset import (
    IndexCIFAR10,
    IndexCIFAR100,
    IndexCINIC10,
    IndexCIFAR100Coarse,
    IndexRESISC45,
    MultiAugmentDataset
)


def prepare_transform(
    dataset_name: str, arch: str, augment: bool = False, mirror_all: bool = False, apply_augmult: bool = False
):
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
    if augment and mirror_all:
        raise ValueError(
            "Both augment and mirror_all are set to True. Only one should be set at a time"
        )

    # CIFAR10 Transforms
    if dataset_name == "CIFAR10":
        # Specific transforms for ResNet architectures
        if "rn" in arch or "resnet" in arch:
            if augment:
                return transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
            elif mirror_all:
                return transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(1),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
            elif apply_augmult:
                return transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(32),
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                            ),
                    ]
                )
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        # Basic transform for other architectures
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    # CINIC10 Transforms
    if dataset_name == "CINIC10":
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
        if augment:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif mirror_all:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    # CIFAR100 Transforms
    elif dataset_name in ["CIFAR100", "CIFAR100Coarse"]:
        if augment:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
        elif mirror_all:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
        # Basic transform
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

    # RESISC45 Transform
    elif dataset_name == "RESISC45":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # Unsupported dataset
    raise NotImplementedError(f"Dataset {dataset_name} is not supported")


def get_trainset(dataset_name: str, transform: transforms.Compose, non_aug_transform: transforms.Compose = None) -> Dataset:
    if dataset_name == "CIFAR10":
        dataset = IndexCIFAR10(
            root=DATA_DIR, train=True, transform=transform, download=True
        )
    elif dataset_name == "CIFAR100":
        dataset = IndexCIFAR100(
            root=DATA_DIR, train=True, transform=transform, download=True
        )
    elif dataset_name == "CINIC10":
        dataset = IndexCINIC10(root=DATA_DIR, partition="train", transform=transform)
    elif dataset_name == "CIFAR100Coarse":
        dataset = IndexCIFAR100Coarse(
            root=DATA_DIR, train=True, transform=transform, download=True
        )
    elif dataset_name == "RESISC45":
        dataset = IndexRESISC45(root=DATA_DIR, split="train_val", transforms=transform)
    elif dataset_name == "MultiAugmentDataset":
        dataset = MultiAugmentDataset(
            root=DATA_DIR, train=True, augmult=2, transform=transform,non_aug_transform=non_aug_transform, download=True
        )
    else:
        raise NotImplementedError(f"Trainset '{dataset_name}' is not supported")
    return dataset


def get_testset(dataset_name: str, transform: transforms.Compose) -> Dataset:
    if dataset_name == "CIFAR10":
        testset = IndexCIFAR10(
            root=DATA_DIR, train=False, transform=transform, download=True
        )
    elif dataset_name == "CIFAR100":
        testset = IndexCIFAR100(
            root=DATA_DIR, train=False, transform=transform, download=True
        )
    elif dataset_name == "CINIC10":
        testset = IndexCINIC10(root=DATA_DIR, partition="test", transform=transform)
    elif dataset_name == "CIFAR100Coarse":
        testset = IndexCIFAR100Coarse(
            root=DATA_DIR, train=False, transform=transform, download=True
        )
    elif dataset_name == "RESISC45":
        testset = IndexRESISC45(root=DATA_DIR, split="test", transforms=transform)
    else:
        raise NotImplementedError(f"Testset '{dataset_name}' is not supported")
    return testset


def get_no_shuffle_train_loader(
    dataset: str,
    arch: str,
    batch_size: int = 100,
    num_workers: int = 4,
    mirror_all: bool = False,
) -> DataLoader:
    transform = prepare_transform(dataset, arch, mirror_all=mirror_all)

    attackset = get_trainset(dataset, transform)
    return DataLoader(
        attackset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def prepare_loaders(
    dataset: Dataset, plain_dataset: Dataset, testset: Dataset, aug_dataset: Dataset, num_classes: int, nonvuln_target: list, vuln_target: list,  shadow_subset_train: bool, args
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    if nonvuln_target is not None and shadow_subset_train is False:
        diff = [i for i in vuln_target if i in nonvuln_target]
        print(len(diff))
        print("Using provided indices for training set")
        if args.selective_clip:
            vuln_dataset = Subset(dataset, vuln_target)
            aug_vuln = Subset(aug_dataset, vuln_target)
        trainset = Subset(dataset, nonvuln_target)
        aug_dataset = Subset(aug_dataset, nonvuln_target)

    elif shadow_subset_train:
        print("Using provided non-vulnerable indices for shadow training")
        non_vulnerable = [i for i in range(len(dataset)) if i not in vuln_target]
        select_indices = get_train_indices(
            args, dataset, num_classes, subset_indices=non_vulnerable
        )

        trainset = Subset(dataset, select_indices)
        aug_dataset = Subset(aug_dataset, select_indices)
        if args.selective_clip:
            print(len(vuln_target), " vulnerable points loaded")
            vuln_dataset = Subset(dataset, vuln_target)
            aug_vuln = Subset(aug_dataset, vuln_target)
    else:
        print("Using default sampling for training set")
        select_indices = get_train_indices(args, dataset, num_classes)

        trainset = Subset(dataset, select_indices)
        aug_dataset = Subset(aug_dataset, select_indices)
        print("trainset length: ", len(trainset))

    workers = min(6, os.cpu_count())
    print(f"Using {workers} workers for data loading")

    trainloader = DataLoader(
        trainset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    plainloader = DataLoader(
        plain_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    augloader = DataLoader(
        aug_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    if args.selective_clip:
        vulnloader = DataLoader(
            vuln_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        aug_vulnloader = DataLoader(
            aug_vuln,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

        return trainloader, plainloader, testloader, augloader, vulnloader, aug_vulnloader
    
    return trainloader, plainloader, testloader, augloader, None, None


def get_train_indices(args, dataset: Dataset, num_classes: int, subset_indices: list = None) -> List[int]:
    ## Some special sampling
    if args.shadow_count is not None:
        if subset_indices is not None:
            all_indices = subset_indices
            print("length of subset_indices: ", len(all_indices))
        else:
            all_indices = [i for i in range(len(dataset))]

        select_indices = [[] for _ in range(args.shadow_count)]
        for i in all_indices:
            chosen_shadow_models = np.random.choice(
                args.shadow_count, args.shadow_count // 2, replace=False
            )
            for model_idx in chosen_shadow_models:
                select_indices[model_idx].append(i)

        select_indices = select_indices[args.shadow_id]

    ## Copy what's there
    elif args.dual:
        target_trained_on_path = os.path.join(
            MODEL_DIR, args.exp_id, "target_trained_on"
        )
        target_path = os.path.join(MODEL_DIR, args.exp_id, "target")
        if os.path.exists(target_trained_on_path):
            select_indices = torch.load(target_trained_on_path)
        elif os.path.exists(target_path):
            saved = torch.load(target_path, "cpu")
            select_indices = saved["trained_on_indices"]
        else:
            raise FileNotFoundError("Could not find target trainset to train this dual")

    ## For a target model
    elif args.track_computed_loss or args.track_free_loss:
        if args.balanced_sampling:
            print("Using balanced sampling for target model")
            class_indices = {i: [] for i in range(num_classes)}
            for _, label, idx in dataset:
                class_indices[label].append(idx)

            samples_per_class = (len(dataset) // num_classes) // 2
            select_indices = []
            for idxs in class_indices.values():
                select_indices.extend(
                    np.random.choice(idxs, samples_per_class, replace=False)
                )

        else:
            print("Not using balanced sampling")
            select_indices, _other_indices = train_test_split(
                list(range(len(dataset))),
                test_size=len(dataset) // 2,
                random_state=args.seed,
            )

        os.makedirs(os.path.join(MODEL_DIR, args.exp_id), exist_ok=True)
        target_trained_on_path = os.path.join(
            MODEL_DIR, args.exp_id, "target_trained_on"
        )
        if os.path.exists(target_trained_on_path):
            print(
                "TRAINING NEW TARGET - WILL NOT OVERWRITE TARGET_TRAINED_ON",
                file=sys.stderr,
            )
        while os.path.exists(target_trained_on_path):
            target_trained_on_path += "_"
        torch.save(select_indices, target_trained_on_path)

    ## Create new set
    elif args.balanced_sampling:  ## TODO: remove option as this is default, but shadow models have their own thing -- make this clear
        class_indices = {i: [] for i in range(num_classes)}
        for _, label, idx in dataset:
            class_indices[label].append(idx)

        samples_per_class = (len(dataset) // num_classes) // 2
        select_indices = []
        for idxs in class_indices.values():
            select_indices.extend(
                np.random.choice(idxs, samples_per_class, replace=False)
            )

    else:
        raise NotImplementedError("Unknown training mode.")

    return select_indices


def get_num_classes(dataset_name: str) -> int:
    if dataset_name == "CIFAR10":
        num_classes = 10
    elif dataset_name == "CIFAR100":
        num_classes = 100
    elif dataset_name == "CINIC10":
        num_classes = 10
    elif dataset_name == "CIFAR100Coarse":
        num_classes = 20
    elif dataset_name == "RESISC45":
        num_classes = 45
    else:
        raise NotImplemented(f"Dataset {dataset_name} is not supported.")

    return num_classes
