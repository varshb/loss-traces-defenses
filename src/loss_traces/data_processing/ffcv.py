import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from ffcv import DatasetWriter, Loader
from ffcv.fields import IntField, RGBImageField
from ffcv.transforms import (
    ToTensor, ToDevice, Squeeze, NormalizeImage, 
    RandomHorizontalFlip, ToTorchImage, Convert,
    RandomResizedCrop, CenterCrop, ImageDecoder
)
from ffcv.loader import OrderOption
from torch.utils.data import Dataset, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from loss_traces.config import DATA_DIR, MODEL_DIR
from loss_traces.data_processing.custom_dataset import (
    IndexCIFAR10,
    IndexCIFAR100,
    IndexCINIC10,
    IndexCIFAR100Coarse,
    IndexRESISC45,
)


class FFCVDatasetConverter:
    """Converts PyTorch datasets to FFCV format"""
    
    @staticmethod
    def convert_dataset_to_ffcv(dataset: Dataset, output_path: str, max_resolution: int = 256):
        """Convert a PyTorch dataset to FFCV format"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define fields for FFCV dataset
        fields = {
            'image': RGBImageField(max_resolution=max_resolution, jpeg_quality=90),
            'label': IntField(),
            'index': IntField()  # For tracking original indices
        }
        
        writer = DatasetWriter(output_path, fields)
        
        # Write dataset to FFCV format
        for i, (image, label, index) in enumerate(dataset):
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            writer.write({
                'image': image,
                'label': label,
                'index': index
            })
        
        writer.close()
        print(f"Dataset converted to FFCV format: {output_path}")


def prepare_ffcv_transforms(
    dataset_name: str, 
    arch: str, 
    augment: bool = False, 
    mirror_all: bool = False,
    device: str = 'cuda'
):
    """
    Prepare FFCV transforms for a given dataset and architecture.
    
    Args:
        dataset_name (str): Name of the dataset
        arch (str): Model architecture
        augment (bool): Whether to use data augmentation
        mirror_all (bool): Whether to apply horizontal flip to all images
        device (str): Target device for data loading
        
    Returns:
        List of FFCV transforms
    """
    if augment and mirror_all:
        raise ValueError(
            "Both augment and mirror_all are set to True. Only one should be set at a time"
        )

    # Base transforms
    base_transforms = [
        ImageDecoder(),
        ToTensor(),
        ToDevice(torch.device(device), non_blocking=True),
        ToTorchImage(),
        Convert(torch.float32)
    ]

    # Dataset-specific transforms
    if dataset_name == "CIFAR10":
        mean = np.array([0.4914, 0.4822, 0.4465]) * 255
        std = np.array([0.2023, 0.1994, 0.2010]) * 255
        
        if augment and ("rn" in arch or "resnet" in arch):
            transforms_list = [
                ImageDecoder(),
                RandomHorizontalFlip(flip_prob=0.5),
                RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                ToTensor(),
                ToDevice(torch.device(device), non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                NormalizeImage(mean, std, np.float32)
            ]
        elif mirror_all:
            transforms_list = base_transforms + [
                RandomHorizontalFlip(flip_prob=1.0),
                NormalizeImage(mean, std, np.float32)
            ]
        else:
            transforms_list = base_transforms + [
                NormalizeImage(mean, std, np.float32)
            ]

    elif dataset_name == "CINIC10":
        mean = np.array([0.47889522, 0.47227842, 0.43047404]) * 255
        std = np.array([0.24205776, 0.23828046, 0.25874835]) * 255
        
        if augment:
            transforms_list = [
                ImageDecoder(),
                RandomHorizontalFlip(flip_prob=0.5),
                RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                ToTensor(),
                ToDevice(torch.device(device), non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                NormalizeImage(mean, std, np.float32)
            ]
        elif mirror_all:
            transforms_list = base_transforms + [
                RandomHorizontalFlip(flip_prob=1.0),
                NormalizeImage(mean, std, np.float32)
            ]
        else:
            transforms_list = base_transforms + [
                NormalizeImage(mean, std, np.float32)
            ]

    elif dataset_name in ["CIFAR100", "CIFAR100Coarse"]:
        mean = np.array([0.5071, 0.4867, 0.4408]) * 255
        std = np.array([0.2675, 0.2565, 0.2761]) * 255
        
        if augment:
            transforms_list = [
                ImageDecoder(),
                RandomHorizontalFlip(flip_prob=0.5),
                RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                ToTensor(),
                ToDevice(torch.device(device), non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                NormalizeImage(mean, std, np.float32)
            ]
        elif mirror_all:
            transforms_list = base_transforms + [
                RandomHorizontalFlip(flip_prob=1.0),
                NormalizeImage(mean, std, np.float32)
            ]
        else:
            transforms_list = base_transforms + [
                NormalizeImage(mean, std, np.float32)
            ]

    elif dataset_name == "RESISC45":
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        
        transforms_list = [
            ImageDecoder(),
            CenterCrop(ratio=256/256, scale=1.0),  # Resize to 256
            ToTensor(),
            ToDevice(torch.device(device), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            NormalizeImage(mean, std, np.float32)
        ]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")

    return transforms_list


def get_or_create_ffcv_dataset(dataset_name: str, split: str = "train") -> str:
    """Get path to FFCV dataset, creating it if it doesn't exist"""
    ffcv_dir = os.path.join(DATA_DIR, "ffcv_datasets")
    os.makedirs(ffcv_dir, exist_ok=True)
    
    ffcv_path = os.path.join(ffcv_dir, f"{dataset_name}_{split}.beton")
    
    if not os.path.exists(ffcv_path):
        print(f"Creating FFCV dataset for {dataset_name} {split}...")
        
        # Create the original dataset
        if split == "train":
            if dataset_name == "CIFAR10":
                dataset = IndexCIFAR10(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
            elif dataset_name == "CIFAR100":
                dataset = IndexCIFAR100(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
            elif dataset_name == "CINIC10":
                dataset = IndexCINIC10(root=DATA_DIR, partition="train", transform=transforms.ToTensor())
            elif dataset_name == "CIFAR100Coarse":
                dataset = IndexCIFAR100Coarse(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
            elif dataset_name == "RESISC45":
                dataset = IndexRESISC45(root=DATA_DIR, split="train_val", transforms=transforms.ToTensor())
            else:
                raise NotImplementedError(f"Dataset {dataset_name} is not supported")
        else:  # test split
            if dataset_name == "CIFAR10":
                dataset = IndexCIFAR10(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
            elif dataset_name == "CIFAR100":
                dataset = IndexCIFAR100(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
            elif dataset_name == "CINIC10":
                dataset = IndexCINIC10(root=DATA_DIR, partition="test", transform=transforms.ToTensor())
            elif dataset_name == "CIFAR100Coarse":
                dataset = IndexCIFAR100Coarse(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
            elif dataset_name == "RESISC45":
                dataset = IndexRESISC45(root=DATA_DIR, split="test", transforms=transforms.ToTensor())
            else:
                raise NotImplementedError(f"Dataset {dataset_name} is not supported")
        
        # Convert to FFCV
        FFCVDatasetConverter.convert_dataset_to_ffcv(dataset, ffcv_path)
    
    return ffcv_path


def get_ffcv_no_shuffle_train_loader(
    dataset: str,
    arch: str,
    batch_size: int = 100,
    num_workers: int = 4,
    mirror_all: bool = False,
    device: str = 'cuda'
) -> Loader:
    """Get FFCV loader for training data without shuffling"""
    ffcv_path = get_or_create_ffcv_dataset(dataset, "train")
    transforms_list = prepare_ffcv_transforms(dataset, arch, mirror_all=mirror_all, device=device)
    
    # Create pipelines for each field
    pipelines = {
        'image': transforms_list,
        'label': [IntField(), ToTensor(), Squeeze(), ToDevice(torch.device(device), non_blocking=True)],
        'index': [IntField(), ToTensor(), Squeeze(), ToDevice(torch.device(device), non_blocking=True)]
    }
    
    loader = Loader(
        ffcv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,  # No shuffling
        pipelines=pipelines,
        drop_last=False
    )
    
    return loader


def prepare_ffcv_loaders(
    dataset_name: str,
    arch: str,
    indices: Optional[List[int]],
    non_vulnerable: Optional[List[int]],
    args,
    device: str = 'cuda'
) -> Tuple[Loader, Loader, Loader]:
    """Prepare FFCV loaders for training, plain evaluation, and testing"""
    
    # Get number of classes
    num_classes = get_num_classes(dataset_name)
    
    # Get FFCV dataset paths
    train_ffcv_path = get_or_create_ffcv_dataset(dataset_name, "train")
    test_ffcv_path = get_or_create_ffcv_dataset(dataset_name, "test")
    
    # Prepare transforms
    train_transforms = prepare_ffcv_transforms(dataset_name, arch, augment=True, device=device)
    eval_transforms = prepare_ffcv_transforms(dataset_name, arch, augment=False, device=device)
    
    # Create pipelines
    train_pipelines = {
        'image': train_transforms,
        'label': [IntField(), ToTensor(), Squeeze(), ToDevice(torch.device(device), non_blocking=True)],
        'index': [IntField(), ToTensor(), Squeeze(), ToDevice(torch.device(device), non_blocking=True)]
    }
    
    eval_pipelines = {
        'image': eval_transforms,
        'label': [IntField(), ToTensor(), Squeeze(), ToDevice(torch.device(device), non_blocking=True)],
        'index': [IntField(), ToTensor(), Squeeze(), ToDevice(torch.device(device), non_blocking=True)]
    }
    
    # Handle training indices selection
    if indices is not None:
        print("Using provided indices for training set")
        train_indices = indices
    elif non_vulnerable is not None:
        print("Using provided non-vulnerable indices for shadow training")
        # Load original dataset to get indices
        original_dataset = get_trainset_for_indexing(dataset_name)
        train_indices = get_train_indices(args, original_dataset, num_classes, subset_indices=non_vulnerable)
    else:
        print("Using default sampling for training set")
        original_dataset = get_trainset_for_indexing(dataset_name)
        train_indices = get_train_indices(args, original_dataset, num_classes)
        print("trainset length: ", len(train_indices))
    
    # Create loaders
    trainloader = Loader(
        train_ffcv_path,
        batch_size=args.batchsize,
        num_workers=4,
        order=OrderOption.RANDOM,  # Shuffled
        indices=train_indices,
        pipelines=train_pipelines,
        drop_last=False
    )
    
    plainloader = Loader(
        train_ffcv_path,
        batch_size=args.batchsize,
        num_workers=4,
        order=OrderOption.SEQUENTIAL,  # No shuffling
        pipelines=eval_pipelines,
        drop_last=False
    )
    
    testloader = Loader(
        test_ffcv_path,
        batch_size=args.batchsize,
        num_workers=4,
        order=OrderOption.RANDOM,  # Shuffled
        pipelines=eval_pipelines,
        drop_last=False
    )
    
    return trainloader, plainloader, testloader


def get_trainset_for_indexing(dataset_name: str) -> Dataset:
    """Get trainset for index computation (without heavy transforms)"""
    basic_transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "CIFAR10":
        dataset = IndexCIFAR10(root=DATA_DIR, train=True, transform=basic_transform, download=True)
    elif dataset_name == "CIFAR100":
        dataset = IndexCIFAR100(root=DATA_DIR, train=True, transform=basic_transform, download=True)
    elif dataset_name == "CINIC10":
        dataset = IndexCINIC10(root=DATA_DIR, partition="train", transform=basic_transform)
    elif dataset_name == "CIFAR100Coarse":
        dataset = IndexCIFAR100Coarse(root=DATA_DIR, train=True, transform=basic_transform, download=True)
    elif dataset_name == "RESISC45":
        dataset = IndexRESISC45(root=DATA_DIR, split="train_val", transforms=basic_transform)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")
    
    return dataset


def get_train_indices(args, dataset: Dataset, num_classes: int, subset_indices: list = None) -> List[int]:
    """Get training indices based on various sampling strategies"""
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
    elif args.balanced_sampling:
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
    """Get number of classes for a dataset"""
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
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    return num_classes


# Backward compatibility functions that wrap FFCV loaders
def prepare_transform(dataset_name: str, arch: str, augment: bool = False, mirror_all: bool = False):
    """Backward compatibility wrapper - now returns FFCV transforms"""
    return prepare_ffcv_transforms(dataset_name, arch, augment, mirror_all)


def get_trainset(dataset_name: str, transform=None) -> str:
    """Backward compatibility wrapper - now returns FFCV dataset path"""
    return get_or_create_ffcv_dataset(dataset_name, "train")


def get_testset(dataset_name: str, transform=None) -> str:
    """Backward compatibility wrapper - now returns FFCV dataset path"""
    return get_or_create_ffcv_dataset(dataset_name, "test")


def get_no_shuffle_train_loader(
    dataset: str,
    arch: str,
    batch_size: int = 100,
    num_workers: int = 4,
    mirror_all: bool = False,
) -> Loader:
    """Backward compatibility wrapper for FFCV loader"""
    return get_ffcv_no_shuffle_train_loader(dataset, arch, batch_size, num_workers, mirror_all)


def prepare_loaders(
    dataset: str, plain_dataset: str, testset: str, num_classes: int, 
    indices: list, non_vulnerable: list, args
) -> Tuple[Loader, Loader, Loader]:
    """Backward compatibility wrapper for FFCV loaders"""
    # Extract dataset name from the dataset parameter (assuming it's a string now)
    dataset_name = dataset if isinstance(dataset, str) else args.dataset
    arch = getattr(args, 'arch', 'resnet18')
    
    return prepare_ffcv_loaders(dataset_name, arch, indices, non_vulnerable, args)