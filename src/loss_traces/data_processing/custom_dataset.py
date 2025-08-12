import torchgeo.datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
from PIL import Image

from loss_traces.config import CINIC_10_PATH


class IndexCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
    


class MultiAugmentDataset(torchvision.datasets.CIFAR10):
    def __init__(self, *args, augmult=4, non_aug_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmult = augmult
        self.non_aug_transform = non_aug_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        augmented_imgs = []
        if self.transform:
            # No augmentation for first image in list, just normalise if applicabkle
            augmented_imgs.append(self.non_aug_transform(img))
        else:
            augmented_imgs.append(transforms.ToTensor()(img))

        for _ in range(self.augmult - 1):
            if self.transform:
                augmented_imgs.append(self.transform(img))
            else:
                augmented_imgs.append(transforms.ToTensor()(img))

        # Shape: (K, C, H, W)
        augmented_imgs = torch.stack(augmented_imgs, dim=0)
        return augmented_imgs, torch.tensor(target), index
    


class IndexCIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index




class IndexCINIC10(torchvision.datasets.VisionDataset):
    classes=["airplane",
     "automobile",
     "bird",
     "cat",
     "deer",
     "dog",
     "frog",
     "horse",
     "ship",
     "truck"]

    def __init__(self, 
                 root: str = '',
                 partition: str = 'train',
                 transform = None):
        

        super().__init__(root, transforms=None, transform=None, target_transform=None)
        self.transform = transform
        self.data = torchvision.datasets.ImageFolder(os.path.join(CINIC_10_PATH, 'cinic-10', partition))

    def __getitem__(self, index):
        img, target = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)
                   

class IndexCIFAR100Coarse(IndexCIFAR100):
    coarse_labels = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                    18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
    
    def __getitem__(self, index):
        img, target, _index = super().__getitem__(index)
        return img, self.coarse_labels[target], index


import torchgeo

class IndexRESISC45(torchgeo.datasets.RESISC45):

    def __init__(self, *args, **kwargs):
        torchgeo.datasets.RESISC45.splits.append('train_val')
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        image, label = self._load_image(index)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label.item(), index
