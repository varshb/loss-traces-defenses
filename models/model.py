import os
import torch
from opacus.validators import ModuleValidator
from torch.nn import Module

from config import MODEL_DIR


## TODO: Remove irrelevant models + possibly the "part_to_train" convention if we're not using it
class ModelLoader:
    # @staticmethod
    # def _load_simple_convnet(num_classes: int) -> Tuple[Module, Module]:
    #     from models.simple_convnet import Net
    #     model = Net()
    #     return model, model

    # @staticmethod
    # def _load_resnet18_pretrained(num_classes: int) -> Tuple[Module, Module]:
    #     model = resnet18(weights='IMAGENET1K_V1')
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, num_classes)
    #     return model, model.fc
    #
    # @staticmethod
    # def _load_resnet18_full_tune(num_classes: int) -> Tuple[Module, Module]:
    #     model = resnet18(weights='IMAGENET1K_V1')
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, num_classes)
    #     model = ModuleValidator.fix(model)
    #     return model, model
    #
    # @staticmethod
    # def _load_resnet18_scratch(num_classes: int) -> Tuple[Module, Module]:
    #     model = resnet18(num_classes=num_classes)
    #     model = ModuleValidator.fix(model)
    #     return model, model

    # @staticmethod
    # def _load_resnet50_pretrained(num_classes: int) -> Tuple[Module, Module]:
    #     model = resnet50(weights='IMAGENET1K_V2')
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, num_classes)
    #     return model, model.fc
    #
    # @staticmethod
    # def _load_bit_resnet50(num_classes: int) -> Tuple[Module, Module]:
    #     import models.bit as bit
    #     MODEL_DIR = 'path/to/model/directory'  # Replace with actual path
    #     model = bit.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
    #     model.load_from(np.load(os.path.join(MODEL_DIR, 'BiT-M-R50x1.npz')))
    #     return model, model.head

    # @staticmethod
    # def _load_vgg11(num_classes: int) -> Tuple[Module, Module]:
    #     model = vgg11(num_classes=num_classes)
    #     return model, model
    #
    # @staticmethod
    # def _load_vgg16(num_classes: int) -> Tuple[Module, Module]:
    #     model = vgg16(num_classes=num_classes)
    #     return model, model

    # @staticmethod
    # def _load_resnet20_cifar(num_classes: int) -> Tuple[Module, Module]:
    #     from models.resnet_cifar import resnet20
    #     model = resnet20()
    #     return model, model

    @classmethod
    def _load_wide_resnet(cls, num_classes: int) -> Module:
        from models.wide_resnet import WideResNet
        model = WideResNet(28, num_classes, 2)
        model = ModuleValidator.fix(model)
        return model

    # Mapping of architecture names to their corresponding loader methods
    _ARCH_LOADERS = {
        'wrn28-2': _load_wide_resnet
    }

    @classmethod
    def load_model(cls, arch: str, num_classes: int) -> Module:
        """
        Load a model based on the specified architecture.

        Args:
            arch (str): Name of the model architecture
            num_classes (int): Number of output classes

        Returns:
            Module: The full model

        Raises:
            ValueError: If the specified architecture is not supported
        """
        # Retrieve the loader method for the specified architecture
        loader = cls._ARCH_LOADERS.get(arch)

        if loader is None:
            raise ValueError(f"Architecture '{arch}' is not supported.")

        # Call the loader method with the number of classes
        return loader.__func__(cls, num_classes)


def get_hyperparameter_from_file(exp_id: str, model_id: str) -> dict:
    path = os.path.join(MODEL_DIR, exp_id, model_id)
    saved = torch.load(path)
    print(len(saved['trained_on_indices']))
    try:
        print(saved['arch'])
        print(saved['train_acc'])
        print(saved['test_acc'])
    except:
        pass
    return saved['hyperparameters']


# Example usage
def load_model(arch: str, num_classes: int) -> Module:
    return ModelLoader.load_model(arch, num_classes)