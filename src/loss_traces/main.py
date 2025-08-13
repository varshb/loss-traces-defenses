import argparse
import time
import random
import numpy as np
import csv
import pickle
from loss_traces.config import STORAGE_DIR
import torch
from opacus.validators import ModuleValidator
from torch.utils.data import Subset
from loss_traces.data_processing.data_processing import (
    prepare_transform,
    get_trainset,
    get_testset,
    prepare_loaders,
    get_num_classes,
)
from loss_traces.models.model import load_model
from loss_traces.trainer import Trainer


def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", default=64, type=int, help="batch size")
    parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.01, type=float, help="base learning rate")
    parser.add_argument(
        "--exp_id", default="exp_default", type=str, help="filename for saving results"
    )
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0, type=float, help="value of momentum")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--arch", default="simple_convnet", type=str, help="model architecture to use"
    )
    parser.add_argument(
        "--dataset", default="CIFAR10", type=str, help="dataset to be trained on"
    )
    parser.add_argument(
        "--clip_norm",
        type=float,
        default=None,
        help="enable per-sample gradient clipping and set clipping norm",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="noise multiplier for training with DP",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--target_epsilon", type=float, default=None, help="target epsilon for DP"
    )
    parser.add_argument(
        "--target_delta", type=float, default=1e-5, help="target delta for DP"
    )

    parser.add_argument(
        "--augment", action="store_true", help="train with augmentation if available"
    )
    parser.add_argument("--checkpoint", action="store_true")

    # For target model training
    parser.add_argument(
        "--track_free_loss",
        action="store_true",
        help="track individual losses from training",
    )
    parser.add_argument(
        "--track_computed_loss",
        action="store_true",
        help="enable individual loss tracking (computed once per epoch)",
    )
    parser.add_argument(
        "--track_confidences",
        action="store_true",
        help="enable individual loss tracking (computed once per epoch)",
    )
    parser.add_argument(
        "--track_grad_norms",
        action="store_true",
        help="enable individual grad norm tracking (computed once per epoch)",
    )

    parser.add_argument(
        "--balanced_sampling",
        action="store_false",
        help="force target model to train on *balanced* subset of training data",
    )

    # For training a model on same set as the target
    parser.add_argument(
        "--dual",
        default=None,
        type=str,
        help="To train a model on an existing set of indices (and name this model)",
    )

    # For shadow model training
    parser.add_argument(
        "--shadow_count",
        default=None,
        type=int,
        help="number of shadow models being trained",
    )
    parser.add_argument(
        "--shadow_id",
        default=None,
        type=int,
        help="id of the individual shadow model to train",
    )
    parser.add_argument(
        "--gpu",
        default="",
        type=str,
        help="select gpu to run e.g. ':0' where multiple are available",
    )

    parser.add_argument("--model_start", type=int, default=0)
    parser.add_argument("--model_stop", type=int, default=1)

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="layer index for removed vulnerable points (default: 0)",
    )
    parser.add_argument(
        "--layer_folder",
        type=str,
        default="",
        help="folder name for storing layer indices (default: empty)",
    )
    parser.add_argument(
        "--augmult",
        action="store_true",
        help="Enable data augmentation multiplicatively"
    )
    parser.add_argument("--selective_clip", action="store_true",
                      help="Enable selective clipping")

    args = parser.parse_args()

    if (args.shadow_id is not None and not args.shadow_count) or (
        args.dual and args.shadow_count is not None
    ):
        raise ValueError("Invalid argument combination")

    return args


def set_seed(seed=0):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_input()
    if args.augmult:
        print("Using multiplicative data augmentation")
    print("Arguments: ", args)

    device = "cuda" + args.gpu if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)

    train_transform = prepare_transform(args.dataset, args.arch, args.augment)
    plain_transform = prepare_transform(args.dataset, args.arch)
    aug_transform = prepare_transform(args.dataset, args.arch, apply_augmult=True)

    start = time.time()
    train_superset = get_trainset(args.dataset, train_transform)
    plain_train_superset = get_trainset(args.dataset, plain_transform)
    aug_dataset = get_trainset("MultiAugmentDataset", aug_transform, plain_transform)
    testset = get_testset(args.dataset, plain_transform)
    stop = time.time()
    print(f"Loading datasets took {stop - start}")

    for model_id in range(args.model_start, args.model_stop):
        print(f"-------- Training model {model_id} --------")
        if args.shadow_count:
            args.shadow_id = model_id

        set_seed(args.seed)
        print("==> Preparing data..")
        num_classes = get_num_classes(args.dataset)
        if args.layer > 0:
            vuln_path = (
                    f"{STORAGE_DIR}/layer_target_indices/"
                    f"{args.layer_folder}/layer_{args.layer-1}_vulnerable.pkl"
                )
            with open(vuln_path, "rb") as f:
                vulnerable = pickle.load(f)
            vulnerable = list(vulnerable)
            save_path = (
                    f"{STORAGE_DIR}/layer_target_indices/"
                    f"{args.layer_folder}/layer_{args.layer-1}_safe.pkl"
                )
            print(f"Loading safe indices from: {save_path}")
            with open(save_path, "rb") as f:
                nonvuln_target = pickle.load(f)

            nonvuln_target = list(nonvuln_target)
            print(len(nonvuln_target), " non-vulnerable points loaded")

            if args.shadow_count is None: # for target model
                print(f"Removing vulnerable points from layer {args.layer}")
                print("Len before removing: ", len(train_superset))



                trainloader, plainloader, testloader, augloader, vulnloader, aug_vulnloader = prepare_loaders(
                    train_superset, plain_train_superset, testset, aug_dataset, num_classes, nonvuln_target, vulnerable, False, args
                )

                print("Len after removing: ", len(trainloader.dataset))
            else: # for shadow model training
                print(f"Removing vulnerable points from layer {args.layer} for shadow model {args.shadow_id}")
                print("Len before removing: ", len(train_superset))
                # save_path = f"{STORAGE_DIR}/layer_target_indices/{args.layer_folder}/layer_{args.layer-1}_full_safe.pkl"
                # with open(save_path, "rb") as f:
                #     non_vulnerable = pickle.load(f)
                # non_vulnerable = list(non_vulnerable['og_idx'])
                trainloader, plainloader, testloader, augloader, vulnloader, aug_vulnloader = prepare_loaders(
                    train_superset, plain_train_superset, testset, aug_dataset, num_classes, nonvuln_target, vulnerable, True, args
                )

        else:  # first model
            print("Using full training set - no vulnerable points")
            trainloader, plainloader, testloader, augloader, vulnloader, aug_vulnloader = prepare_loaders(
                train_superset, plain_train_superset, testset, aug_dataset, num_classes, None, None,False, args
            )

        num_training = len(trainloader.dataset)
        num_test = len(testloader.dataset)
        print("Training on: ", num_training, "Testing on: ", num_test)

        steps_per_epoch = num_training // args.batchsize
        if num_training % args.batchsize != 0:
            steps_per_epoch += 1

        print("\n==> Initialising the model..", args.checkpoint)

        model = load_model(args.arch, num_classes).to(device)

        if (args.clip_norm or args.private or args.track_grad_norms):
            model = ModuleValidator.fix(model)
        trainer = Trainer(args, (trainloader, plainloader, testloader, augloader, vulnloader, aug_vulnloader), device)

        trainer.train_test(model, args, model_id)

        del model
    print("\n==> Finished training")


if __name__ == "__main__":
    main()
