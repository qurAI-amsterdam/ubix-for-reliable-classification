import os
import sys
import subprocess
import shutil
import random
import time
import copy
import urllib.request
from tqdm.auto import tqdm

from pathlib import Path

import numpy as np
import scipy
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import wandb
import json
import yaml

# AddChanneld
from monai.transforms import (
    Spacingd,
    LoadImaged,
    CastToTyped,
    Orientationd,
    CastToTyped,
    Flipd,
    ToTensord,
    ScaleIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    Randomizable,
    RandGaussianNoise,
    Flip,
    Compose,
    EnsureChannelFirstd,
)
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.data import PersistentDataset, Dataset

from ubix.models.resnet_2d import resnet18 as resnet18_2d
from ubix.models.mil import MIL

from typing import Union, Dict, Optional, Any, Callable, Hashable, Mapping


def set_random_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    try:
        scipy.random.seed(seed)
    except AttributeError:
        pass

    random.seed(seed)


def get_random_states(device):
    states = dict()
    states["numpy"] = (np.random.get_state(),)
    states["torch"] = (torch.get_rng_state(),)

    if str(device).startswith("cuda"):
        states["torch_cuda"] = torch.cuda.get_rng_state()

    # states['scipy'] = scipy.random.get_state(),
    states["random"] = random.getstate()

    return states


def set_random_states(states, device):
    cudnn.deterministic = True
    cudnn.benchmark = False

    np.random.set_state(states["numpy"][0])
    torch.set_rng_state(states["torch"][0])

    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state(states["torch_cuda"])

    # scipy.random.set_state(states['scipy'][0])
    random.setstate(states["random"])


def print_jobid():
    slurm_jobid = (
        os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else "undefined"
    )
    print(f"slurm_jobid: {slurm_jobid}")


def get_std_file(config):
    make_model_results_folder(config)
    std_dir = os.path.join(get_model_results_folder(config), "stdout")
    make_last_dir_in(std_dir)
    f = os.path.join(std_dir, f"stdout_{int(time.time())}.txt")

    return f


"""
Log utils
"""


def log_stdout_to_file(config):
    std_file = get_std_file(config)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = None

        def write(self, message):
            self.terminal.write(message)

            try:
                self.log = open(std_file, "a")
                self.log.write(message)
                self.log.close()
            except ValueError as e:
                print("Received ValueError {}".format(e))

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            pass

    sys.stdout = Logger()


"""
Config utilities 
"""


class Run:
    def __init__(self, name) -> None:
        self.name = name


class Config:
    def __init__(self, config) -> None:
        for k, v in config.items():
            setattr(self, k, v)
    
    def update(self, dct, **kwargs):
        for k, v in dct.items():
            setattr(self, k, v)
        


def init_wandb_experiment(model_name, change_config=None):
    print(f"Running experiment {model_name}")

    wandb_key_file = Path("wandb.key")
    if wandb_key_file.exists():
        wandb.login(key=open(wandb_key_file, "r").read().strip())
    else:
        pass
        # raise FileNotFoundError(
        #     "You need to provide a wandb.key file in the root project folder."
        # )

    default_path = "ubix/experiments/default.yaml"

    # Dashes are replaced with paths and everything that comes after / will be removed.
    # We do the latter because the / should be used for giving separate folders to different ensemble instances.
    custom_path = os.path.join(
        "ubix/experiments", model_name.split("/")[0].replace("-", "/") + ".yaml"
    )

    def read_yaml(p):
        os.listdir(os.path.dirname(p))
        with open(p, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data = {k: v["value"] for k, v in data.items()}

        return data

    default = read_yaml(default_path)
    custom = read_yaml(custom_path)
    config = dict(list(default.items()) + list(custom.items()))

    if change_config is not None:
        for k, v in change_config.items():
            if type(config) is dict:
                config[k] = v
            else:
                setattr(config, k, v)

    if wandb_key_file.exists():
        wandb.init(
            config=config,
            name=model_name.replace("/", "#"),
            id=model_name.replace("/", "-"),
            project="ubix-experiments",
            resume=True,
            reinit=True,
            dir="./results",
            allow_val_change=True,
        )
    else:
        print('=' * 100)
        print(config)
        wandb.config = Config(config)
        wandb.run = Run(model_name)


"""
Path utilities
"""


def print_config(config):
    col1_len = max([len(i) for i in config.__dict__.keys()]) + 5
    for field_name, field_value in config.__dict__.items():
        print(("{:<" + str(col1_len) + "}{}").format(field_name, field_value))


def get_model_name(config):
    # if "model_name" in config._items.__dict__:
    try:
        if hasattr(config, "model_name"):
            return config.model_name
    except KeyError:
        pass
    return wandb.run.name


def get_model_results_folder(config):
    run_name = get_model_name(config).replace("#", "/")
    return os.path.join(config.model_results_root, run_name)


def get_predictions_folder(config, model_type):
    return make_last_dir_in(get_saved_model_path(config, model_type) + "_predictions")


def make_model_results_folder(config):
    model_folder_path = get_model_results_folder(config)
    if os.path.exists(model_folder_path):
        if not config.allow_model_overwrite:
            user_allow_model_overwrite = False

            while user_allow_model_overwrite not in ["y", "n"]:
                user_allow_model_overwrite = input(
                    f"Model with name {get_model_name(config)} already exists."
                    + "Do you want to continue and overwrite? (y/n)"
                )

            if user_allow_model_overwrite == "n":
                quit(f"Overwriting model {get_model_name(config)} not allowed.")
    else:
        os.makedirs(model_folder_path)


def make_last_dir_in(p):
    """
    Make the last directory in a certain path p if necessary. If the path before
    the last directory does not exit, raise error. This is to prevent the creation
    of low-level directories due to typos in filenames.
    :return: p
    """
    parent_folders_all = p.split(os.path.sep)[:-1]
    parent_folder = os.path.sep.join(parent_folders_all)

    if not os.path.exists(p):
        os.system(f"ls '{Path(parent_folder).parent}'")
        if os.path.exists(parent_folder):
            os.makedirs(p)
        else:
            raise FileNotFoundError(
                f"Parent folder '{parent_folder}' " + "does not exist"
            )
    return p


def get_saved_model_path(config, model_type):
    """
    See definition of model_type in function `save_model`.
    """
    saved_models_root = os.path.join(get_model_results_folder(config), "saved_models")
    Path(saved_models_root).mkdir(exist_ok=True, parents=True)
    saved_model_path = os.path.join(saved_models_root, f"model_{model_type}.pt")
    return saved_model_path


"""
Model utils
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    config,
    epoch,
    iteration,
    model,
    optimizer,
    loss,
    random_states,
    model_type,
    latest_es_monitor_batch,
):
    """
    Save a model. model_type could for example be 'latest', 'best_loss', 'best_acc'
    or 'best_auc'.
    """
    model_path = get_saved_model_path(config, model_type)

    best_val_scalars = {}

    for k, v in wandb.run.summary.__dict__.items():
        if k.startswith("best_"):
            best_val_scalars[k] = v
            print("Found k, best_val_scalars[k]", k, best_val_scalars[k])

    torch.save(
        {
            "epoch": epoch,
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "random_states": random_states,
            "best_val_scalars": best_val_scalars,
            "latest_es_monitor_batch": latest_es_monitor_batch,
        },
        model_path,
    )


def make_model(config, train_set=None, sample_size=None, sample_duration=None):
    assert config.input_type == "full_3d"
    input_channels = 1

    if not config.architecture and not config.drop_rate == 0:
        raise ValueError(
            f"Dropout not supported for architecture " + "{config.architecture}"
        )

    if config.architecture.startswith("timedist_"):
        main_architecture_name = config.architecture.split("timedist_")[-1]

        input_channels = 3 if config.load_pretrained_model_weights else 1

        kwargs = {
            "pretrained": config.load_pretrained_model_weights,
            "progress": True,
            "config": config,
        }

        main_architecture = resnet18_2d

        if (
            main_architecture_name.startswith("resnet")
            and config.network_norm_type == "group_norm"
        ):
            kwargs["norm_layer"] = lambda num_channels: torch.nn.GroupNorm(
                32, num_channels
            )
        elif (
            main_architecture_name.startswith("resnet")
            and config.network_norm_type == "instance_norm"
        ):
            kwargs["norm_layer"] = lambda num_channels: torch.nn.InstanceNorm2d(
                num_channels
            )

        if config.mil_pooling_function in [
            "attention",
            "attention_with_distribution",
            "distribution",
        ]:
            kwargs["num_classes"] = config.mil_pooling_function_attention_L
        if config.mil_pooling_function == "transmil":
            kwargs["num_classes"] = 1024

        architecture = lambda *args, **kwargs: MIL(
            main_architecture(*args, **kwargs), config=config
        )
    elif config.architecture == "resnet18":
        architecture = resnet18_2d

        kwargs = {
            "pretrained": False,
            "progress": True,
            "config": config,
        }

        if config.network_norm_type == "group_norm":
            kwargs["norm_layer"] = lambda num_channels: torch.nn.GroupNorm(
                32, num_channels
            )
    else:
        raise ValueError(f"Unknown architecture {config.architecture}")

    if "num_classes" not in kwargs:
        kwargs["num_classes"] = len(config.label_names)

    return architecture(input_channels=input_channels, **kwargs)


def get_optimizer(config, model):
    if config.custom_model_finetuning:
        params = filter(
            lambda p: p.requires_grad, model.parameters()
        )  # Layers to freeze have been flagged
        # with requires_grad==False
    else:
        # For earlier models the parameters with requires_grad=False were still
        # included in the optimizer. Must construct the optimizer in the same way
        # to load optimizer_state_dict of earlier models without errors
        params = model.parameters()
    return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)


def initialize_model(config, train_set=None, sample_size=None, sample_duration=None):
    model = make_model(config, train_set, sample_size, sample_duration)
    return model


def download_with_progress(url, model_path):
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(model_path),
    ) as t:
        urllib.request.urlretrieve(url, model_path, reporthook=t.update_to)


def download_model_weights_if_available(config, model_type, model_path):
    with open("ubix/public_models_urls.json", "r") as f:
        public_models_urls = json.load(f)

    model_name = get_model_name(config).replace("#", "/")

    print(f"Attempting to find public weights online for {model_name} ({model_type}).")
    if (model_name in public_models_urls) and (
        model_type in public_models_urls[model_name]
    ):
        url = public_models_urls[model_name][model_type]
        print(
            f"Found weights online for {model_name} ({model_type}), now downloading..."
        )

        download_with_progress(url, model_path)

        print(f"Finished downloading public weights for {model_name} ({model_type}).")
    else:
        print(
            f"Public weights for {model_name} ({model_type}) do not seem to be available"
        )


def load_model(config, model_type, train_set=None, device="cpu", load_optimizer=True):
    """
    Loads a model. See definition of model_type in function ```save_model```.
    """

    model = initialize_model(config, train_set=train_set)
    model.to(device)
    optimizer = get_optimizer(config, model)

    model_path = get_saved_model_path(config, model_type)

    # map_location = None if torch.cuda.is_available() else {'cuda:0': 'cpu'}
    map_location = device

    os.system(f"ls {Path(model_path).parent}")

    if not os.path.exists(model_path):
        download_model_weights_if_available(config, model_type, model_path)

    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    iteration = checkpoint["iteration"]
    loss = checkpoint["loss"]
    random_states = checkpoint["random_states"]

    for k, v in checkpoint["best_val_scalars"].items():
        if hasattr(wandb.run, "summary"):
            wandb.run.summary[k] = v

    if "latest_es_monitor_batch" in checkpoint:
        latest_es_monitor_batch = checkpoint["latest_es_monitor_batch"]
    else:
        latest_es_monitor_batch = None

    return (
        model,
        optimizer,
        epoch,
        iteration,
        loss,
        random_states,
        latest_es_monitor_batch,
    )


def perform_merge_classes(
    old_vector, classes_to_merge, classes_to_exclude, merge_function
):
    new_vector = copy.copy(old_vector)
    classes_to_remove = (
        [] if classes_to_exclude is None else copy.deepcopy(classes_to_exclude)
    )

    for combination in classes_to_merge:
        smallest_class = min(combination)
        combination_label_names = [old_vector[c] for c in combination]

        if None in combination_label_names:
            raise ValueError(
                f"classes_to_merge {classes_to_merge} has at least one class in "
                + "multiple combinations"
            )

        new_vector[smallest_class] = merge_function(combination_label_names)

        for label_i in range(len(new_vector)):
            if label_i in combination and label_i != smallest_class:
                classes_to_remove.append(label_i)

    new_vector = [
        label for i, label in enumerate(new_vector) if i not in classes_to_remove
    ]

    return new_vector


def update_config_dataset(config):
    if config.data_set == "EUGENDA_OCT":
        if config.merge_classes is not None:
            config.update(
                {
                    "label_names": perform_merge_classes(
                        config.label_names,
                        config.merge_classes,
                        config.exclude_classes,
                        lambda x: " + ".join(x),
                    )
                },
                allow_val_change=True,
            )
    else:
        if config.merge_classes is not None:
            config.update(
                {
                    "label_names": perform_merge_classes(
                        config.label_names,
                        config.merge_classes,
                        config.exclude_classes,
                        lambda x: " + ".join(x),
                    )
                },
                allow_val_change=True,
            )
    config.update({"n_classes": len(config.label_names)}, allow_val_change=True)


"""
Data utils
"""


def one_hot(lbl, n_classes):
    assert type(lbl) is int
    out = np.zeros(n_classes)
    out[lbl] = 1

    return out


def convert_label(config, y_idx):
    if config.merge_classes is not None:
        y_idx = one_hot(y_idx, config.n_classes_before_merge)
        y_idx = perform_merge_classes(
            y_idx, config.merge_classes, config.exclude_classes, np.sum
        )
        y_idx = int(np.argmax(y_idx))

    if config.soft_ordinal_target:
        y_idx = y_idx / (config.n_classes - 1)

    return y_idx


# # this is necessary because otherwise multiprocessing is compaining
# # @dataclass
# class ConfigData:
#     def __init__(self, config) -> None:
#         try:
#             items = config.items()
#         except KeyError:
#             items = config._items.items()

#         for k, v in items:
#             setattr(self, k, v)


def get_augmentation_transforms(config, translate_range_z=2):
    device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1 and config.gpu_aug else "cpu"
    )

    return [
        RandAffined(
            keys=["img"],
            prob=0.15,
            # prob=1.,
            device=device,
            rotate_range=[20 / 360 * 2 * np.pi, 0, 0],
            # rotate_range=[0, 20 / 360 * 2 * np.pi, 20 / 360 * 2 * np.pi],
            shear_range=[(0, 0), (0, 0), (0, 0), (-0.1, 0.1), (0, 0), (-0.1, 0.1)],
            # Shear matrix (NB: tensor has shape z, y, x):
            # 1                         h_zy (shear_range[0])    h_zx (shear_range[1])    0
            # h_yz (shear_range[2])     1                        h_yx (shear_range[3])    0
            # h_xz (shear_range[4])     h_xy (shear_range[5])    1                        0
            # 0                         0                        0                        1
            translate_range=[translate_range_z, 20, 20],
            scale_range=[0, 0.1, 0.1],
            # as_tensor_output=False,
        ),
        RandFlipd(
            keys=["img"],
            prob=0.15,
            spatial_axis=2,
            # as_tensor_output=False,
        ),
        RandGaussianNoised(
            keys=["img"],
            std=config.aug_noise_std,  # 0.05,
            prob=config.aug_noise_prob,  # .15,
        ),
        RandAdjustContrastd(
            keys=["img"],
            gamma=(config.aug_contrast_lower, config.aug_contrast_upper),
            # (.75, 3),
            prob=config.aug_contrast_prob,
        ),
    ]


def get_tta_transforms(config, translate_range_z=0, device=None):
    device = get_device()

    return [
        RandAffined(
            keys=["img"],
            prob=0.15,
            # prob=1.,
            device=device,
            rotate_range=[20 / 360 * 2 * np.pi, 0, 0],
            # rotate_range=[0, 20 / 360 * 2 * np.pi, 20 / 360 * 2 * np.pi],
            shear_range=[(0, 0), (0, 0), (0, 0), (-0.1, 0.1), (0, 0), (-0.1, 0.1)],
            # Shear matrix (NB: tensor has shape z, y, x):
            # 1                         h_zy (shear_range[0])    h_zx (shear_range[1])    0
            # h_yz (shear_range[2])     1                        h_yx (shear_range[3])    0
            # h_xz (shear_range[4])     h_xy (shear_range[5])    1                        0
            # 0                         0                        0                        1
            translate_range=[translate_range_z, 20, 20],
            scale_range=[0, 0.1, 0.1],
            # as_tensor_output=False,
        ),
        RandFlipd(
            keys=["img"],
            prob=0.15,
            spatial_axis=2,
            # as_tensor_output=False,
        ),
        RandGaussianNoised(
            keys=["img"],
            std=config.aug_noise_std,  # 0.05,
            prob=config.aug_noise_prob,  # .15,
        ),
        RandAdjustContrastd(
            keys=["img"],
            gamma=(config.aug_contrast_lower, config.aug_contrast_upper),  # (.75, 3),
            prob=config.aug_contrast_prob,
        ),
    ]


class FixedAxisSpacingd(MapTransform):
    def __init__(self, keys: KeysCollection, fixed_axis: int, *args, **kwargs) -> None:
        super().__init__(keys)
        self.spacing_transform = Spacingd(keys, *args, **kwargs)
        self.fixed_axis = fixed_axis

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            n = data[key].shape[self.fixed_axis]
            a = []
            for axis_idx in range(n):
                data_idx = copy.deepcopy(data)

                # Get one slice
                # if type(data_idx[key]) is torch.Tensor:
                #     data_idx[key] = data_idx[key].numpy()

                assert self.fixed_axis == -1

                data_idx[key] = data_idx[key][..., axis_idx]

                # data_idx[key] = torch.from_numpy(data_idx[key])

                # print("data_idx[key].shape:", data_idx[key].shape)
                ai = self.spacing_transform(data_idx)
                a.append(ai[key])

            d[key] = np.stack(a, axis=self.fixed_axis)
        return d


try:
    wandb_config_class = wandb.wandb_config.Config
except AttributeError:
    wandb_config_class = wandb.Config


class ArtificialArtifactsd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        config,
        prob: float = 0.1,
        group_size_override=None,
        locations_override=None,
    ) -> None:
        super().__init__(keys)
        self.prob = prob

        self.config = config

        self._do_transform = False

        self.group_size_override = group_size_override
        self.locations_override = locations_override

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random_sample() < self.prob

    def random_bscan_nr(self, bscan_count):
        mn = self.config.artificial_artifacts_bscan_size_rel_min
        mx = self.config.artificial_artifacts_bscan_size_rel_max

        return int(round((self.R.random_sample() * (mx - mn) + mn) * bscan_count))

    def random_val(self, mn=0.0, mx=1.0):
        return self.R.random_sample() * (mx - mn) + mn

    def random_location(self, bscan_count, black_count):
        return int(round(self.R.random_sample() * (bscan_count - black_count)))

    @staticmethod
    def to_tensor(a):
        if type(a) is np.ndarray:
            return torch.from_numpy(a)
        else:
            return a.as_tensor()

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        self.randomize()
        d = dict(data)

        if not self._do_transform:
            d["artificial_artifacts"] = None
            return d

        for key in self.keys:
            n_groups = (
                round(self.R.random_sample()) + 1
                if self.config.artificial_artifacts_override_ngroups is None
                else int(self.config.artificial_artifacts_override_ngroups)
            )

            bscan_count = d[key].shape[1]

            group_size = [self.random_bscan_nr(bscan_count) for _ in range(n_groups)]

            locations = [self.random_location(bscan_count, sz) for sz in group_size]

            if self.group_size_override:
                group_size = self.group_size_override
            if self.locations_override:
                locations = self.locations_override

            out = d[key]

            for loc, sz in zip(locations, group_size):
                if self.config.artificial_artifacts == "black_bscans":
                    noisy_black = RandGaussianNoise(
                        prob=1,
                        mean=out.median().item(),
                        std=out.std().item(),
                    )
                    zrs = np.zeros((sz, out.shape[2], out.shape[3]))

                    out[0, loc : loc + sz] = self.to_tensor(noisy_black(zrs))
                elif self.config.artificial_artifacts == "vertically_flipped_bscans":
                    flipper = Flip(spatial_axis=1)
                    out[:, loc : loc + sz] = self.to_tensor(
                        flipper(out[:, loc : loc + sz].numpy())
                    )
                elif self.config.artificial_artifacts == "shadow":
                    sub_image = out[:, loc : loc + sz].numpy()

                    ascan_count = sub_image.shape[-1]
                    mu = self.random_val(mn=0, mx=ascan_count)
                    sigma = self.random_val(
                        mn=ascan_count * 0.25, mx=ascan_count * 0.75
                    )
                    x = np.linspace(0, ascan_count, ascan_count)
                    one_line = scipy.stats.norm.pdf(x, mu, sigma)
                    shadow = np.tile(one_line, sub_image.shape[:3] + (1,))
                    shadow_norm = (shadow - shadow.min()) / (
                        shadow.max() - shadow.min()
                    )

                    new_sub_image = sub_image * (1 - shadow_norm)

                    out[:, loc : loc + sz] = self.to_tensor(new_sub_image)
                elif self.config.artificial_artifacts == "noise":
                    sub_image = out[:, loc : loc + sz].numpy()

                    noise = RandGaussianNoise(prob=1, mean=0, std=out.std().item() * 4)
                    sub_image_new = noise(sub_image)
                    out[0, loc : loc + sz] = self.to_tensor(sub_image_new)
                elif self.config.artificial_artifacts == "none":
                    continue
                else:
                    raise ValueError(
                        f"Unknown value {self.config.artificial_artifacts}"
                        f" for self.config.artificial_artifacts."
                    )

            d["artificial_artifacts"] = {
                "group_size": group_size,
                "locations": locations,
            }

            d[key] = out

        return d


def get_data_set(
    config,
    subsets=("train", "val"),
    random_seed=0,
    shuffle_train=False,
    override_aug_transforms=None,
):
    root = config.data_root

    to_return = []
    for subset in subsets:
        root_subset = os.path.join(root, subset)
        labels_path = os.path.join(root_subset, config.label_json_name)
        with open(labels_path, "r") as f:
            labels_raw = json.load(f)

        labels = {}

        for idx, label in labels_raw.items():
            if not config.label_field:
                lbl = label
            else:
                lbl = label[config.label_field]

            if lbl in config.exclude_classes:
                continue

            labels[idx] = lbl

        if len(labels) == 0:
            to_return.append(None)
            continue

        def make_balanced_set(data, y):
            y_unique, y_count = np.unique(y, return_counts=True)
            min_count = np.min(y_count)

            ids = []
            for yi in y_unique:
                ids += list(
                    np.random.permutation(np.argwhere(y == yi)[:, 0])[:min_count]
                )
            ids = np.random.permutation(ids)

            data = [data[idx] for idx in ids]
            y = y[ids]

            return data, y

        data = [
            {
                "img": os.path.join(root_subset, f"{nr}.nii.gz"),
                "label": convert_label(config, label),
                "nr": nr,
            }
            for nr, label in labels.items()
        ]
        y = [a["label"] for a in data]

        if shuffle_train and subset == "train":
            c = list(zip(data, y))
            random.shuffle(c)
            data, y = zip(*c)
            y = np.array(y)

        if (config.balanced_train_set and subset == "train") or (
            config.balanced_val_set and subset == "val"
        ):
            data, y = make_balanced_set(data, y)

        spacing_transform = Spacingd(
            keys=["img"], pixdim=config.spacing, mode="nearest"
        )

        transforms = [
            LoadImaged(keys=["img"]),
            ScaleIntensityd(keys=["img"], minv=0.0, maxv=config.scale_intensity_max),
            CastToTyped(
                keys=["img"],
                dtype=np.float32 if config.scale_intensity_max == 1.0 else np.int,
            ),
            # AddChanneld(keys=["img"])
            EnsureChannelFirstd(keys=["img"]),
        ]

        if config.resample_input:
            transforms.append(spacing_transform)

        transforms += [
            Orientationd(keys=["img"], axcodes=config.orientationd_axcodes),
            CastToTyped(keys=["img"], dtype=np.float32),
        ]

        if not config.resample_input:
            transforms.append(Flipd(keys=["img"], spatial_axis=1))

        transforms += [
            ToTensord(keys=["img", "label"]),
        ]

        if config.artificial_artifacts:
            # config_data = ConfigData(config)

            transforms.append(
                ArtificialArtifactsd(
                    keys=["img"],
                    #    config=config,
                    config=config_data,
                    prob=config.artificial_artifacts_prob,
                )
            )

        if override_aug_transforms is not None and len(override_aug_transforms) > 0:
            print("using override_aug_transforms")
            transforms += override_aug_transforms
        elif config.use_data_augmentation and subset == "train":
            print("using get_augmentation_transforms")
            transforms += get_augmentation_transforms(config)

        transforms = Compose(transforms)
        transforms.set_random_state(seed=random_seed)

        persistent_folder = Path(f"monai_cache/monai_{wandb.run.name}_{time.time()}")
        if not hasattr(config, "remove_temp") or (
            hasattr(config, "remove_temp") and config.remove_temp
        ):
            shutil.rmtree(persistent_folder, ignore_errors=True)

        if not persistent_folder.parent.exists():
            persistent_folder.parent.mkdir(exist_ok=True)

        persistent_folder.mkdir(exist_ok=True)

        ds = PersistentDataset(
            data=data, transform=transforms, cache_dir=persistent_folder
        )
        # ds = Dataset(data=data, transform=transforms)

        ds.y = y
        ds.nrs = list(labels.keys())

        to_return.append(ds)

    return to_return


def get_class_weight(config, train_set, device):
    int_classes = train_set.y

    class_weight = compute_class_weight(
        "balanced", classes=np.unique(int_classes), y=list(int_classes)
    )

    # Fill in classes with weight 0 that aren't in y
    class_weight = list(class_weight)
    for c in range(np.max(int_classes)):
        if c not in int_classes:
            class_weight = class_weight[:c] + [0] + class_weight[c:]
    class_weight = np.array(class_weight)

    print("class_weight:", class_weight)
    class_weight = torch.from_numpy(class_weight.copy()).float()
    class_weight = class_weight.to(device)

    return class_weight


from torch import Tensor
from torch.utils.data import Sampler
from typing import Iterator, Sequence


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")


def make_data_loaders(
    config, train_set, val_set, device, test_set=None, force_no_balance=False
):
    train_sampler = None
    train_shuffle = True

    def random_sampler(data_set):
        y = data_set.y
        y = torch.from_numpy(y)

        weights = get_class_weight(config, data_set, device)
        samples_weights = weights[y.to(device)]

        # if str(device).startswith('mps'):
        samples_weights = samples_weights.to(torch.float32)

        # sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights))
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        return sampler

    if config.balanced_train_sampling and not force_no_balance:
        train_sampler = random_sampler(train_set)
        train_shuffle = False

    # with std_report('Making DataLoaders'):
    if train_set is not None and len(train_set) > 0:
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=train_shuffle,
            num_workers=config.num_workers,
            sampler=train_sampler,
        )
    else:
        train_loader = None

    if config.balanced_val_sampling and not force_no_balance:
        sampler = random_sampler(val_set)
    else:
        sampler = None

    val_loader = (
        DataLoader(
            val_set,
            batch_size=config.inference_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=config.num_workers,
        )
        if val_set is not None
        else None
    )

    to_return = [train_loader, val_loader]

    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=config.inference_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        to_return.append(test_loader)

    return to_return


def tensor_or_tuple_of_tensors_to_numpy(a):
    if type(a) is tuple:
        return [[ai.detach().cpu().numpy() for ai in a]]
    else:
        return a.detach().cpu().numpy()


def run_silently(cmd):
    """
    Executes a shell command without printing the stdout.

    Args:
        cmd (str): The command to be executed.
    """
    subprocess.run(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
