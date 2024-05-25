import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pathlib import Path
from ubix.models.TransMIL import TransMIL

import torch
import torch.nn.functional as F
import numpy as np


def extract_patches(x_reshape, patch_size, samples, input_size_z, return_coords=False):
    """
    Extract patches from a tensor and optionally return the coordinates of the patches.

    Parameters:
    x_reshape (torch.Tensor): The input tensor from which to extract patches.
    patch_size (tuple): The dimensions of the patches.
    samples (int): The number of samples in x_reshape.
    input_size_z (int): The size of the z dimension in x_reshape.
    return_coords (bool): Whether to return the coordinates of the patches.

    Returns:
    torch.Tensor: The extracted patches.
    list (optional): The coordinates of the patches.
    """
    # Extract patch dimensions for readability
    patch_height, patch_width = patch_size

    # Get the dimensions of the input tensor
    _, channels, input_size_y, input_size_x = x_reshape.size()

    # Determine the number of patches in each dimension
    n_patches_y = (input_size_y + patch_height - 1) // patch_height
    n_patches_x = (input_size_x + patch_width - 1) // patch_width
    n_patches_per_slice = n_patches_y * n_patches_x

    # Initialize a list to hold the patches and coordinates
    patches = []
    coords = []

    # Loop through the input tensor to extract patches
    for s in range(samples):
        for z in range(input_size_z):
            for i in range(n_patches_y):
                for j in range(n_patches_x):
                    # Calculate the top left corner coordinates of the patch
                    y_start, x_start = i * patch_height, j * patch_width

                    # Extract patch
                    patch = x_reshape[
                        s * input_size_z + z,
                        :,
                        y_start : y_start + patch_height,
                        x_start : x_start + patch_width,
                    ]

                    # print(patch.size())

                    # Pad the patch to the desired size if necessary
                    pad_y = patch_height - patch.size(1)
                    pad_x = patch_width - patch.size(2)
                    patch = F.pad(patch, (0, pad_x, 0, pad_y))

                    # Append the patch and coordinates
                    patches.append(patch.unsqueeze(0))
                    coords.append((s, z, y_start, x_start))

    # Stack the patches along a new dimension
    patches_tensor = torch.cat(patches, dim=0)

    if return_coords:
        return patches_tensor, coords
    else:
        return patches_tensor


# Usage:
# patches_tensor, coords = extract_patches(x_reshape, patch_size, samples, input_size_z, return_coords=True)


def samples_and_instances_to_first_axis(x):
    x_reshape = x.contiguous().view(
        -1, x.size(1), x.size(3), x.size(4)
    )  # (samples * input_size_z, channels, input_size_y, input_size_x)

    return x_reshape


class MIL(nn.Module):
    def __init__(self, module, batch_first=True, config=None):
        super(MIL, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.config = config
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        if self.config.mil_pooling_function == "attention":
            # from https://github.com/max-ilse/AttentionDeepMIL/blob/master/model.py

            self.attention = nn.Sequential(
                nn.Linear(
                    self.config.mil_pooling_function_attention_L,
                    self.config.mil_pooling_function_attention_D,
                ),
                nn.Tanh(),
                nn.Linear(
                    self.config.mil_pooling_function_attention_D,
                    self.config.mil_pooling_function_attention_K,
                ),
            )

            if not self.config.mil_pooling_function_attention_instancelevel:
                # embedding-level
                self.classifier = nn.Linear(
                    self.config.mil_pooling_function_attention_L
                    * self.config.mil_pooling_function_attention_K,
                    self.config.n_classes,
                )
        elif self.config.mil_pooling_function == "transmil":
            self.transmil = TransMIL(n_classes=self.config.n_classes)
        elif self.config.mil_pooling_function == "attention_with_distribution":
            # from https://github.com/onermustafaumit/mil_pooling_filters/blob/main/MIL_tasks_on_histopathology_image_patches/multi_class_classification/model.py

            num_bins = self.config.mil_pooling_function_distribution_numbins
            num_features = self.config.mil_pooling_function_attention_L
            sigma = self.config.mil_pooling_function_distribution_sigma
            num_classes = self.config.n_classes

            self._attention = Attention(num_in=num_features)
            self._attention2 = Attention2(num_in=num_features)
            self._distribution_pooling_filter = DistributionPoolingFilter(
                num_bins=num_bins, sigma=sigma
            )
            self._representation_transformation = RepresentationTransformation(
                num_in=num_features * num_bins, num_out=num_classes, config=self.config
            )
        elif self.config.mil_pooling_function == "distribution":
            # from https://github.com/onermustafaumit/mil_pooling_filters/blob/main/MIL_tasks_on_histopathology_image_patches/multi_class_classification/model.py

            num_bins = self.config.mil_pooling_function_distribution_numbins
            num_features = self.config.mil_pooling_function_attention_L
            sigma = self.config.mil_pooling_function_distribution_sigma
            num_classes = self.config.n_classes

            self._distribution_pooling_filter = DistributionPoolingFilterBase(
                num_bins=num_bins, sigma=sigma
            )
            self._representation_transformation = RepresentationTransformation(
                num_in=num_features * num_bins, num_out=num_classes, config=self.config
            )
        elif self.config.mil_pooling_function == "dense":
            assert (
                config.full_oct_shape[0] > 0
            ), "Assuming z-size of oct is specified in config object (so not -1)"
            self.final_linear = nn.Linear(
                self.config.n_classes * self.config.full_oct_shape[0],
                self.config.n_classes,
            )
            nn.init.constant_(self.final_linear.bias, 0)

    def forward(
        self,
        x,
        return_instances=False,
        instances_softmaxed=False,
        return_devries_confidence=False,
        y_instances=None,
    ):
        # Assuming shape of x: (N, C, Z, Y, X)

        if y_instances is None:
            if self.config.load_pretrained_model_weights:
                x = torch.cat([x] * 3, dim=1)

            # Squash samples and timesteps into a single axis
            x_reshape = samples_and_instances_to_first_axis(
                x
            )  # (samples * input_size_z, channels, input_size_y, input_size_x)

            if self.config.mil_instance_definition == "patches_2d":
                print("x_reshape.shape before extract_patches:", x_reshape.shape)
                x_reshape = extract_patches(
                    x_reshape,
                    self.config.mil_instance_patch_size,
                    x.shape[0],
                    x.shape[2],
                )
                out_file = "x_reshape.pt"
                if not Path(out_file).exists():
                    with open(out_file, "wb") as f:
                        torch.save(x_reshape.detach().cpu(), f)
                print("x_reshape.shape after extract_patches:", x_reshape.shape)

            confidence = None

            if self.config.use_gradient_checkpoints_per_module:
                if return_devries_confidence:
                    raise NotImplemented(
                        "return_devries_confidence not implemented when self.config.use_gradient_checkpoints_per_module == True."
                    )

                # Without the dummy tensor, the gradient will not be calculated until the input
                y = []
                for sample_idx_start in range(
                    0, x_reshape.shape[0], self.config.modules_per_checkpoint
                ):
                    samples = x_reshape[
                        sample_idx_start : sample_idx_start
                        + self.config.modules_per_checkpoint
                    ]
                    print("129 |", sample_idx_start, "| samples.shape:", samples.shape)
                    checkpoint = torch.utils.checkpoint.checkpoint(
                        lambda q, _: self.module(q), samples, self.dummy_tensor
                    )
                    y.append(checkpoint)
                y = torch.cat(y, 0)
            else:
                if return_devries_confidence and self.config.devries_confidence:
                    y, confidence = self.module(
                        x_reshape, return_devries_confidence=return_devries_confidence
                    )
                else:
                    y = self.module(x_reshape)

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), -1, y.size(-1)
                )  # (samples, timesteps, output_size)

                if confidence is not None:
                    confidence = confidence.contiguous().view(
                        x.size(0), -1, confidence.size(-1)
                    )
            else:
                y = y.view(
                    -1, x.size(1), y.size(-1)
                )  # (timesteps, samples, output_size)

                if confidence is not None:
                    confidence = confidence.view(-1, x.size(1), confidence.size(-1))

            y_instances = y
            confidence_instances = confidence
        else:
            y = y_instances

        # Reduce output per slice
        if self.config.mil_pooling_function == "max":
            y, _ = torch.max(y, 1)
        elif self.config.mil_pooling_function in ["avg", "mean"]:
            y = torch.mean(y, 1)
        elif self.config.mil_pooling_function == "dense":
            y = torch.flatten(y, 1)
            y = self.final_linear(y)
        elif self.config.mil_pooling_function == "transmil":
            y = self.transmil(data=y)["logits"]
        elif self.config.mil_pooling_function == "attention":
            # y, _ = torch.max(y, 1)
            # y = y[..., -self.config.n_classes:]

            A = self.attention(y)  # BxNxK (B is batch size)
            A = torch.transpose(A, 2, 1)  # BxKxN
            A = F.softmax(A, dim=2)  # softmax over N, (BxKxN)

            M = torch.bmm(A, y)  # BxKxL

            assert self.config.mil_pooling_function_attention_K == 1, "Assuming K is 1"

            if self.config.mil_pooling_function_attention_instancelevel:
                y = M[
                    :, 0, -self.config.n_classes :
                ]  # [:(all samples in batch), 0(K is assumed to be 1), only last classes]
                y_instances = y_instances[..., -self.config.n_classes :]
            else:
                y = M[:, 0]
                y = self.classifier(y)
        elif self.config.mil_pooling_function == "attention_with_distribution":
            y = torch.sigmoid(y)

            num_instances = y.shape[1]
            num_features = self.config.mil_pooling_function_attention_L
            num_bins = self.config.mil_pooling_function_distribution_numbins

            attention_values = self._attention(y)
            attention_values2 = self._attention2(y)
            y = torch.reshape(y, (-1, num_instances, num_features))
            out = attention_values2 * y
            out = self._distribution_pooling_filter(out, attention_values)
            out = torch.reshape(out, (-1, num_features * num_bins))

            y = self._representation_transformation(out)
        elif self.config.mil_pooling_function == "distribution":
            # print("y.max():", y.max())
            # print("y.min():", y.min())
            # print("y.shape:", y.shape)
            y = torch.sigmoid(y)

            num_instances = y.shape[1]
            num_features = self.config.mil_pooling_function_attention_L
            num_bins = self.config.mil_pooling_function_distribution_numbins

            y = torch.reshape(y, (-1, num_instances, num_features))

            # print("y.max():", y.max())
            # print("y.min():", y.min())
            # print("y.shape:", y.shape)

            out = self._distribution_pooling_filter(y)

            # print("out.max():", out.max())
            # print("out.min():", out.min())
            # print("out.shape:", out.shape)

            out = torch.reshape(out, (-1, num_features * num_bins))
            # print("out.max():", out.max())
            # print("out.min():", out.min())
            # print("out.shape:", out.shape)

            y = self._representation_transformation(out)

            # print("y.max():", y.max())
            # print("y.min():", y.min())
            # print("y.shape:", y.shape)

            ysm = torch.softmax(y, -1)

            # print("ysm.max():", ysm.max())
            # print("ysm.min():", ysm.min())
            # print("ysm.shape:", ysm.shape)
        else:
            raise ValueError(
                f"Unknown value for config.mil_pooling_function: {self.config.mil_pooling_function}"
            )

        # Reduce output per slice
        if return_devries_confidence:
            if self.config.devries_mil_pool == "max":
                confidence, _ = torch.max(confidence, 1)
            elif self.config.devries_mil_pool == "avg":
                confidence = torch.mean(confidence, 1)
            elif self.config.devries_mil_pool == "min":
                # print("80 | confidence.shape:", confidence.shape)
                confidence, _ = torch.min(confidence, 1)
                # print("82 | confidence.shape:", confidence.shape)
            else:
                raise ValueError(
                    f"Unknown value for config.devries_mil_pool: {self.config.devries_mil_pool}"
                )

        if return_instances:
            if instances_softmaxed:
                y_instances = torch.softmax(y_instances, -1)

            if return_devries_confidence:
                return (y, y_instances), (confidence, confidence_instances)

            return y, y_instances

        if return_devries_confidence:
            return y, confidence
        else:
            return y


class DistributionPoolingFilterBase(nn.Module):
    __constants__ = ["num_bins", "sigma"]

    def __init__(self, num_bins=1, sigma=0.1):
        super(DistributionPoolingFilterBase, self).__init__()

        self.num_bins = num_bins
        self.sigma = sigma
        self.alfa = 1 / math.sqrt(2 * math.pi * (sigma**2))
        self.beta = -1 / (2 * (sigma**2))

        sample_points = torch.linspace(
            0, 1, steps=num_bins, dtype=torch.float32, requires_grad=False
        )
        self.register_buffer("sample_points", sample_points)

    def extra_repr(self):
        return "num_bins={}, sigma={}".format(self.num_bins, self.sigma)

    def forward(self, data):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(
            batch_size, num_instances, num_features, 1
        )
        # print('sample_points.size():', sample_points.size())
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff**2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


class DistributionPoolingFilter(DistributionPoolingFilterBase):
    def __init__(self, num_bins=1, sigma=0.1):
        super(DistributionPoolingFilter, self).__init__(num_bins, sigma)

    def forward(self, data, attention_weights):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(
            batch_size, num_instances, num_features, 1
        )
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff**2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        # attention_weights.size() --> (batch_size,num_instances)
        attention_weights = torch.reshape(
            attention_weights, (batch_size, num_instances, 1, 1)
        )
        # attention_weights.size() --> (batch_size,num_instances,1,1)
        attention_weights = attention_weights.repeat(1, 1, num_features, self.num_bins)
        # attention_weights.size() --> (batch_size,num_instances,num_features,num_bins)

        result = attention_weights * result
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


class RepresentationTransformation(nn.Module):
    def __init__(self, num_in=32, num_out=10, config=None):
        super(RepresentationTransformation, self).__init__()

        if config.use_dropout_distribution_mil:
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_in, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, num_out),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_in, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, num_out),
            )

    def forward(self, x):
        out = self.fc(x)

        return out


class Attention(nn.Module):
    def __init__(self, num_in=32):
        super(Attention, self).__init__()
        # self._num_instances = num_instances
        self.fc = nn.Sequential(nn.Linear(num_in, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        num_instances = x.shape[1]
        out = self.fc(x)
        out = torch.reshape(out, (-1, num_instances, 1))
        out = F.softmax(out, dim=1)

        return out


class Attention2(nn.Module):
    def __init__(self, num_in=32):
        super(Attention2, self).__init__()
        # self._num_instances = num_instances
        self.fc = nn.Sequential(nn.Linear(num_in, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        num_instances = x.shape[1]

        out = self.fc(x)
        out = torch.reshape(out, (-1, num_instances, 1))
        out = torch.sigmoid(out)

        return out
