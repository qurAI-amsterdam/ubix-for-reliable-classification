import os

# os.chdir(os.path.dirname(__file__))

from distutils.dir_util import copy_tree
import torch
import wandb

from ubix.utils import *
from ubix.metrics import *
from tqdm.auto import tqdm
import SimpleITK as sitk

import numpy as np

import time

import argparse


def make_scalars_zero(metrics):
    """
    Initializes a dictionary with metric names as keys and zero as their values.

    Args:
        metrics (dict): Dictionary of metric functions.

    Returns:
        dict: Dictionary with metric names as keys and zero as their values.
    """

    out = {}

    out["loss"] = 0.0
    for metric_name in metrics.keys():
        out[metric_name] = 0.0

    return out


def get_metrics_lower_is_better():
    """
    Returns a list of metrics for which lower values are better.

    Returns:
        list: List of metric names where lower values indicate better performance.
    """

    return ["loss", "aece", "ece", "nll", "brier_score"]


def make_best_scalars(metrics):
    """
    Initializes the best scalars for tracking performance in wandb run summary.

    Args:
        metrics (dict): Dictionary of metric functions.
    """
    # For loss, the lower the better will be assumed
    wandb.run.summary["best_loss"] = np.inf

    # For metrics, the higher the better will be assumed, unless metric name is in ```metrics_lower_is_better```
    metrics_lower_is_better = get_metrics_lower_is_better()
    for m in metrics:
        if m in metrics_lower_is_better:
            wandb.run.summary[f"best_{m}"] = np.inf
        else:
            wandb.run.summary[f"best_{m}"] = -np.inf


metrics = {
    "quadratic_weighted_kappa": quadratic_weighted_kappa,
    "linearly_weighted_kappa": linearly_weighted_kappa,
    "kappa": kappa,
    "acc": accuracy,
    "auc": auc_roc,
    "auc_offset3": auc_roc_offset3,
    "auc_offset4": auc_roc_offset4,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "aece": aece,
    "brier_score": brier_score,
    "nll": nll,
    "fdauc": fdauc,  # Failure detection AUC
}


def get_metrics(config):
    """
    Filters out metrics based on the configuration.

    Args:
        config (obj): Configuration object containing the experiment parameters.

    Returns:
        dict: Dictionary of filtered metric functions.
    """
    for m in config.exclude_metrics:
        if m in metrics:
            del metrics[m]

    return metrics


class Train:
    def __init__(self, config, random_seed=0):
        """
        Initialize the training environment.

        Args:
            config (obj): Configuration object containing the experiment parameters.
            random_seed (int, optional): Seed for random number generators. Defaults to 0.
        """
        self.config = config

        set_random_seeds(seed=random_seed)
        print_config(self.config)
        print_jobid()
        make_model_results_folder(self.config)
        update_config_dataset(self.config)

        self.device = get_device()

        self.train_set, self.val_set = get_data_set(
            config, random_seed=random_seed, shuffle_train=True
        )

        self.train_loader, self.val_loader = make_data_loaders(
            config, self.train_set, self.val_set, self.device
        )
        self.metrics = get_metrics(self.config)

        if not self.latest_model_exists() or self.config.restart_if_model_exists:
            self.model = initialize_model(config, self.train_set)

            self.optimizer = get_optimizer(config, self.model)
            self.criterion = self.get_criterion(config)

            self.train_scalars = make_scalars_zero(self.metrics)
            make_best_scalars(self.metrics)

            self.epoch = 0
            self.batch = 0
            self.latest_es_monitor_batch = 0

            self.model.to(self.device)
        else:
            (
                self.model,
                self.optimizer,
                self.epoch,
                self.batch,
                self.criterion,
                random_states,
                self.latest_es_monitor_batch,
            ) = load_model(
                config, "latest", train_set=self.train_set, device=self.device
            )

            set_random_states(random_states, self.device)
            self.train_scalars = make_scalars_zero(self.metrics)

        wandb.watch(self.model)

        print("count_parameters(self.model):", count_parameters(self.model))

    def scalars_to_writer(
        self,
        set_name,
        total_scalars,
        n_batches,
        current_batch,
        y_true,
        outputs,
        metrics,
    ):
        """
        Logs scalar values to wandb and returns calculated metrics.

        Args:
            set_name (str): Name of the dataset (e.g., 'train', 'val').
            total_scalars (dict): Dictionary of scalar values.
            n_batches (int): Number of batches.
            current_batch (int): Current batch number.
            y_true (list): Ground truth labels.
            outputs (list): Model outputs.
            metrics (dict): Dictionary of metric functions.

        Returns:
            dict: Dictionary of metric values.
        """
        metric_values = {}

        loss_mean = total_scalars["loss"] / n_batches
        print(f"loss = {loss_mean}")

        metric_values["loss"] = loss_mean

        wandb.log({f"{set_name} loss": loss_mean}, step=current_batch)

        y_true_post = np.array(y_true)

        y_pred = outputs
        y_pred_post = np.array(y_pred)

        for metric_name, metric_func in metrics.items():
            metric_kwargs = {}
            metric_value = metric_func(y_true_post, y_pred_post, **metric_kwargs)
            wandb.log({f"{set_name} {metric_name}": metric_value}, step=current_batch)

            metric_values[metric_name] = metric_value

        return metric_values

    def get_criterion(self, config):
        """
        Determines and returns the appropriate loss criterion.

        Args:
            config (obj): Configuration object containing the experiment parameters.

        Returns:
            callable: Loss function.
        """
        if config.loss_type == "ce":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type {config.loss_type}")

    def latest_model_exists(self):
        """
        Checks if the latest model checkpoint exists.

        Returns:
            bool: True if the latest model checkpoint exists, False otherwise.
        """
        return os.path.exists(get_saved_model_path(self.config, "latest"))

    def train_batch(self, local_batch, local_labels, verbose=0):
        self.optimizer.zero_grad()

        kwargs = {}
        outputs = self.model(local_batch, **kwargs)

        loss = self.criterion(outputs, local_labels)
        loss.backward()
        self.optimizer.step()
        self.train_scalars["loss"] += loss.item()

        outputs_train_batch = list(tensor_or_tuple_of_tensors_to_numpy(outputs))
        y_true_train_batch = list(local_labels.cpu().numpy())

        return outputs_train_batch, y_true_train_batch

    def report_training(self, y_true_train, y_pred_train):
        self.scalars_to_writer(
            "train",
            self.train_scalars,
            self.config.tb_log_interval_train,
            self.batch,
            y_true_train,
            y_pred_train,
            self.metrics,
        )

    @staticmethod
    def check_monitor_improved(cur_scalars, monitor_name):
        lower_is_better = (
            monitor_name == "loss" or monitor_name in get_metrics_lower_is_better()
        )

        if lower_is_better:
            return cur_scalars[monitor_name] < wandb.run.summary[f"best_{monitor_name}"]
        else:
            return cur_scalars[monitor_name] > wandb.run.summary[f"best_{monitor_name}"]

    def report_validation(self):
        if self.config.model_eval:
            self.model.eval()
        else:
            self.model.train()

        val_scalars = make_scalars_zero(self.metrics)
        y_true_val, outputs_val = [], []
        with torch.set_grad_enabled(False):
            val_iteration = 0
            for sample in tqdm(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"validation while training {get_model_name(self.config)}",
            ):
                local_batch, local_labels = sample["img"], sample["label"]

                # Transfer to GPU
                local_batch, local_labels = local_batch.to(
                    self.device
                ), local_labels.to(self.device)

                kwargs = {}
                if self.config.devries_confidence:
                    kwargs["return_devries_confidence"] = self.config.devries_confidence

                outputs = self.model(local_batch, **kwargs)

                loss = self.criterion(outputs, local_labels)

                # Model computations
                val_scalars["loss"] += loss.item()

                outputs_val_batch = list(tensor_or_tuple_of_tensors_to_numpy(outputs))
                y_true_val_batch = list(local_labels.cpu().numpy())

                outputs_val += outputs_val_batch
                y_true_val += y_true_val_batch

                val_iteration += 1

                torch.cuda.empty_cache()

        metric_values = self.scalars_to_writer(
            "val",
            val_scalars,
            len(self.val_loader),
            self.batch,
            y_true_val,
            outputs_val,
            self.metrics,
        )

        for monitor_name in self.config.best_model_monitor:
            if self.check_monitor_improved(metric_values, monitor_name):
                if self.config.es_patience_monitor == monitor_name:
                    self.latest_es_monitor_batch = self.batch

                print(
                    f'Metric {monitor_name} improved from {wandb.run.summary[f"best_{monitor_name}"]} '
                    f"to {metric_values[monitor_name]}."
                )

                wandb.run.summary[f"best_{monitor_name}"] = metric_values[monitor_name]

                save_model(
                    self.config,
                    self.epoch,
                    self.batch,
                    self.model,
                    self.optimizer,
                    self.criterion,
                    get_random_states(self.device),
                    f"best_{monitor_name}",
                    self.latest_es_monitor_batch,
                )

        print("wandb.run.summary:", wandb.run.summary)

    def train(self, verbose=1):
        y_true_train, y_pred_train = [], []

        t_prev = time.time()

        # Loop over epochs
        for self.epoch in range(self.config.max_epochs):
            for sample in self.train_loader:
                local_batch, local_labels = sample["img"], sample["label"]

                print(
                    "Batch {}, time diff: {}".format(self.batch, time.time() - t_prev)
                )
                # Transfer to GPU if available
                local_batch, local_labels = local_batch.to(
                    self.device
                ), local_labels.to(self.device)

                y_pred_train_batch, y_true_train_batch = self.train_batch(
                    local_batch, local_labels, verbose=verbose
                )

                del sample, local_batch, local_labels

                y_pred_train += y_pred_train_batch
                y_true_train += y_true_train_batch

                if (
                    self.batch % self.config.tb_log_interval_train == 0
                    and self.batch != 0
                ):
                    self.report_training(y_true_train, y_pred_train)
                    self.train_scalars = make_scalars_zero(self.metrics)
                    y_true_train, y_pred_train = [], []

                if self.batch % self.config.latest_model_interval_iteration == 0:
                    save_model(
                        self.config,
                        self.epoch,
                        self.batch,
                        self.model,
                        self.optimizer,
                        self.criterion,
                        get_random_states(self.device),
                        "latest",
                        self.latest_es_monitor_batch,
                    )

                if self.batch - self.latest_es_monitor_batch > self.config.es_patience:
                    print(
                        f"Stopped early at batch {self.batch}, since latest monitor update of "
                        + f"{self.config.es_patience_monitor} was at batch {self.latest_es_monitor_batch}"
                        + f" (patience is {self.config.es_patience})."
                    )
                    return
                else:
                    print(
                        f"Did not stop early at batch {self.batch}, since latest update was at "
                        + f"batch {self.latest_es_monitor_batch} (patience is {self.config.es_patience})."
                    )
                t_prev = time.time()

                if self.batch % self.config.tb_log_interval_val == 0:
                    self.report_validation()
                    self.model.train()

                self.batch += 1

                if self.batch >= 10:
                    verbose = 0

                torch.cuda.empty_cache()


def train(config, random_seed=0, only_return_t=False, experiment=""):
    if "/" in experiment:
        random_seed = int(experiment.split("/")[-1])
    print(f"Using random seed {random_seed}")

    log_stdout_to_file(config)

    t = Train(config, random_seed=random_seed)

    if only_return_t:
        return t

    t.train()


def train_experiment(experiment):
    init_wandb_experiment(experiment)

    print("Running training for experiment '{experiment}'")
    train(wandb.config, experiment=experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="new-maxpooling/0"
        # This experiment described the experiment name
        # and the path to the yaml file relative to the
        # experiments folder. Dashes (-) are folder separators
        # and the number after the / indicates the model
        # number in the ensemble.
    )
    args = parser.parse_args()

    train_experiment(args.experiment)
