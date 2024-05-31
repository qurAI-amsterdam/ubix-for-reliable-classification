import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import scipy
from tqdm.auto import tqdm
import torch
import pickle
from typing import List, Optional, Dict, Any, Union, Tuple
import wandb


from ubix.metrics import quadratic_weighted_kappa
from ubix.predict import Predict
from ubix.utils import get_device


class UBIX:
    def __init__(
        self,
        uncertainty_measure: str = "ordinal_entropy",
        ubix_type: str = "soft",
        deep_ensemble_n: int = 5,
    ):
        """
        Initialize the UBIX class. The class implements the UBIX method described in the paper
        "Uncertainty-Aware Multiple-Instance Learning for Reliable Classification: Application to Optical Coherence Tomography"
        from de Vente et al.

        Args:
            uncertainty_measure (str): The type of uncertainty measure to use. Default is 'ordinal_entropy'.
            ubix_type (str): The type of UBIX ('hard' or 'soft'). Default is 'soft'.
            deep_ensemble_n (int): Number of models in the ensemble. Only used if an ensemble is used and model weights actually
                                   need to be loaded. The latter is not the case for Max and Mean pooling.
        """
        self.uncertainty_measure = uncertainty_measure
        self.ubix_type = ubix_type
        self.ubix_params = None
        self.deep_ensemble_n = deep_ensemble_n
        self.loaded_predictors = None

    def _calculate_uncertainties(
        self,
        softmaxed: List[List[np.ndarray]],
        logits: List[List[np.ndarray]],
        rng_seed: int = 42,
    ) -> List[List[float]]:
        """
        Calculates uncertainties based on the specified uncertainty measure.

        Args:
            softmaxed (List[List[np.ndarray]]): A nested list of softmaxed arrays. The "shape"
            (not well defined because of lists with varying sizes) should be (N, I, M, C).
            logits (List[List[np.ndarray]]): A nested list of logits arrays. The "shape"
            (not well defined because of lists with varying sizes) should be (N, I, M, C).
            rng_seed (int): Random seed for reproducibility when using random uncertainty measure.

        Returns:
            List[List[float]]: Calculated uncertainties.
        """
        if self.uncertainty_measure == "entropy":
            uq_metric = _operate_instance_class(
                softmaxed, lambda n: scipy.stats.entropy(n.mean(0).T)
            )
        elif self.uncertainty_measure == "single_entropy":
            uq_metric = _operate_instance_class(
                softmaxed, lambda n: scipy.stats.entropy(n[0].T)
            )
        elif self.uncertainty_measure == "ordinal_entropy":
            uq_metric = _operate_instance_class(softmaxed, _ordinal_entropy)
        elif self.uncertainty_measure == "max_logit_maxmax_confidence":
            # take max logits within models individually, then the max over all of those values
            uq_metric = _operate_instance_class(logits, lambda n: n.max(-1).max())
        elif self.uncertainty_measure == "max_logit_maxmean_confidence":
            # take max logits within models individually, then the mean over all of those values
            uq_metric = _operate_instance_class(logits, lambda n: n.max(-1).mean())
        elif self.uncertainty_measure == "max_logit_max0th_confidence":
            # take max logits within first model, then the mean over all of those values
            # N.B.: this measure is used in the UBIX paper when we are using MaxLogit
            uq_metric = _operate_instance_class(logits, lambda n: n[0].max(-1))
        elif (
            self.uncertainty_measure == "confidence"
        ):  # Referred to as "maximum class probability" in the paper.
            uq_metric = _operate_instance_class(
                softmaxed, lambda n: n.mean(0).T.max(-1)
            )
        elif self.uncertainty_measure == "single_confidence":
            uq_metric = _operate_instance_class(softmaxed, lambda n: n[0].mean(0))
        elif self.uncertainty_measure == "variance":
            uq_metric = _operate_instance_class(softmaxed, lambda n: n.var(0).mean())
        elif self.uncertainty_measure == "ordinal_variance":
            uq_metric = _operate_instance_class(
                softmaxed, lambda n: (n * np.arange(n.shape[-1])).sum(-1).var(0)
            )
        elif self.uncertainty_measure == "random":
            rng = np.random.RandomState(rng_seed)
            uq_metric = _operate_instance_class(softmaxed, lambda n: rng.uniform(0, 1))
        else:
            raise ValueError(
                f"Unknown `self.uncertainty_measure` {self.uncertainty_measure}"
            )

        return uq_metric

    def _measure_is_uncertainty(self) -> bool:
        """
        Checks if the current uncertainty measure is actually an uncertainty measure or a confidence measure.

        Returns:
            bool: True if the measure is a confidence measure, False otherwise.
        """

        return not (
            self.uncertainty_measure.endswith("confidence")
            or self.uncertainty_measure == "random"
        )

    def _compute_post_hard_ubix_logits(
        self,
        logits: List[List[np.ndarray]],
        uncertainties: List[List[float]],
        ubix_params: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Computes the post-hard UBIX logits.

        Args:
            logits (List[List[np.ndarray]]): A nested list of logits arrays.
            uncertainties (List[List[float]]): A nested list of uncertainties.
            ubix_params (Dict[str, Any]): UBIX parameters.

        Returns:
            Tuple[List[np.ndarray], int]: Post-UBIX logits and the number of instances that fell outside criteria.
        """

        if not self._measure_is_uncertainty():
            # Confidence is inversely correlated with uncertainty, so we want to exclude the lowest confidence
            # scores, not the highest
            comp_fn = lambda a, b: a >= b
        else:
            comp_fn = lambda a, b: a <= b

        # Looping over the instances and excluding the ones with too high uncertainty / too low confidence
        post_ubix_logits = []
        all_instances_kept = []
        n_fell_outside_criteria = 0
        for log_withnans, unc_withnans in zip(logits, uncertainties):
            log = _strip_nan(log_withnans)
            unc = _strip_nan(unc_withnans)
            instances_to_keep = comp_fn(unc, ubix_params["unc_thresh_tau"])

            if not np.any(instances_to_keep):
                # We will take the most certain instance to still be able to predict something,
                # but send a warning later that actually all instances were too uncertain.
                if self._measure_is_uncertainty():
                    instances_to_keep = [np.argmin(unc)]
                else:
                    instances_to_keep = [np.argmax(unc)]
                n_fell_outside_criteria += 1

            post_ubix_logits_instance = log[instances_to_keep]
            post_ubix_logits.append(post_ubix_logits_instance)
            all_instances_kept.append(instances_to_keep)

        if n_fell_outside_criteria > 0:
            print(
                f"WARNING! {n_fell_outside_criteria} cases had all instances excluded "
                + f'(for unc_thresh_tau {ubix_params["unc_thresh_tau"]}).'
            )

        return post_ubix_logits, n_fell_outside_criteria

    def _compute_post_soft_ubix_logits(
        self,
        logits: np.ndarray,
        uncertainties: List[List[float]],
        ubix_params: Dict[str, Any],
    ) -> np.ndarray:
        """
        Computes the post-soft UBIX logits.

        Args:
            logits (np.ndarray): An array of logits.
            uncertainties (List[List[float]]): A nested list of uncertainties.
            ubix_params (Dict[str, Any]): UBIX parameters.

        Returns:
            np.ndarray: Post-UBIX logits.
        """
        min_logits_per_class_rep = np.tile(
            ubix_params["min_logits_per_class"], logits.shape[:3] + (1,)
        )
        uncertainties_rep = np.stack(
            [np.stack([uncertainties] * logits.shape[2], 2)] * logits.shape[3], 3
        )
        if self.uncertainty_measure.endswith(
            "variance"
        ) or self.uncertainty_measure.endswith("entropy"):
            delta = ubix_params["delta"] * -1
        else:
            delta = ubix_params["delta"]
        gamma_hat = (
            ubix_params["gamma"] * (ubix_params["umax"] - ubix_params["umin"])
            + ubix_params["umin"]
        )

        device = get_device()

        # If cuda is available, we perform soft UBIX with cuda for speed.
        n_batches = 200  # Number of batches in which the full dataset is split
        n_samples = len(logits)
        n_samples_per_chunk = int(np.ceil(n_samples / n_batches))
        batches = [
            (cnk * n_samples_per_chunk, (cnk + 1) * n_samples_per_chunk)
            for cnk in range(n_batches)
        ]

        out = []

        for start, end in batches:
            logits_c = torch.from_numpy(logits[start:end]).to(torch.float32).to(device)
            min_logits_per_class_rep_c = torch.from_numpy(
                min_logits_per_class_rep[start:end]
            ).to(torch.float32).to(device)
            uq_metric_rep_c = torch.from_numpy(uncertainties_rep[start:end]).to(torch.float32).to(device)

            out_c = (logits_c - min_logits_per_class_rep_c) / (
                1 + torch.exp(-delta * (uq_metric_rep_c - gamma_hat))
            ) + min_logits_per_class_rep_c

            out_c = out_c.cpu()
            out.append(out_c)

        out = torch.cat(out).numpy()

        return out

    def _milpool(self, logits, predictor):
        """
        Applies the MIL pooling function based on the type of predictor.

        Args:
            logits (np.ndarray): An array of logits.
            predictor (Predict): A predictor object.

        Returns:
            np.ndarray: Bag level-logits after applying MIL pooling.
        """
        loading_models_needed = predictor.config.mil_pooling_function in [
            "attention",
            "attention_with_distribution",
            "distribution",
            "transmil",
        ]

        if loading_models_needed:
            model_name = predictor.model_name

            if (self.loaded_predictors is None) or (
                not self.loaded_predictors[0].model_name.startswith(model_name)
            ):
                self.loaded_predictors = [
                    Predict(
                        model_name=f"{model_name}/{m}",
                        model_type=predictor.model_type,
                        change_config=predictor.change_config,
                    )
                    for m in range(self.deep_ensemble_n)
                ]
                
                # Ensuring the model_name of this run was not influenced by loading
                # the models
                wandb.run.name = model_name
                wandb.config.model_name = model_name
            else:
                # Using cached self.loaded_predictors
                pass

            def milpoolfn(a):
                out = []
                for nth_model, loaded_predictor in enumerate(self.loaded_predictors):
                    with torch.no_grad():
                        y_instances = a[:, nth_model].unsqueeze(0)
                        out_i = loaded_predictor.model.forward(
                            None, y_instances=y_instances
                        )
                    out.append(out_i)

                out = torch.cat(out, dim=0)
                return out.cpu()

            device = self.loaded_predictors[0].device
            logits = [
                torch.from_numpy(np.array(a)).to(torch.float32).to(device)
                for a in logits
            ]
        elif predictor.config.mil_pooling_function == "max":
            milpoolfn = lambda a: np.nanmax(a, 0)
        elif predictor.config.mil_pooling_function == "mean":
            milpoolfn = lambda a: np.nanmean(a, 0)
        else:
            raise ValueError(
                f"Unknown MIL pooling function {predictor.config.mil_pooling_function}"
            )

        bag_logits = [milpoolfn(a) for a in logits]  # axis 0 of a is instance axis
        bag_softmax = np.array([scipy.special.softmax(a, -1) for a in bag_logits])
        bag_softmax_mean = bag_softmax.mean(1)  # mean of models in ensemble

        return bag_softmax_mean

    def predict_non_ubix(
        self,
        predictor: Predict,
        predict_set: str,
        rng_seed: int = 42,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], np.ndarray, List[List[float]]]:
        """
        Performs prediction without UBIX.

        Args:
            predictor (Predict): The predictor object used for making predictions.
            predict_set (str): The split set to predict on (e.g., 'val' or 'test').
            rng_seed (int): Random seed for reproducibility. Default is 42. Only used when using random uncertainty.
            **kwargs: Additional keyword arguments for the predictor.

        Returns:
            Tuple[Dict[str, Any], np.ndarray, List[List[float]]]:
            - predictions: The predictions made by the predictor.
            - logits: Instance-level logits, reshaped as N, I, M, C.
            - uncertainties: Calculated uncertainties.
        """
        predictions = predictor.predict_x_set(predict_set, **kwargs)

        # Instance-level logits, reshaped as N, I, M, C
        logits = _get_ins_nimc(predictions["y_pred_instances_all"])
        softmaxed = _operate_instance_class(
            logits, lambda n: scipy.special.softmax(n, -1)
        )

        # Calculate uncertainties
        uncertainties = self._calculate_uncertainties(
            softmaxed, logits, rng_seed=rng_seed
        )
        uncertainties = _pad_instances_with_nans(uncertainties)

        # We pad `logits` with nan values to the `max_instances` (maximum number of instances in the
        # dataset), such that it has a consistent shape in every dimension.
        logits = _pad_instances_with_nans(logits)

        return predictions, logits, uncertainties

    def _get_unc_thresh_from_tau(self, uncertainties, tau):
        if self._measure_is_uncertainty():
            real_tau = tau
        else:
            real_tau = 100 - tau

        return np.percentile(uncertainties[~np.isnan(uncertainties)], real_tau)

    def _optimize_hard_ubix_parameters(
        self,
        predictions: Dict[str, Any],
        logits: List[np.ndarray],
        uncertainties: List[List[float]],
        predictor: Predict,
        ubix_params: Dict[str, Any],
        eval_metric: callable,
        tau_grid_search: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Optimizes parameters for hard UBIX using a grid search over tau values.

        Args:
            predictions (Dict[str, Any]): Predictions made by the predictor.
            logits (List[np.ndarray]): List of instance-level logits.
            uncertainties (List[List[float]]): List of uncertainties.
            predictor (Predict): The predictor object used for making predictions.
            ubix_params (Dict[str, Any]): Initial UBIX parameters.
            eval_metric (callable): Evaluation metric to optimize.
            tau_grid_search (np.ndarray): Grid search values for tau.

        Returns:
            Dict[str, Any]: Optimized UBIX parameters.
        """
        perfs = []
        for tau in tqdm(tau_grid_search, desc="Hard UBIX grid search (for tau)"):
            ubix_params["tau"] = tau
            ubix_params["unc_thresh_tau"] = self._get_unc_thresh_from_tau(
                uncertainties, tau
            )

            (
                post_ubix_logits,
                n_fell_outside_criteria,
            ) = self._compute_post_hard_ubix_logits(logits, uncertainties, ubix_params)

            if n_fell_outside_criteria > 0:
                print(f"For tau {tau}, metrics could not be calculcated properly.")
                break

            y_pred_ubix = self._milpool(post_ubix_logits, predictor)
            perf = eval_metric(predictions["y_true"], y_pred_ubix)
            perfs.append(perf)

        optimal_tau_index = int(np.argmax(perfs))
        optimal_tau = tau_grid_search[optimal_tau_index]

        ubix_params["tau"] = optimal_tau
        ubix_params["unc_thresh_tau"] = self._get_unc_thresh_from_tau(
            uncertainties, optimal_tau
        )

        print(
            f"Optimal tau at {optimal_tau} (unc_thresh_tau = {ubix_params['unc_thresh_tau']}) "
            + f"(at performance {np.max(perfs)})"
        )

        return ubix_params

    def _optimize_soft_ubix_parameters(
        self,
        predictions: Dict[str, Any],
        logits: List[np.ndarray],
        uncertainties: List[List[float]],
        predictor: Predict,
        ubix_params: Dict[str, Any],
        eval_metric: callable,
        delta_grid_search: List[int],
        gamma_grid_search: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Optimizes parameters for soft UBIX using a grid search over delta and gamma values.

        Args:
            predictions (Dict[str, Any]): Predictions made by the predictor.
            logits (List[np.ndarray]): List of instance-level logits.
            uncertainties (List[List[float]]): List of uncertainties.
            predictor (Predict): The predictor object used for making predictions.
            ubix_params (Dict[str, Any]): Initial UBIX parameters.
            eval_metric (callable): Evaluation metric to optimize.
            delta_grid_search (List[int]): Grid search values for delta.
            gamma_grid_search (np.ndarray): Grid search values for gamma.

        Returns:
            Dict[str, Any]: Optimized UBIX parameters.
        """
        perfs = np.zeros((len(delta_grid_search), len(gamma_grid_search)))
        for di, delta in tqdm(
            enumerate(delta_grid_search),
            total=len(delta_grid_search),
            desc="Soft UBIX grid search (for delta and gamma)",
        ):
            for gi, gamma in enumerate(gamma_grid_search):
                ubix_params["delta"] = delta
                ubix_params["gamma"] = gamma
                post_ubix_logits = self._compute_post_soft_ubix_logits(
                    logits, uncertainties, ubix_params
                )

                y_pred_ubix = self._milpool(post_ubix_logits, predictor)
                perf = eval_metric(predictions["y_true"], y_pred_ubix)

                perfs[di, gi] = perf

        optimal_delta_index, optimal_gamma_index = np.unravel_index(
            np.argmax(perfs, axis=None), perfs.shape
        )
        optimal_delta = delta_grid_search[optimal_delta_index]
        optimal_gamma = gamma_grid_search[optimal_gamma_index]

        print(
            f"Optimal delta at {optimal_delta} and gamma at {optimal_gamma} "
            + f"(at performance {np.max(perfs)})"
        )

        ubix_params["delta"] = optimal_delta
        ubix_params["gamma"] = optimal_gamma

        return ubix_params

    def optimize_ubix_parameters(
        self,
        predictor: Predict,
        predict_set: str = "val",
        tau_grid_search: np.ndarray = np.arange(100, 79.9, -0.1),
        delta_grid_search: List[int] = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        gamma_grid_search: np.ndarray = np.arange(-0.5, 1.5, 0.05),
        eval_metric: callable = quadratic_weighted_kappa,
        rng_seed: int = 42,
        **predict_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optimizes UBIX parameters using a grid search.

        Args:
            predictor (Predict): The predictor object used for making predictions.
            predict_set (str): The dataset to predict on. Default is 'val'.
            tau_grid_search (np.ndarray): Grid search values for tau in hard UBIX.
            delta_grid_search (List[int]): Grid search values for delta in soft UBIX.
            gamma_grid_search (np.ndarray): Grid search values for gamma in soft UBIX.
            eval_metric (callable): Evaluation metric to optimize.
            rng_seed (int): Random seed for reproducibility. Default is 42. Only used when uncertainty is "random",
                            in all other cases, predicting and optimizing UBIX should be deterministic.
            **predict_kwargs: Additional keyword arguments for the predictor.

        Returns:
            Dict[str, Any]: Optimized UBIX parameters.
        """
        predictions, logits, uncertainties = self.predict_non_ubix(
            predictor, predict_set, rng_seed, **predict_kwargs
        )

        # Compute statistics related to the dataset
        min_logits_per_class = list(np.nanmin(logits, (0, 1, 2)))
        mean_logits_per_model_and_class = list(np.nanmean(logits, (0, 1)))
        std_logits_per_model_and_class = list(np.nanstd(logits, (0, 1)))

        # Calculating minimum and maximum uncertainty in the dataset
        umin = np.nanmin(uncertainties)
        umax = np.nanmax(uncertainties)

        # Overal statistics of the dataset
        ubix_params = {
            "min_logits_per_class": min_logits_per_class,  # not used for hard UBIX
            "mean_logits_per_model_and_class": mean_logits_per_model_and_class,  # not used at all currently
            "std_logits_per_model_and_class": std_logits_per_model_and_class,  # not used at all currently
            "umin": umin,
            "umax": umax,
        }

        # Performing a grid search for the optimal performance (in terms of eval_metric)
        if self.ubix_type == "hard":
            ubix_params = self._optimize_hard_ubix_parameters(
                predictions,
                logits,
                uncertainties,
                predictor,
                ubix_params,
                eval_metric,
                tau_grid_search,
            )
        elif self.ubix_type == "soft":
            ubix_params = self._optimize_soft_ubix_parameters(
                predictions,
                logits,
                uncertainties,
                predictor,
                ubix_params,
                eval_metric,
                delta_grid_search,
                gamma_grid_search,
            )
        else:
            raise ValueError(f"Unknown self.ubix_type {self.ubix_type}")

        self.ubix_params = ubix_params

        return ubix_params

    def save_ubix_parameters(self, save_path: str) -> None:
        """
        Saves UBIX parameters to a file.

        Args:
            save_path (str): The path to save the UBIX parameters.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.ubix_params, f)

    def load_ubix_parameters(self, save_path: str) -> Dict[str, Any]:
        """
        Loads UBIX parameters from a file.

        Args:
            save_path (str): The path to load the UBIX parameters from.

        Returns:
            Dict[str, Any]: Loaded UBIX parameters.
        """
        with open(save_path, "rb") as f:
            self.ubix_params = pickle.load(f)

        return self.ubix_params

    def predict(
        self,
        predictor: Predict,
        predict_set: str = "test",
        **predict_kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Performs prediction with UBIX.

        Args:
            predictor (Predict): The predictor object used for making predictions.
            predict_set (str): The dataset to predict on. Default is 'test'.
            **predict_kwargs: Additional keyword arguments for the predictor.

        Returns:
            np.ndarray: Predictions with UBIX.
        """
        if self.ubix_params is None:
            raise ValueError(
                "Cannot predict before UBIX parameters are set. "
                + "This can be done by calling `UBIX.optimize_ubix_parameters` or "
                + "`UBIX.load_ubix_parameters`."
            )

        if type(self.ubix_params) is not dict:
            raise TypeError("Expected self.ubix_params to be of type dict.")

        _, logits, uncertainties = self.predict_non_ubix(
            predictor, predict_set, **predict_kwargs
        )

        if self.ubix_type == "hard":
            if "tau" not in self.ubix_params:
                raise KeyError(
                    'Expected "tau" to be available in self.ubix_params for hard UBIX.'
                )

            (
                post_ubix_logits,
                n_fell_outside_criteria,
            ) = self._compute_post_hard_ubix_logits(
                logits, uncertainties, self.ubix_params
            )

            if n_fell_outside_criteria > 0:
                print(
                    f"Warning! For {n_fell_outside_criteria} samples, predictions could not be "
                    + "properly calculated, as all instances were excluded by UBIX (uncertainties were "
                    + "apparently too high for all instances in those samples)."
                )

            y_pred_ubix = self._milpool(post_ubix_logits, predictor)
        elif self.ubix_type == "soft":
            if "gamma" not in self.ubix_params or "delta" not in self.ubix_params:
                raise KeyError(
                    'Expected both "gamma" and "delta" to be available in self.ubix_params for hard UBIX.'
                )

            post_ubix_logits = self._compute_post_soft_ubix_logits(
                logits, uncertainties, self.ubix_params
            )
            y_pred_ubix = self._milpool(post_ubix_logits, predictor)
        else:
            raise ValueError(f"Unknown self.ubix_type {self.ubix_type}")

        return y_pred_ubix


def _ordinal_entropy(softmax: np.ndarray) -> float:
    """
    Computes ordinal entropy for the given softmax probabilities.

    Args:
        softmax (np.ndarray): Softmax probabilities.

    Returns:
        float: Computed ordinal entropy.
    """
    mn = softmax.mean(0)
    sm = 0
    for i in range(1, len(mn)):
        a = sum(mn[:i])
        b = sum(mn[i:])
        sm += scipy.stats.entropy([a, b])
    return sm


def _pad_instances_with_nans(values):
    """
    Pads `values` with nan values to in the instance axis to
    the maximum number of instances in the dataset,
    such that it has a consistent shape in every dimension.

    Can take a list of arrays with varying shapes and input and outputs
    an array with consistent dimensions.
    """

    # Calculate maximum number of instances in the dataset
    max_instances = max([len(a) for a in values])

    # Perform padding
    return np.array(
        [
            np.concatenate(
                [a, np.ones((max_instances - len(a),) + a[0].shape) * np.nan]
            )
            for a in values
        ]
    )


def _strip_nan(arr: List[np.ndarray]) -> np.ndarray:
    """
    Removes NaN values from a list of numpy arrays.

    Args:
        arr (List[np.ndarray]): List of numpy arrays.

    Returns:
        np.ndarray: Array with NaN values removed.
    """
    return np.array([x for x in arr if not np.isnan(x).any()])


def _get_ins_nimc(y_pred_instances: np.ndarray) -> List[np.ndarray]:
    """
    Reshapes an array from [M, N, 1, I, C] to [N, I, M, C].

    Args:
        y_pred_instances (np.ndarray): The input array of shape [M, N, 1, I, C].

    Returns:
        List[np.ndarray]: The reshaped array of shape [N, I, M, C].
    """
    ins = [[n.squeeze(0) for n in m] for m in y_pred_instances]
    ins_nimc = []
    for n in range(len(ins[0])):
        ins_nimc.append([])
        for i in range(len(ins[0][n])):
            ins_nimc[n].append(np.array([ins[m][n][i] for m in range(len(ins))]))
        ins_nimc[n] = np.array(ins_nimc[n])
    return ins_nimc


def _operate_instance_class(a: List[List[np.ndarray]], fn: callable) -> List[List[Any]]:
    """
    Applies a function to each element in a nested list.

    Args:
        a (List[List[np.ndarray]]): A nested list of arrays.
        fn (callable): A function to apply to each element.

    Returns:
        List[List[Any]]: A nested list with the function applied to each element.
    """
    return [[fn(n) for n in m] for m in a]


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"

    ubix_type = "soft"
    # ubix_type = "hard"
    # uncertainty_measure = "max_logit_max0th_confidence"
    uncertainty_measure = "ordinal_entropy"

    model_name = "pretrained-maxpooling/0"
    predictions_postfix = "_aug_10_mean"  # TTA

    # model_name = "pretrained-maxpooling"

    # uncertainty_measure = "max_logit_max0th_confidence"
    # model_name = "pretrained-meanpooling"
    # model_name = "pretrained-attentionpooling"
    # model_name = "pretrained-distributionpooling"
    # model_name = "pretrained-distributionwithattentionpooling"
    # model_name = "pretrained-transmilpooling"
    # predictions_postfix = "_deep_ensemble_5_mean"
    
    save_path = f"./results/models/{model_name}/ubix_params_{ubix_type}{predictions_postfix}_{uncertainty_measure}.p"

    ubix = UBIX(ubix_type=ubix_type, uncertainty_measure=uncertainty_measure)
    optimization_predictor = Predict(
        model_name=model_name,  # as the model_name does not end
        # with /{model_nr}, it will load the predictions that should have been made
        # for the deep ensemble.
        model_type="best_quadratic_weighted_kappa",
        do_not_load_model=True,
    )
    ubix.optimize_ubix_parameters(
        optimization_predictor,
        predict_set="val",
        predictions_postfix=predictions_postfix,
    )

    ubix.save_ubix_parameters(save_path)
    ubix.load_ubix_parameters(save_path)

    test_predictor = Predict(
        model_name=model_name,
        model_type="best_quadratic_weighted_kappa",
        do_not_load_model=True,
    )
    y_pred_ubix = ubix.predict(
        test_predictor, predict_set="test", predictions_postfix=predictions_postfix
    )
    predictions_non_ubix, _, _ = ubix.predict_non_ubix(
        test_predictor, predict_set="test", predictions_postfix=predictions_postfix
    )

    qwk_ubix = quadratic_weighted_kappa(predictions_non_ubix["y_true"], y_pred_ubix)
    qwk_non_ubix = quadratic_weighted_kappa(
        predictions_non_ubix["y_true"], predictions_non_ubix["y_pred"]
    )
    print("qwk_ubix:", qwk_ubix)
    print("qwk_non_ubix:", qwk_non_ubix)
