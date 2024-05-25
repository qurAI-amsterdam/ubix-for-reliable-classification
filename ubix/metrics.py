import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from torch import nn
import torch


def softmax(x, axis=-1):
    return np.exp(x) / np.stack([np.sum(np.exp(x), axis=axis)] * x.shape[-1], axis=axis)


def accuracy(y_true, y_pred):
    labels_int = y_true
    outputs_int = np.argmax(y_pred, axis=1)

    correct = np.sum(labels_int.squeeze() == outputs_int.squeeze())
    total = len(labels_int)

    acc = correct / total

    return acc


def brier_score(y_true, y_pred, confidence=None):
    y_pred_softmax = softmax(y_pred, axis=-1)

    one_hot = np.zeros(y_pred_softmax.shape)
    one_hot[range(len(y_true)), y_true] = 1

    probs = y_pred_softmax
    targets = one_hot
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def auc_roc_offset3(y_true, y_pred, **kwargs):
    # Offset at class 3 (a.k.a. CORADS 4)
    return auc_roc(y_true, y_pred, offset=3, **kwargs)


def auc_roc_offset4(y_true, y_pred, **kwargs):
    # Offset at class 4 (a.k.a. CORADS 5)
    return auc_roc(y_true, y_pred, offset=4, **kwargs)


def auc_roc(
    y_true,
    y_pred,
    plot=False,
    classification_report=False,
    only_return_opt_thresh=False,
    only_return_classification_report=False,
    config=None,
    offset=None,
    use_max_prob=False,
    do_softmax=True,
):
    """
    Binary AUC
    """

    if len(np.unique(y_true)) == 1:
        return np.nan

    if y_pred.ndim == 1:
        y_true_post = y_true
        y_pred_post = y_pred
    else:
        # Setting smallest_positive_class, a.k.a. diseased onset

        if offset is None:
            if y_pred.shape[1] == 5 or y_pred.shape[1] == 6:
                offset = 2  # From intermediate AMD
            elif y_pred.shape[1] == 4:
                offset = (
                    1  # From intermediate AMD, assuming No AMD and Early AMD are merged
                )
            elif y_pred.shape[1] == 3:
                offset = 1  # From intermediate AMD, assuming No AMD and Early AMD are merged and
                # two Advanced AMD: GA and Advanced AMD: CNV are merged
            elif y_pred.shape[1] == 2:
                offset = 1  # From intermediate AMD, assuming No AMD and Early AMD are merged (and last
                # three classes are also merged)
            else:
                raise ValueError(
                    f"Incorrect value for y_pred.shape[1]: {y_pred.shape[1]}"
                )

        y_true_post = y_true >= offset

        if use_max_prob:
            y_pred_post = np.max(y_pred, axis=-1)

            # print("y_pred:", y_pred)
            # print("y_pred_post:", y_pred_post)
            # print("alt:", np.sum(softmax(y_pred, axis=-1)[:, offset:], axis=-1))
        else:
            y_pred_post = y_pred
            if do_softmax:
                y_pred_post = softmax(y_pred_post, axis=-1)
            y_pred_post = np.sum(y_pred_post[:, offset:], axis=-1)

    if len(np.unique(y_true_post)) == 1:
        return np.nan

    # print(np.unique(y_true_post))
    # print(np.unique(y_pred_post))

    roc_auc = sklearn.metrics.roc_auc_score(y_true_post, y_pred_post)
    # print("roc_auc:", roc_auc)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true_post, y_pred_post)

    opt_thresh = thresholds[np.argmax(tpr - fpr)]

    if only_return_opt_thresh:
        return opt_thresh

    if classification_report or only_return_classification_report:
        report = sklearn.metrics.classification_report(
            y_true_post, y_pred_post >= opt_thresh, output_dict=True
        )

        if only_return_classification_report:
            return report
        else:
            print("Classification report:")
            print(report)

    return roc_auc


def fdauc(y_true, y_pred, confidence=None):
    y_pred_amax = y_pred.argmax(-1)

    if y_true.ndim == 2 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)

    correct = y_true == y_pred_amax

    if confidence is None:
        sm = scipy.special.softmax(y_pred, -1)
        confidence = sm.max(-1)

    if confidence.ndim == 2 and confidence.shape[-1] == 1:
        confidence = confidence.squeeze(-1)

    if np.unique(correct).size == 1:
        return 1.0

    return sklearn.metrics.roc_auc_score(correct, confidence)


def kappa(y_true, y_pred):
    y_true_post = y_true
    y_pred_post = np.argmax(y_pred, axis=-1)

    return sklearn.metrics.cohen_kappa_score(y_true_post, y_pred_post)


def quadratic_weighted_kappa(y_true, y_pred):
    y_true_post = y_true
    y_pred_post = np.argmax(y_pred, axis=-1)

    return sklearn.metrics.cohen_kappa_score(
        y_true_post, y_pred_post, weights="quadratic"
    )


def linearly_weighted_kappa(y_true, y_pred):
    y_true_post = y_true
    y_pred_post = np.argmax(y_pred, axis=-1)

    return sklearn.metrics.cohen_kappa_score(y_true_post, y_pred_post, weights="linear")


def sensitivity(y_true, y_pred, **kwargs):
    report = auc_roc(y_true, y_pred, only_return_classification_report=True, **kwargs)

    if not type(report) is dict and np.isnan(report):
        return np.nan

    # print('report:')
    # print(report)
    return report["True"]["recall"]


def specificity(y_true, y_pred, **kwargs):
    report = auc_roc(y_true, y_pred, only_return_classification_report=True, **kwargs)

    if not type(report) is dict and np.isnan(report):
        return np.nan

    # print('report:')
    # print(report)
    return report["False"]["recall"]


def aece(y_true, y_pred):
    """
    Adaptive expected calibration error.
    """

    y_pred_softmax = softmax(y_pred, axis=-1)
    y_pred_amax = np.argmax(y_pred_softmax, axis=-1)
    # print('y_pred_amax:', y_pred_amax)
    # print('y_pred_amax.shape:', y_pred_amax.shape)

    probs = y_pred_softmax[range(len(y_pred_amax)), y_pred_amax]
    correctness = y_pred_amax == y_true

    infer_results = [[p, c] for p, c in zip(probs, correctness)]

    aece_out, _, _, _, _, _ = adaptive_binning(
        infer_results, show_reliability_diagram=False
    )

    return aece_out


def nll(y_true, y_pred):
    nll_criterion = nn.CrossEntropyLoss()
    return nll_criterion(
        torch.tensor(y_pred) if isinstance(y_pred, np.ndarray) else y_pred,
        torch.tensor(y_true) if isinstance(y_true, np.ndarray) else y_true,
    ).item()


def adaptive_binning(infer_results, show_reliability_diagram=True):
    """
    Thanks to https://github.com/yding5/AdaptiveBinning/blob/master/AdaptiveBinning.py

    This function implement adaptive binning. It returns AECE, AMCE and some other useful values.
    Arguements:
    infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample. res[0] is the confidence score r and res[1] is the correctness score c. Since c is either 1 or 0, here res[1] is True if the prediction is correctd and False otherwise.
    show_reliability_diagram (boolean): a boolean value to denote wheather to plot a Reliability Diagram.
    Return Values:
    AECE (float): expected calibration error based on adaptive binning.
    AMCE (float): maximum calibration error based on adaptive binning.
    confidence (list): average confidence in each bin.
    accuracy (list): average accuracy in each bin.
    cof_min (list): minimum of confidence in each bin.
    cof_max (list): maximum of confidence in each bin.
    """

    # Intialize.
    infer_results.sort(key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    assert (
        infer_results[0][0] <= 1 and infer_results[0][0] >= 0
    ), "Confidence score should be in [0,1]"

    z = 1.645
    # z = .5
    num = [0 for i in range(n_total_sample)]
    final_num = [0 for i in range(n_total_sample)]
    correct = [0 for i in range(n_total_sample)]
    confidence = [0 for i in range(n_total_sample)]
    cof_min = [1 for i in range(n_total_sample)]
    cof_max = [0 for i in range(n_total_sample)]
    acc = [0 for i in range(n_total_sample)]

    ind = 0
    target_number_samples = float("inf")

    # Traverse all samples for a initial binning.
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 40 and cof_min[ind] - infer_results[-1][0] > 0.05:
                ind += 1
                target_number_samples = float("inf")
        num[ind] += 1
        confidence[ind] += confidence_score

        assert correctness in [True, False], "Expect boolean value for correctness!"
        if correctness == True:
            correct[ind] += 1

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float("inf")
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    n_bins = ind + 1

    # Get final binning.
    if target_number_samples - num[ind] > 0:
        needed = target_number_samples - num[ind]
        extract = [0 for i in range(n_bins - 1)]
        final_num[n_bins - 1] = num[n_bins - 1]
        for i in range(n_bins - 1):
            extract[i] = int(needed * num[ind] / n_total_sample)
            final_num[i] = num[i] - extract[i]
            final_num[n_bins - 1] += extract[i]
    else:
        final_num = num
    final_num = final_num[:n_bins]

    # Re-intialize.
    num = [0 for i in range(n_bins)]
    correct = [0 for i in range(n_bins)]
    confidence = [0 for i in range(n_bins)]
    cof_min = [1 for i in range(n_bins)]
    cof_max = [0 for i in range(n_bins)]
    acc = [0 for i in range(n_bins)]
    gap = [0 for i in range(n_bins)]
    neg_gap = [0 for i in range(n_bins)]
    # Bar location and width.
    x_location = [0 for i in range(n_bins)]
    width = [0 for i in range(n_bins)]

    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        num[ind] += 1
        confidence[ind] += confidence_score

        if correctness == True:
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            acc[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - acc[ind] > 0:
                gap[ind] = confidence[ind] - acc[ind]
            else:
                neg_gap[ind] = confidence[ind] - acc[ind]
            ind += 1

    # Get AECE and AMCE based on the binning.
    AMCE = 0
    AECE = 0
    for i in range(n_bins):
        AECE += abs((acc[i] - confidence[i])) * final_num[i] / n_total_sample
        AMCE = max(AMCE, abs((acc[i] - confidence[i])))

    # Plot the Reliability Diagram if needed.
    if show_reliability_diagram:
        lw = 3
        f1, ax = plt.subplots()
        plt.bar(
            x_location, acc, width, linewidth=lw, color=(0, 0, 1), edgecolor=(0, 0, 0.6)
        )
        plt.bar(
            x_location,
            gap,
            width,
            bottom=acc,
            linewidth=lw,
            color=(0, 1, 0),
            edgecolor=(0, 0.6, 0),
            alpha=0.5,
        )
        plt.bar(
            x_location,
            neg_gap,
            width,
            bottom=acc,
            linewidth=lw,
            color=(1, 0, 0),
            edgecolor=(0.6, 0, 0),
            alpha=0.5,
        )
        plt.legend(["Accuracy", "Positive gap", "Negative gap"], fontsize=18, loc=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Confidence", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)

        plt.plot([0, 1], [0, 1], "--", linewidth=lw, color=(0.5, 0.5, 0.5))

        plt.show()

    return AECE, AMCE, cof_min, cof_max, confidence, acc


def metrics_test():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array(
        [
            [0.0, 0.1, 0.4, 0.25, 0.25],
            [0.0, 0.4, 0.1, 0.25, 0.25],
            [0.0, 0.1, 0.25, 0.4, 0.25],
            [0.0, 0.1, 0.4, 0.25, 0.25],
            [0.0, 0.1, 0.25, 0.25, 0.4],
        ]
    )

    acc = accuracy(y_true, y_pred)
    auc = auc_roc(y_true, y_pred, plot=True)
    k = kappa(y_true, y_pred)
    qk = quadratic_weighted_kappa(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)

    plot_confusion_matrix(y_true, np.argmax(y_pred, -1), np.unique(y_true).astype(int))
    plt.show()

    print("kappa:", k)
    print("accuracy:", acc)
    print("auc:", auc)
    print("qk:", qk)
    print("sens:", sens)
    print("spec:", spec)


"""
Functions for ROC curve with confidence interval for multiple models (e.g. trained with different random seeds) 
instead of bootstrapping).
"""


from typing import List, NamedTuple, Tuple

import numpy as np
from numpy import ndarray
from sklearn import metrics
import sklearn.metrics as skl_metrics


def plot_with_ci_different_models(
    y_true, y_prob, save_path, title="", num_bootstraps=1000, clr="b", model_desc=""
):
    """

    Args:
        y_true:
        y_prob:
        save_path:
        title:

    Returns:

    """
    all_fps, sens_mean, sens_lb, sens_up, _, _ = get_bootstrapped_roc_ci_curves(
        y_prob, y_true, num_bootstraps=num_bootstraps
    )

    # plt.figure(figsize=(6, 6))
    ax = plt.gca()

    auc = round(skl_metrics.auc(all_fps, sens_mean), 3)
    ci = (
        round(skl_metrics.auc(all_fps, sens_lb), 3),
        round(skl_metrics.auc(all_fps, sens_up), 3),
    )

    label = "AUC=" + str(auc) + " | 95% CI (" + str(ci[0]) + " - " + str(ci[1]) + ")"
    print(label)

    if model_desc != "":
        label = model_desc + " | " + label

    plt.plot(all_fps, sens_up, color=clr, ls=":")
    plt.plot(all_fps, sens_mean, color=clr, label=label)
    plt.plot(all_fps, sens_lb, color=clr, ls=":")

    ax.fill_between(all_fps, sens_lb, sens_up, facecolor=clr, alpha=0.05)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True, which="both")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.title(title)

    plt.savefig(save_path, bbox_inches=0, dpi=300)

    roc_stats = {"auc": auc, "ci": "95% CI (" + str(ci[0]) + " - " + str(ci[1]) + ")"}
    return roc_stats


class BootstrappedROCCICurves(NamedTuple):
    fpr_vals: ndarray
    mean_tpr_vals: ndarray
    low_tpr_vals: ndarray
    high_tpr_vals: ndarray
    low_az_val: ndarray
    high_az_val: ndarray


def get_bootstrapped_roc_ci_curves(
    y_pred: ndarray,
    y_true: ndarray,
    num_bootstraps: int = 100,
    ci_to_use: float = 0.95,
) -> BootstrappedROCCICurves:
    """
    Produces Confidence-Interval Curves to go alongside a regular ROC curve
    This is done by using boostrapping.
    Bootstrapping is done by selecting len(y_pred) samples randomly
    (with replacement) from y_pred and y_true.
    This is done num_boostraps times.

    Parameters
    ----------
    y_pred
        The predictions (scores) produced by the system being evaluated
    y_true
        The true labels (1 or 0) which are the reference standard being used
    num_bootstraps
        How many times to make a random sample with replacement
    ci_to_use
        Which confidence interval is required.

    Returns
    -------
    fpr_vals
        An equally spaced set of fpr vals between 0 and 1
    mean_tpr_vals
        The mean tpr vals (one per fpr_val) obtained by boostrapping
    low_tpr_vals
        The tpr vals (one per fpr_val) representing lower curve for CI
    high_tpr_vals
        The tpr vals (one per fpr_val) representing the upper curve for CI
    low_Az_val
        The lower Az (AUC) val for the given CI_to_use
    high_Az_val
        The higher Az (AUC) val for the given CI_to_use
    """

    rng_seed = 40  # control reproducibility
    bootstrapped_az_scores: List[float] = []

    tprs_list: List[ndarray] = []
    base_fpr = np.linspace(0, 1, 101)
    rng = np.random.RandomState(rng_seed)

    do_bootstrap = y_pred.ndim != 2

    idx = 0
    while (do_bootstrap and (len(bootstrapped_az_scores) < num_bootstraps)) or (
        not do_bootstrap and (len(bootstrapped_az_scores) < len(y_pred))
    ):
        # bootstrap by sampling with replacement on the prediction indices

        if do_bootstrap:
            indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        else:
            indices = idx
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        # get the fpr and tpr for this bootstrap
        fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_pred[indices])
        # get values at fixed fpr locations
        tpr_b = np.interp(base_fpr, fpr, tpr)
        tpr_b[0] = 0.0
        # append to list for all bootstraps
        tprs_list.append(tpr_b)

        # Get the Az score
        az_score = metrics.auc(fpr, tpr)
        bootstrapped_az_scores.append(az_score)

        idx += 1

    # Get the mean of the boostrapped tprs (at each fpr location)
    tprs_array = np.array(tprs_list)
    mean_tprs = tprs_array.mean(axis=0)

    # half the error margin allowed
    one_sided_ci = (1 - ci_to_use) / 2

    tprs_lower, tprs_upper = _get_confidence_intervals(
        n_bootstraps=len(base_fpr),
        one_sided_ci=one_sided_ci,
        points_array=tprs_array,
    )

    sorted_az_scores = np.array(bootstrapped_az_scores)
    sorted_az_scores.sort()

    az_ci_lower = sorted_az_scores[int(one_sided_ci * len(sorted_az_scores))]
    az_ci_upper = sorted_az_scores[int((1 - one_sided_ci) * len(sorted_az_scores))]

    return BootstrappedROCCICurves(
        fpr_vals=base_fpr,
        mean_tpr_vals=mean_tprs,
        low_tpr_vals=tprs_lower,
        high_tpr_vals=tprs_upper,
        low_az_val=az_ci_lower,
        high_az_val=az_ci_upper,
    )


class BootstrappedCIPointError(NamedTuple):
    mean_fprs: ndarray
    mean_tprs: ndarray
    low_tpr_vals: ndarray
    high_tpr_vals: ndarray
    low_fpr_vals: ndarray
    high_fpr_vals: ndarray


def get_bootstrapped_ci_point_error(
    y_score: ndarray,
    y_true: ndarray,
    num_bootstraps: int = 100,
    ci_to_use: float = 0.95,
    exclude_first_last: bool = True,
) -> BootstrappedCIPointError:
    """
    Produces Confidence-Interval errors for individual points from ROC
    Useful when only few ROC points exist so they will be plotted individually
    e.g. when range of score values in y_score is very small
    (e.g. manual observer scores)

    Note that this method only works by analysing the cloud of boostrapped
    points generatedfor a particular threshold value.  A fixed number of
    threshold values is essential. Therefore the scores in y_score must be
    from a fixed discrete set of values, eg. [1,2,3,4,5]

    Bootstrapping is done by selecting len(y_score) samples randomly
    (with replacement) from y_score and y_true.
    This is done num_boostraps times.

    Parameters
    ----------
    y_score
        The scores produced by the system being evaluated. A discrete set of
        possible scores must be used.
    y_true
        The true labels (1 or 0) which are the reference standard being used
    num_bootstraps: integer
        How many times to make a random sample with replacement
    ci_to_use
        Which confidence interval is required.
    exclude_first_last
        The first and last ROC point (0,0 and 1,1) are usually irrelevant
        in these scenarios where only a few ROC points will be
        individually plotted.
        Set this to true to ignore these first and last points.

    Returns
    -------
    mean_fprs
        The array of mean fpr values (1 per possible ROC point)
    mean_tprs
        The array of mean tpr values (1 per possible ROC point)
    low_tpr_vals
        The tpr vals (one per ROC point) representing lowest val in CI
    high_tpr_vals
        The tpr vals (one per ROC point) representing the highest val in CI
    low_fpr_vals
        The fpr vals (one per ROC point) representing lowest val in CI_to_use
    high_fpr_vals
        The fpr vals (one per ROC point) representing the highest val in CI
    """
    rng_seed = 40  # control reproducibility
    tprs_list: List[ndarray] = []
    fprs_list: List[ndarray] = []
    rng = np.random.RandomState(rng_seed)

    num_possible_scores = len(np.unique(y_score))

    while len(tprs_list) < num_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_score) - 1, len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        # get ROC data this boostrap
        fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_score[indices])
        if len(fpr) < num_possible_scores + 1:
            # if all scores are not represented in this selection then a
            # different number of ROC thresholds will be defined.
            # This causes problems.
            continue

        # remove first and last items - these are just end points of the ROC
        if exclude_first_last:
            fpr = fpr[1:-1]
            tpr = tpr[1:-1]

        # append these boostrap values to the list
        tprs_list.append(tpr)
        fprs_list.append(fpr)

    # Get the mean values for fpr and tpr at each ROC location
    tprs_array = np.array(tprs_list)
    fprs_array = np.array(fprs_list)

    mean_tprs = tprs_array.mean(axis=0)
    mean_fprs = fprs_array.mean(axis=0)

    # half the error margin allowed
    one_sided_ci = (1 - ci_to_use) / 2

    tprs_lower, tprs_upper = _get_confidence_intervals(
        n_bootstraps=tprs_array.shape[1],
        one_sided_ci=one_sided_ci,
        points_array=tprs_array,
    )
    fprs_lower, fprs_upper = _get_confidence_intervals(
        n_bootstraps=fprs_array.shape[1],
        one_sided_ci=one_sided_ci,
        points_array=fprs_array,
    )

    return BootstrappedCIPointError(
        mean_fprs=mean_fprs,
        mean_tprs=mean_tprs,
        low_tpr_vals=tprs_lower,
        high_tpr_vals=tprs_upper,
        low_fpr_vals=fprs_lower,
        high_fpr_vals=fprs_upper,
    )


def _get_confidence_intervals(
    *, n_bootstraps: int, one_sided_ci: float, points_array
) -> Tuple[ndarray, ndarray]:
    ci_upper = []
    ci_lower = []

    for bootstrap_point in range(n_bootstraps):
        points = points_array[:, bootstrap_point]
        points.sort()

        tpr_upper = points[int((1 - one_sided_ci) * len(points))]
        ci_upper.append(tpr_upper)
        tpr_lower = points[int(one_sided_ci * len(points))]
        ci_lower.append(tpr_lower)

    return np.asarray(ci_lower), np.asarray(ci_upper)


if __name__ == "__main__":
    metrics_test()

# %%
