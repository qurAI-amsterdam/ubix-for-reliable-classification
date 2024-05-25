import numpy as np
import pickle
import scipy
from tqdm.auto import tqdm
import wandb


def inference_results(
    p, n_its, model_names, load_from_file, child_name, postfix_empty, predict_set
):
    all_out = {
        "y_true": [],
        "y_pred": [],
        "y_pred_instances": [],
    }

    for it in tqdm(range(n_its), desc="Stochastic inference iteration"):
        if model_names is not None:
            p.config.update({"model_name": model_names[it]}, allow_val_change=True)
            wandb.run.name = model_names[it]

            if not load_from_file:
                p.load_model()

        predictions_postfix = "" if postfix_empty else f"_{child_name}{it}"
        out_i = p.predict_x_set(
            predict_set,
            load_from_file=load_from_file,
            predictions_postfix=predictions_postfix,
        )

        if (
            out_i["y_pred"].shape[1] == 1
        ):  # Older versions of predictions.p have an empty dimension 1
            out_i["y_pred"] = np.squeeze(out_i["y_pred"], 1)
            out_i["y_true"] = np.squeeze(out_i["y_true"], 1)

        all_out["y_pred"].append(out_i["y_pred"])
        if "y_pred_instances" in out_i:
            all_out["y_pred_instances"].append(out_i["y_pred_instances"])

        if it == 0:
            all_out["y_true"] = out_i["y_true"]
        else:
            assert np.all(
                all_out["y_true"].squeeze() == out_i["y_true"].squeeze()
            ), f"{all_out['y_true']} \n =============== != =============== \n {out_i['y_true']}"

    all_out["y_true"] = np.array(all_out["y_true"])
    all_out["y_pred"] = np.array(all_out["y_pred"])

    return all_out


def stochastic_inference(
    p,
    n_its,
    load_from_file,
    child_name,
    model_names=None,
    postfix_empty=False,
    model_base_name=None,
    predict_set="val",
):
    all_out = inference_results(
        p, n_its, model_names, load_from_file, child_name, postfix_empty, predict_set
    )

    reduce_fns = {"median": np.median, "mean": np.mean}

    for reduce_name, reduce_fn in reduce_fns.items():
        # Take the softmax
        y_pred_softmax = scipy.special.softmax(all_out["y_pred"], axis=-1)

        # Calculate the reduced (e.g. mean) version of the softmax vector
        reduced = reduce_fn(y_pred_softmax, axis=0)

        # Renormalize
        reduced = reduced / np.stack([np.sum(reduced, axis=-1)] * reduced.shape[-1], -1)

        # Calculate logit from softmax
        eps = 1e-7
        logit_reduced = np.log(reduced + eps)

        softmax_logit_reduced = scipy.special.softmax(logit_reduced, axis=-1)
        assert np.all(np.abs(softmax_logit_reduced - reduced) < 0.00001)

        # Uncertainty in terms of standard deviation of the classes
        sigma_mean_classes = np.std(y_pred_softmax, axis=0)
        sigma_mean_classes = np.mean(sigma_mean_classes, axis=-1)

        # Uncertainty for binary classification
        smallest_positive_class = 2
        p_positive = np.sum(y_pred_softmax[..., smallest_positive_class:], axis=-1)
        sigma_onset = np.std(p_positive, axis=0)

        all_entropy = [scipy.stats.entropy(yp.T) for yp in y_pred_softmax]
        mean_entropy = np.mean(all_entropy, axis=0)

        out = {
            "y_true": all_out["y_true"],
            "y_pred": logit_reduced,
            "y_pred_all": all_out["y_pred"],
            "y_pred_instances_all": all_out["y_pred_instances"],
            f"sigma_onset_{smallest_positive_class}": sigma_onset,
            "sigma_mean_classes": sigma_mean_classes,
            "all_entropy": all_entropy,
            "mean_entropy": mean_entropy,
        }

        if model_base_name is not None:
            p.config.update({"model_name": model_base_name}, allow_val_change=True)

        predictions_postfix = p.postfix(
            {"predictions_postfix": f"_{child_name}_{n_its}_{reduce_name}"}, predict_set
        )
        predictions_path = p.current_pred_path(predictions_postfix)

        with open(predictions_path, "wb") as f:
            pickle.dump(out, f)

    return p
