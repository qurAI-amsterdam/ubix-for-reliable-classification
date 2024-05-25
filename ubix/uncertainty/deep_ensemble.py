import os

from ubix.predict import Predict
from ubix.utils import set_random_seeds
from ubix.uncertainty.stochastic_inference import stochastic_inference


def deep_ensemble(
    model_base_name,
    model_type,
    n_its=10,
    load_from_file=False,
    change_config=None,
    exclude_models=(),
    predict_set="val",
    num_workers=None,
):
    model_names = []
    model_nr = 0
    while len(model_names) < n_its:
        if model_nr not in exclude_models:
            model_names.append(f"{model_base_name}/{model_nr}")
        model_nr += 1

    # p = Predict(model_name=model_names[0], model_type=model_type, do_not_load_model=load_from_file,
    #             change_config=change_conf ig)
    #
    # print(p.current_pred_path())
    p = Predict(
        model_name=model_base_name,
        model_type=model_type,
        do_not_load_model=True,
        change_config=change_config,
    )
    print(p.current_pred_path())

    if num_workers is not None:
        p.config.update(
            {
                "num_workers": num_workers,
            },
            allow_val_change=True,
        )

    child_name = "deep_ensemble"
    if len(exclude_models) > 0:
        child_name += "_exclude_{}".format(exclude_models)
    p = stochastic_inference(
        p,
        n_its,
        load_from_file,
        child_name,
        model_names=model_names,
        postfix_empty=True,
        model_base_name=model_base_name,
        predict_set=predict_set,
    )

    return p


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"

    set_random_seeds(seed=0)

    for model_base_name in [
        # "pretrained-maxpooling",
        # "pretrained-meanpooling",
        # "pretrained-attentionpooling",
        # "pretrained-distributionpooling",
        # "pretrained-distributionwithattentionpooling",
        "pretrained-transmilpooling",
    ]:
        for predict_set in ["val", "test"]:
            deep_ensemble(
                model_base_name=model_base_name,
                model_type="best_quadratic_weighted_kappa",
                n_its=5,
                load_from_file=False,
                predict_set=predict_set,
                change_config={
                    "data_root": "ubix-dummy-data",
                    "num_workers": 0
                    # 'data_root': '/Volumes/retina/data/EUGENDA_OCT_NIFTI/spacing_0.013899999670684338_0.0038999998942017555_-1.0_size_735_496_-1_no_pad'
                },
            )
