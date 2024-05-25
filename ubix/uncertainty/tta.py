from ubix.predict import Predict
from ubix.utils import *
from ubix.uncertainty.stochastic_inference import stochastic_inference


def tta(
    model_name,
    model_type,
    n_tta=10,
    load_from_file=False,
    change_config=None,
    predict_set="val",
):
    """
    n_tta: Number of test time augmentations
    load_from_file: Whether to load the predictions from file. They will be computed otherwise.
    """
    p = Predict(
        model_name=model_name,
        model_type=model_type,
        do_not_load_model=load_from_file,
        change_config=change_config,
    )

    p.override_aug_transforms = get_tta_transforms(
        p.config, translate_range_z=0, device="cuda:1"
    )
    # p.override_aug_transforms = get_tta_transforms(p.config, translate_range_z=0, device='cpu')
    # p.override_aug_transforms = None

    p.update_data_loaders()

    p = stochastic_inference(p, n_tta, load_from_file, "aug", predict_set=predict_set)

    return p


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"

    set_random_seeds(seed=0)

    n_tta = 10

    # tta(
    #     model_name='2020-11-09-eugenda_2dmil_ensemble/0',
    #     model_type='best_quadratic_weighted_kappa',
    #     n_tta=n_tta,
    #     load_from_file=False,
    #     predict_set='val',
    # )

    tta(
        model_name="pretrained-maxpooling/0",
        model_type="best_quadratic_weighted_kappa",
        n_tta=n_tta,
        load_from_file=False,
        predict_set="test",
    )
