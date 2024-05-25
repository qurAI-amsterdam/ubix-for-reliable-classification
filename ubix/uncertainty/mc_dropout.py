from ubix.predict import Predict
from ubix.utils import *
from ubix.uncertainty.stochastic_inference import stochastic_inference


def mc_dropout(
    model_name,
    model_type,
    n_its=10,
    load_from_file=False,
    config_modifications=None,
    load_all_to_memory=False,
    predict_set="val",
    num_workers=None,
):
    p = Predict(
        model_name=model_name,
        model_type=model_type,
        do_not_load_model=load_from_file,
        change_config=config_modifications,
    )

    if not load_from_file:
        p.config.update(
            {
                "force_dropout": True,
            },
            allow_val_change=True,
        )

        if num_workers is not None:
            p.config.update(
                {
                    "num_workers": num_workers,
                },
                allow_val_change=True,
            )

        assert p.config.drop_rate > 0

        if load_all_to_memory:
            p.val_set.load_all_to_memory()

    p = stochastic_inference(
        p, n_its, load_from_file, "mc_dropout", predict_set=predict_set
    )

    return p


if __name__ == "__main__":
    set_random_seeds(seed=0)

    for predict_set in ["val", "test"]:
        mc_dropout(
            model_name="pretrained-maxpooling_mcdo/0",
            model_type="best_quadratic_weighted_kappa",
            n_its=32,
            load_from_file=False,
            predict_set="test",
        )
