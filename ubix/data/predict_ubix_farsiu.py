import os
from ubix.ubix_class import UBIX
from ubix.predict import Predict
import sklearn.metrics
import numpy as np
import argparse


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="/path/to/nifti/resampled")
    args = parser.parse_args()
    
    ubix_type = "soft"
    # ubix_type = "hard"
    # uncertainty_measure = "max_logit_max0th_confidence"
    uncertainty_measure = "ordinal_entropy"

    # model_name = "pretrained-maxpooling/0"
    # predictions_postfix = "_aug_10_mean"  # TTA
    
    model_name = "pretrained-maxpooling"
    predictions_postfix = "_deep_ensemble_5_mean"
    
    change_config = {"data_root": args.folder}
    
    save_path = f"./results/models/{model_name}/ubix_params_hval_{ubix_type}{predictions_postfix}_{uncertainty_measure}.p"

    ubix = UBIX(ubix_type=ubix_type, uncertainty_measure=uncertainty_measure)

    #ubix.save_ubix_parameters(save_path)
    ubix.load_ubix_parameters(save_path)

    test_predictor = Predict(
        model_name=model_name,
        model_type="best_quadratic_weighted_kappa",
        do_not_load_model=True,
        change_config=change_config
    )
    y_pred_ubix = ubix.predict(
        test_predictor, predict_set="test", predictions_postfix=predictions_postfix
    )
    predictions_non_ubix, _, _ = ubix.predict_non_ubix(
        test_predictor, predict_set="test", predictions_postfix=predictions_postfix
    )
    
    qwk = lambda a, b: sklearn.metrics.cohen_kappa_score(a, b, weights="quadratic")
    y_pred_ubix_merged = np.argmax(y_pred_ubix, -1)
    y_pred_ubix_merged = (y_pred_ubix_merged >= 1).astype(int)
    
    y_pred_non_ubix_merged = np.argmax(predictions_non_ubix["y_pred"], -1)
    y_pred_non_ubix_merged = (y_pred_ubix_merged >= 1).astype(int)
    
    qwk_ubix = qwk(predictions_non_ubix["y_true"], y_pred_ubix_merged)
    qwk_non_ubix = qwk(predictions_non_ubix["y_true"], y_pred_non_ubix_merged)
    
    print("qwk_ubix:", qwk_ubix)
    print("qwk_non_ubix:", qwk_non_ubix)

    pass
