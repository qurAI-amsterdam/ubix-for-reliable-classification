from ubix.utils import *
import os
import hashlib
import pickle
from tqdm.auto import tqdm
import numpy as np


class Predict:
    def __init__(
        self,
        model_name=None,
        model_type=None,
        do_not_load_model=False,
        do_not_make_datasets=False,
        change_config=None,
        override_aug_transforms=(),
    ):
        self.model_type = model_type
        self.override_aug_transforms = override_aug_transforms
        self.model_name = model_name
        init_wandb_experiment(model_name, change_config)
        self.config = wandb.config
        self.do_not_load_model = do_not_load_model
        self.device = get_device()
        self.change_config = change_config
        update_config_dataset(self.config)

        self.train_set, self.val_set, self.test_set = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        if not do_not_make_datasets:
            self.update_data_loaders()

        self.set_model_type(self.model_type)

    def _change_config(self):
        for k, v in self.change_config.items():
            setattr(self.config, k, v)
            assert (
                getattr(self.config, k) == v
            ), f"for key {k}: {getattr(self.config, k) == v} != {v}"

    def update_data_loaders(self):
        self.train_set, self.val_set, self.test_set = get_data_set(
            self.config,
            subsets=("train", "val", "test"),
            override_aug_transforms=self.override_aug_transforms,
        )

        self.train_loader, self.val_loader, self.test_loader = make_data_loaders(
            self.config,
            self.train_set,
            self.val_set,
            self.device,
            test_set=self.test_set,
            force_no_balance=True,
        )

    def set_model_type(self, model_type):
        self.model_type = model_type

        if not self.do_not_load_model:
            self.load_model()

            self.model.to(self.device)

            if self.config.model_eval:
                self.model.eval()
            else:
                self.model.train()

            if self.config.force_dropout and hasattr(self.model, "enable_dropout"):
                self.model.enable_dropout()
                print("Did self.model.enable_dropout()")
            else:
                pass
                # print("Did not self.model.enable_dropout()")

    def load_model(self):
        self.load_model_returned = load_model(
            self.config,
            self.model_type,
            device=self.device,
            load_optimizer=False,
        )
        self.model = self.load_model_returned[0]

    def postfix(self, kwargs, set_name):
        predictions_postfix = (
            kwargs["predictions_postfix"] if "predictions_postfix" in kwargs else ""
        )

        predictions_postfix += "_" + set_name

        if self.change_config is not None:
            predictions_postfix += (
                "_changed"
                + str(list(self.change_config.keys()))
                + "to"
                + str(list(self.change_config.values()))
            )

        predictions_postfix = predictions_postfix.replace("/", "#")

        kwargs["predictions_postfix"] = predictions_postfix
        return predictions_postfix

    def predict_train_set(self, *args, **kwargs):
        self.postfix(kwargs, "train")
        return self.predict_data_loader(self.train_loader, *args, **kwargs)

    def predict_val_set(self, *args, **kwargs):
        return self.predict_validation_set(*args, **kwargs)

    def predict_validation_set(self, *args, **kwargs):
        self.postfix(kwargs, "val")
        return self.predict_data_loader(self.val_loader, *args, **kwargs)

    def predict_test_set(self, *args, **kwargs):
        self.postfix(kwargs, "test")
        return self.predict_data_loader(self.test_loader, *args, **kwargs)

    def x_set(self, set_name):
        return eval(f"self.{set_name}_set")

    def predict_x_set(self, set_name, *args, **kwargs):
        return eval(f"self.predict_{set_name}_set")(*args, **kwargs)

    @property
    def all_loader(self):
        data_sets_list = [self.train_set, self.val_set, self.test_set]
        data_set = torch.utils.data.ConcatDataset(
            [d for d in data_sets_list if d is not None]
        )
        data_loader = DataLoader(
            data_set,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return data_loader

    def predict_all_set(self, *args, **kwargs):
        self.postfix(kwargs, "all")
        return self.predict_data_loader(self.all_loader, *args, **kwargs)

    def predictions_postfix_hash(self, folder, predictions_postfix):
        hash_code = hashlib.sha256(bytearray(predictions_postfix, "utf-8")).hexdigest()
        fname = os.path.join(folder, "longfiles.json")
        run_silently(f"ls {folder}")
        dct = {}
        if os.path.exists(fname):
            with open(fname, "r") as f:
                dct = json.load(f)
        dct[hash_code] = predictions_postfix
        with open(fname, "w") as f:
            f.write(json.dumps(dct, indent=4))

        # print("Folder", folder)
        # print(f"Hashed {predictions_postfix} to {hash_code}")

        return hash_code

    def current_pred_path(self, predictions_postfix=""):
        folder = get_predictions_folder(self.config, self.model_type)

        def make_path():
            return os.path.join(
                folder,
                f"predictions_{self.config.data_subset_name}{predictions_postfix}.p",
            )

        out = make_path()

        if len(os.path.basename(out)) > 255:
            # print("hashing predictions_postfix", predictions_postfix)
            predictions_postfix = self.predictions_postfix_hash(
                folder, predictions_postfix
            )
            # print("hash output:", predictions_postfix)
            out = make_path()

        return out

    def predict_input(self, scan: str, lesions: str):
        raise NotImplementedError()

    def predict_data_loader(
        self,
        data_loader,
        load_from_file=True,
        predictions_postfix="",
        only_first_n=np.inf,
    ):
        outputs = []
        y_true = []

        predictions_path = self.current_pred_path(predictions_postfix)

        if (type(load_from_file) is bool and load_from_file) or (
            type(load_from_file) is str
            and load_from_file == "if_exists"
            and os.path.exists(predictions_path)
        ):
            run_silently(f"ls '{os.path.dirname(predictions_path)}'")
            with open(predictions_path, "rb") as f:
                return pickle.load(f)
        else:
            self.do_not_load_model = False
            self.set_model_type(self.model_type)

            has_mil_instances = self.config.architecture.startswith("timedist")
            y_pred_instances = []
            confidence_instances = []

            nrs = []
            for sample in tqdm(data_loader, "Looping over inference dataset"):
                x, y_true_i = sample["img"], sample["label"]
                x = x.to(self.device)

                with torch.no_grad():
                    if has_mil_instances:
                        y_pred_i, y_pred_instances_i = self.model.forward(
                            x, return_instances=True, return_devries_confidence=False
                        )

                        y_pred_instances.append(
                            y_pred_instances_i.cpu().detach().numpy()
                        )
                    else:
                        y_pred_i = self.model(x)

                outputs.append(tensor_or_tuple_of_tensors_to_numpy(y_pred_i)[0])
                y_true.append(y_true_i.cpu().detach().numpy())

                if len(outputs) >= only_first_n:
                    break

                nrs.append(sample["nr"][0])

            y_true = np.concatenate(y_true)

            y_pred = np.stack(outputs)

            out = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_instances": y_pred_instances,
                "confidence_instances": confidence_instances,
                "nrs": nrs,
            }

            with open(predictions_path, "wb") as f:
                pickle.dump(out, f)

            return out


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"

    set_random_seeds(seed=0)

    p = Predict(
        model_name="pretrained-meanpooling/0",
        model_type="best_quadratic_weighted_kappa",
        # change_config={
        #     'data_root': '/mnt/netcache/retina/data/Farsiu_NIFTI/resampled_[0.013899999670684338, 0.0038999998942017555, -1.0]',
        #     'resample_input': False
        # }
    )
    # p.predict_test_set(load_from_file=False)
    p.predict_val_set(load_from_file=False)
