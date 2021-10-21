import os
import random
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from desed_task.utils.scaler import TorchScaler
import numpy as np
from desed_task.data_augm.rand_augm_agg import RandAugment
from .utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
)
# from desed_task.data_augm.rand_augm_agg import RandAugment

class SEDTask4_2021(pl.LightningModule):
    """ Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed_student: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: asteroid.engine.schedulers.BaseScheduler subclass object, the scheduler to be used. This is
            used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
            self,
            hparams,
            encoder,
            sed_student,
            device,
            opt=None,
            train_data=None,
            valid_data=None,
            test_data=None,
            train_sampler=None,
            scheduler=None,
            fast_dev_run=False,
    ):
        super(SEDTask4_2021, self).__init__()
        self.hparams = hparams
        # print("Using ICT loss")
        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = deepcopy(sed_student)
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        ).to(device)

        for param in self.sed_teacher.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError
        if hparams["augs"]["consis_loss"] == "bce":
            self.consis_loss_func = torch.nn.BCELoss()
        elif hparams["augs"]["consis_loss"] == "mse":
            self.consis_loss_func = torch.nn.MSELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = pl.metrics.classification.F1(
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        self.get_weak_teacher_f1_seg_macro = pl.metrics.classification.F1(
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        self.scaler = self._init_scaler()

        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}

        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()

        self.using_consis = hparams["augs"]["consis"]
        self.augmentor = RandAugment(self.hparams, device=device)
        print("mixup method:", hparams['augs']["mixup"])
        print("mixup scale:", hparams['augs']['mixup_scale'])
        print("other augmentations:", hparams['augs']['aug_methods'])
        print("augmentation scale:", hparams["augs"]["aug_scale"])
        print("using consistency loss:", hparams["augs"]["consis"])
        print("Rand Augmenter settled done!")

    def update_ema(self, alpha, global_step, model, ema_model):
        """ Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        """ Scaler inizialization

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        """ Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, model, temp=None):
        return model(self.scaler(self.take_log(mel_feats)), temp=temp)

    def training_step(self, batch, batch_indx):
        """ Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        audio, labs, padded_indxs = batch
        features, labels, masks, distortions = self.augmentor.random_augment(audio, labs)
        # print(features.shape)
        indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        bsz = indx_synth + indx_weak + indx_unlabelled
        weak_mask = masks["weak"]
        strong_mask = masks["strong"]
        unsup_mask = masks["unsup"]

        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        # sed student forward
        strong_preds_student, weak_preds_student = self.detect(
            features, self.sed_student
        )

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds_student[strong_mask], labels[strong_mask]
        )
        loss_weak = self.supervised_loss(weak_preds_student[weak_mask], labels_weak)
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak

        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.detect(
                features, self.sed_teacher
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[strong_mask], labels[strong_mask]
            )

            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak
            )
        # we apply consistency between the predictions, use the scheduler for learning rate
        weight = 2 * self.scheduler["scheduler"]._get_scaling_factor()

        strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student, strong_preds_teacher.detach()
        )
        weak_self_sup_loss = self.selfsup_loss(
            weak_preds_student, weak_preds_teacher.detach()
        )
        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight

        if self.using_consis:
            # Self-consistency loss
            weak_consis_loss = self.consistency_loss(weak_preds_student,
                                                     strong_preds_student,
                                                     weak_mask,
                                                     indx_weak,
                                                     distortions["weak"])
                                                     # teacher=(weak_preds_teacher.detach(), strong_preds_teacher.detach()))
            strong_consis_loss = self.consistency_loss(weak_preds_student,
                                                       strong_preds_student,
                                                       strong_mask,
                                                       indx_synth,
                                                       distortions["strong"])
                                                       # teacher=(weak_preds_teacher.detach(), strong_preds_teacher.detach()))
            unsup_consis_loss = self.consistency_loss(weak_preds_student,
                                                      strong_preds_student,
                                                      unsup_mask,
                                                      indx_unlabelled,
                                                      distortions["unsup"])
                                                      # teacher=(weak_preds_teacher.detach(), strong_preds_teacher.detach()))     # teacher=(weak_preds_teacher, strong_preds_teacher)

            tot_consis_loss = (weak_consis_loss / 4 + strong_consis_loss / 4 + unsup_consis_loss / 2) * weight

            tot_loss = tot_loss_supervised + tot_self_loss + tot_consis_loss
            self.log("train/consistency_loss", tot_consis_loss)
        else:
            tot_loss = tot_loss_supervised + tot_self_loss

        self.log("train/student/tot_loss", tot_loss)
        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/step", self.scheduler["scheduler"].step_num, prog_bar=True)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/weight", weight)
        self.log("train/student/tot_supervised", strong_self_sup_loss, prog_bar=True)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)
        return tot_loss

    def consistency_loss(self, weak_preds_student, strong_preds_student, mask, num_data, distortions, teacher=None):
        if teacher is None:
            teacher = (weak_preds_student, strong_preds_student)
        # Loss from model weak prediction
        weak_preds_loss = self.consistency_loss_opt(weak_preds_student[mask],
                                                    num_data,
                                                    distortions,
                                                    shift_ignore=True,
                                                    ref=teacher[0][mask])
        # Loss from model strong prediction
        strong_preds_loss = self.consistency_loss_opt(strong_preds_student[mask],
                                                      num_data,
                                                      distortions,
                                                      ref=teacher[1][mask])
        return weak_preds_loss + strong_preds_loss
        # return strong_preds_loss

    def consistency_loss_opt(self, preds, num_data, distortions, shift_ignore=False, ref=None):
        loss_augment = self.recover_loss_opt(preds, num_data, distortions["augments"], shift_ignore, _cat=False, _ref=ref)
        loss_mixup = self.recover_loss_opt(preds, num_data, distortions["mixup"], shift_ignore, _cat=False, _ref=ref)
        self.log("train/consistency/mixup", loss_mixup)
        self.log("train/consistency/augment", loss_augment)
        return (loss_augment + loss_mixup) / 2

    def recover_loss_opt(self, _preds, _num_data, _distortion, _shift_ignore, _ref=None, _cat=False):
        if _distortion[0] == "Soft mixup":
            perm_list, weights = _distortion[1], _distortion[2]
            teacher_preds = _ref[:_num_data]
            mixed_preds = weights[0] * teacher_preds  # Changed 17:00 7/7
            for perm, w in zip(perm_list, weights[1:]):
                mixed_preds += w * _ref[perm]
            return self.consis_loss_func(
                _preds[_num_data: _num_data * 2],
                mixed_preds
             )

        if _distortion[0] == "Hard mixup":
            perm_list = _distortion[1]
            org_preds = _preds[:_num_data]
            mixed_higher, mixed_lower = self.binary_pred(org_preds)
            for perm in perm_list:
                mixed_cp_higher, mixed_cp_lower = self.binary_pred(org_preds[perm])
                mixed_higher = mixed_higher | mixed_cp_higher
                mixed_lower = mixed_lower & mixed_cp_lower
            final_mask = mixed_higher | mixed_lower
            return self.consis_loss_func(
                _preds[_num_data: _num_data * 2] * final_mask.type(_preds.type()),
                mixed_higher.type(_preds.type()),
            )

        elif _distortion[0] == "Time shift" and (not _shift_ignore):
            shift_scale = _distortion[1]
            shifted_preds = torch.cat([_ref[:_num_data, :, -shift_scale:],
                                       _ref[:_num_data, :, :-shift_scale]], dim=2)
            return self.consis_loss_func(
                _preds[_num_data * 2:],
                shifted_preds,
            )

        else:
            return self.consis_loss_func(
                _preds[_num_data * 2:],
                _ref[:_num_data]
            )

    def binary_pred(self, preds, upper=0.95, lower=0.05):
        return preds > upper, preds < lower

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )

    def truncate_target(self, x, thd_up=0.8, thd_down=0.2, mimic=None):
        if mimic is None:
            return x > thd_up
        else:
            x_conf = x > thd_up
            x_uncer = mimic * ((thd_down < x) * (x < thd_up))
            return x_conf + x_uncer

    def validation_step(self, batch, batch_indx):
        """ Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames = batch
        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student, temp=1)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher, temp=1)

        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
                .to(audio)
                .bool()
        )
        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["synth_val_folder"]))
                    for x in filenames
                ]
            )
                .to(audio)
                .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()
            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )
            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[mask_weak], labels_weak
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["synth_val_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_student_synth.keys()),
            )

            for th in self.val_buffer_student_synth.keys():
                self.val_buffer_student_synth[th] = self.val_buffer_student_synth[
                    th
                ].append(decoded_student_strong[th], ignore_index=True)

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_teacher_synth.keys()),
            )
            for th in self.val_buffer_teacher_synth.keys():
                self.val_buffer_teacher_synth[th] = self.val_buffer_teacher_synth[
                    th
                ].append(decoded_teacher_strong[th], ignore_index=True)

        return

    def validation_epoch_end(self, outputs):
        """ Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """

        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()

        # synth dataset
        intersection_f1_macro_student = compute_per_intersection_macro_f1(
            self.val_buffer_student_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["synth_val_dur"],
        )

        synth_student_event_macro = log_sedeval_metrics(
            self.val_buffer_student_synth[0.5], self.hparams["data"]["synth_val_tsv"],
        )[0]

        intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
            self.val_buffer_teacher_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["synth_val_dur"],
        )

        synth_teacher_event_macro = log_sedeval_metrics(
            self.val_buffer_teacher_synth[0.5], self.hparams["data"]["synth_val_tsv"],
        )[0]

        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = intersection_f1_macro_student
        elif obj_metric_synth_type == "event":
            synth_metric = synth_student_event_macro
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_student
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        obj_metric = torch.tensor(weak_student_f1_macro.item() + synth_metric)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/macro_F1", weak_student_f1_macro)
        self.log("val/weak/teacher/macro_F1", weak_teacher_f1_macro)
        self.log(
            "val/synth/student/intersection_f1_macro", intersection_f1_macro_student
        )
        self.log(
            "val/synth/teacher/intersection_f1_macro", intersection_f1_macro_teacher
        )
        self.log("val/synth/student/event_f1_macro", synth_student_event_macro)
        self.log("val/synth/teacher/event_f1_macro", synth_teacher_event_macro)

        # free the buffers
        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """ Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames = batch
        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student, temp=1)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher, temp=1)

        loss_strong_student = self.supervised_loss(strong_preds_student, labels)
        loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

        self.log("test/student/loss_strong", loss_strong_student)
        self.log("test/teacher/loss_strong", loss_strong_teacher)

        # compute psds
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()),
        )

        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = self.test_psds_buffer_student[
                th
            ].append(decoded_student_strong[th], ignore_index=True)

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()),
        )

        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = self.test_psds_buffer_teacher[
                th
            ].append(decoded_teacher_strong[th], ignore_index=True)

        # compute f1 score
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )

        self.decoded_student_05_buffer = self.decoded_student_05_buffer.append(
            decoded_student_strong[0.5]
        )

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )

        self.decoded_teacher_05_buffer = self.decoded_teacher_05_buffer.append(
            decoded_teacher_strong[0.5]
        )

    def on_test_epoch_end(self):
        # pub eval dataset
        try:
            log_dir = self.logger.log_dir
        except Exception as e:
            log_dir = self.hparams["log_dir"]
        save_dir = os.path.join(log_dir, "metrics_test")

        psds_score_scenario1 = compute_psds_from_operating_points(
            self.test_psds_buffer_student,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "student", "scenario1"),
        )

        psds_score_scenario2 = compute_psds_from_operating_points(
            self.test_psds_buffer_student,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "student", "scenario2"),
        )

        psds_score_teacher_scenario1 = compute_psds_from_operating_points(
            self.test_psds_buffer_teacher,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "teacher", "scenario1"),
        )

        psds_score_teacher_scenario2 = compute_psds_from_operating_points(
            self.test_psds_buffer_teacher,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "teacher", "scenario2"),
        )

        event_macro_student = log_sedeval_metrics(
            self.decoded_student_05_buffer,
            self.hparams["data"]["test_tsv"],
            os.path.join(save_dir, "student"),
        )[0]

        event_macro_teacher = log_sedeval_metrics(
            self.decoded_teacher_05_buffer,
            self.hparams["data"]["test_tsv"],
            os.path.join(save_dir, "teacher"),
        )[0]
        # synth dataset
        intersection_f1_macro_student = compute_per_intersection_macro_f1(
            {"0.5": self.decoded_student_05_buffer},
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
        )

        # synth dataset
        intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
            {"0.5": self.decoded_teacher_05_buffer},
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
        )

        best_test_result = torch.tensor(max(psds_score_scenario1, psds_score_scenario2))

        results = {
            "hp_metric": best_test_result,
            "test/student/psds_score_scenario1": psds_score_scenario1,
            "test/student/psds_score_scenario2": psds_score_scenario2,
            "test/teacher/psds_score_scenario1": psds_score_teacher_scenario1,
            "test/teacher/psds_score_scenario2": psds_score_teacher_scenario2,
            "test/student/event_f1_macro": event_macro_student,
            "test/student/intersection_f1_macro": intersection_f1_macro_student,
            "test/teacher/event_f1_macro": event_macro_teacher,
            "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
        }
        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=False)

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader
