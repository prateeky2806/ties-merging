import os
import torch
import json
import re
import numpy as np

from collections import OrderedDict
from src.utils.utils import getValueOfKey_inDictionary, saveTo_gcp, safe_makedirs
from src.utils.distributed_utils import is_distributedSetup

from src.model.load_model import get_modelParameters_stateDict
from src.model.utils import softmax_with_temperature


METRICS_PRIORITY = ["score_to_select_checkpoint", "loss", "batch_idx"]


class Checkpointer(object):
    def __init__(
        self,
        trainableParameter_regex,
        experiment_dir,
        should_saveMostRecentCheckpoint,
        should_saveEveryCheckpoint,
        world_size,
        should_saveToGCP,
        gradient_accumulation_factor,
        current_bestScore,
        training_config=None,
    ):
        self.trainableParameter_regex = trainableParameter_regex
        self.experiment_dir = experiment_dir
        self.should_saveMostRecentState = should_saveMostRecentCheckpoint
        self.should_saveEveryCheckpoint = should_saveEveryCheckpoint
        self.world_size = world_size
        self.should_saveToGCP = should_saveToGCP
        self.gradient_accumulation_factor = gradient_accumulation_factor
        self.training_config = training_config

        self.runningSum_ofMetrics = {}
        self.number_ofMetricUpdates = 0

        self.current_bestScore = current_bestScore
        self.numCheckpoints_sinceBestCheckpoint = 0
        self.best_task_mixing_weights = []
        self.best_pt_mixing_weights = []

    # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    def _convertDistributedDict_toNonDistributedDict(self, state_dict):

        if is_distributedSetup(self.world_size):
            nonDistributed_stateDict = OrderedDict()
            pattern = re.compile("module.")
            for k, v in state_dict.items():
                if re.search("module", k):
                    nonDistributed_stateDict[re.sub(pattern, "", k)] = v
                else:
                    nonDistributed_stateDict = state_dict

            return nonDistributed_stateDict
        else:
            return state_dict

    def _get_trainableParameters(self, model):
        nonDistributed_stateDict = self._convertDistributedDict_toNonDistributedDict(
            model.state_dict()
        )
        trainable_parameters = get_modelParameters_stateDict(
            nonDistributed_stateDict, self.trainableParameter_regex
        )
        return trainable_parameters

    def _is_bestCheckpoint(self, current_log):
        """

        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        return current_score > self.current_bestScore

    def _update_bestCheckpoint(self, current_log):
        """

        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        self.current_bestScore = current_score
        self.numCheckpoints_sinceBestCheckpoint = 0

    def _save_checkpoint(self, model, save_fp):
        torch.save(self._get_trainableParameters(model), save_fp)
        saveTo_gcp(self.should_saveToGCP, save_fp)

    def _save_trainingState(self, optimizer, scheduler, batch_idx, save_fp):
        """

        Args:
            optimizer:
            scheduler:
            batch_idx:
            save_fp:

        Returns:

        """
        current_stateDict = {
            "num_batches": batch_idx,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "current_best_score": self.current_bestScore,
        }
        torch.save(current_stateDict, save_fp)
        saveTo_gcp(self.should_saveToGCP, save_fp)

    def update_runningSumOfMetrics(self, current_metrics):
        """
        Update running sum of metrics

        Args:
            current_metrics:

        """
        for k in current_metrics.keys():
            # Divide by gradient accumulation factor similar to loss so that the the metric is
            # computed per example
            scaled_metric = current_metrics[k] / self.gradient_accumulation_factor

            if len(self.runningSum_ofMetrics) == 0:
                self.runningSum_ofMetrics[k] = scaled_metric
            else:
                self.runningSum_ofMetrics[k] += scaled_metric

        self.number_ofMetricUpdates += 1

    def _get_averageMetrics(self):
        """
        Get average metric per batch since the last time we got the average.

        Note that average is per effective batch size, not batch size
        (i.e. every gradient update, not every forward pass).

        :return: average_metric
        """
        average_metric = {}
        for k in self.runningSum_ofMetrics.keys():
            average_metric[k] = float(
                "%.3f" % (self.runningSum_ofMetrics[k] / self.number_ofMetricUpdates)
            )

        # Reset running dict_metrics and counter when we take average
        self.runningSum_ofMetrics = {}
        self.number_ofMetricUpdates = 0

        return average_metric

    def _log_metricAndScores(self, batch_idx, scores):
        """
        Log current metrics and scores

        Args:
            batch_idx:
            scores:

        Returns:

        """
        current_log = {}
        current_log["batch_idx"] = batch_idx
        current_log.update(self._get_averageMetrics())
        current_log.update(scores)

        log_fp = os.path.join(self.experiment_dir, "log.json")

        with open(log_fp, "a+") as f_out:
            f_out.write(json.dumps(current_log))
            f_out.write("\n")

        saveTo_gcp(self.should_saveToGCP, log_fp)

        return current_log

    def checkpoint(
        self, model, optimizer, scheduler, scores, batch_idx, dont_saveModel=False
    ):
        """
        Handles checkpointing which means
        1) logging metrics and scores
        2) saving the model if needed

        Args:
            model:
            scores:
            batch_idx:
            dont_saveModel:


        Returns:
            current_log
        """
        current_log = self._log_metricAndScores(batch_idx, scores)

        self.numCheckpoints_sinceBestCheckpoint += 1

        if not dont_saveModel:
            if self.should_saveMostRecentState:
                self._save_trainingState(
                    optimizer,
                    scheduler,
                    batch_idx,
                    os.path.join(self.experiment_dir, f"training_state.dict"),
                )

            if self.should_saveEveryCheckpoint:
                checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
                safe_makedirs(checkpoint_dir)
                self._save_checkpoint(
                    model, os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.pt")
                )

            else:
                if self._is_bestCheckpoint(current_log):
                    bestModel_fp = os.path.join(self.experiment_dir, "best_model.pt")
                    self._save_checkpoint(model, bestModel_fp)

        if self._is_bestCheckpoint(current_log):
            self._update_bestCheckpoint(current_log)
            self._update_and_save_best_mixing_weights(model, current_log)

        return current_log, self.numCheckpoints_sinceBestCheckpoint

    def _update_and_save_best_mixing_weights(self, model, current_log):
        # write mixing weights to a file
        mixing_weights_fp = os.path.join(self.experiment_dir, "mixing_weights.json")
        dump_str = {}

        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "task_mixing_weights"):
                self.best_task_mixing_weights = (
                    model.transformer.task_mixing_weights.detach().cpu()
                )
                dump_str["best_task_mixing_weights"] = np.round(
                                np.array(softmax_with_temperature(self.best_task_mixing_weights, self.training_config.temperature)), 2
                            ).tolist()
            if hasattr(
                model.transformer, "pretrained_mixing_weights"
            ):
                self.best_pt_mixing_weights = (
                    model.transformer.pretrained_mixing_weights.detach().cpu()
                )
                dump_str["best_pt_mixing_weights"] = np.round(
                                np.array(softmax_with_temperature(self.best_pt_mixing_weights, self.training_config.temperature)), 2
                            ).tolist()

        dump_str['best_log'] = current_log

        with open(mixing_weights_fp, "a+") as f_out:
            json.dump(
                dump_str,
                f_out,
            )
            f_out.write("\n")
