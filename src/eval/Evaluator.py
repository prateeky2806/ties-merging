import os
import json

from src.eval.Scorer import Scorer
from src.eval.Logger import Logger
from src.eval.utils import get_dirAndRunIdx_fromPredictionFp


class Evaluator(object):
    def __init__(self, metrics, prediction_fp, should_save_to_gcp):
        """
        Evaluates all metrics for a dataset

        Args:
            metrics:
            prediction_fp:
            should_save_to_gcp:
        """
        self.scorer = Scorer(metrics)
        self.prediction_fp = prediction_fp

        if self.prediction_fp is not None:
            self.logger = Logger(prediction_fp, should_save_to_gcp)

        self.seen_idxs = {}

    def add_batch(self, batchOf_evalInfo):

        batchOf_idxs = batchOf_evalInfo["idx"]

        # For distributed setup, the batch might have duplicate examples due to padding that we
        # have to remove.
        # 1) Compute the indices we have to remove
        idx_toRemove = []
        for batch_idx, idx in enumerate(batchOf_idxs):
            if idx in self.seen_idxs:
                idx_toRemove.append(batch_idx)
            self.seen_idxs[idx] = True

        # 2) Remove these indices
        filteredBatch_ofEvalInfo = {}
        for (key, batchOf_values) in batchOf_evalInfo.items():

            filtered_value = []
            for batch_idx, value in enumerate(batchOf_values):
                if batch_idx not in idx_toRemove:
                    filtered_value.append(value)

            filteredBatch_ofEvalInfo[key] = filtered_value

        self.scorer.add_batch(filteredBatch_ofEvalInfo)
        if self.prediction_fp is not None:
            self.logger.log_batch(filteredBatch_ofEvalInfo)

    def updateCache_runFinished(self):
        """

        Args:

        Returns:

        """
        specificPrediction_dir, run_idx = get_dirAndRunIdx_fromPredictionFp(
            self.prediction_fp
        )
        evaluationRuns_fp = os.path.join(specificPrediction_dir, "evaluation_runs.json")

        evaluationConfigDict_runs = json.load(open(evaluationRuns_fp, "r"))
        assert (
            len(evaluationConfigDict_runs) - 1 == run_idx
        ), "The idx of the run should be the last evaluation config, but is not."

        # TODO find a better solution to update evaluation config to mark the run finished,
        # since ideally evaluation config should be immutable.
        evaluationConfigDict_runs[-1]["did_run_finish"] = True
        assert evaluationConfigDict_runs[-1]["did_run_finish"]

        json.dump(evaluationConfigDict_runs, open(evaluationRuns_fp, "w+"))

    def get_result(self):
        """

        Returns:

        """
        if self.prediction_fp is not None:
            self.logger.close_logger()
            self.updateCache_runFinished()
        return self.scorer.get_score()
