import json

from src.utils.utils import convert_dictOfLists_to_listOfDicts, saveTo_gcp


class Logger(object):
    def __init__(self, logger_fp, should_saveToGCP):
        """

        Args:
            logger_fp:
            should_saveToGCP:
        """
        self.logger_fp = logger_fp
        self.should_saveToGCP = should_saveToGCP

        self.logger_file = open(self.logger_fp, "w+")

    def _convert_dictOfLists_to_listOfDicts(self, dictOfLists):
        return convert_dictOfLists_to_listOfDicts(dictOfLists)

    def log_batch(self, batchOf_evalInfo):
        listOf_evalInfo = self._convert_dictOfLists_to_listOfDicts(batchOf_evalInfo)
        for eval_info in listOf_evalInfo:
            self.logger_file.write(json.dumps(eval_info) + "\n")
        self.logger_file.flush()

    def close_logger(self):
        saveTo_gcp(self.should_saveToGCP, self.logger_fp)
