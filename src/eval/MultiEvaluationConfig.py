import itertools

from src.eval.EvaluationConfig import EvaluationConfig
from src.utils.utils import breadth_first_search

class MultiEvaluationConfig(EvaluationConfig):
    def __init__(self,
                 fields_toIterateOver,
                 values_toIterateOver,
                 configDict_toInitializeFrom,
                 fields_toUpdate=None,
                 kwargs=None):
        '''

        Args:
            fields_toIterateOver: list of fields to iterate over
            conditionalFieldValues_toIterateOver: nested dictionary of possible values for 1
                                                  field to possible values for another field
                                                  where the possible values for each field depend on
                                                  one another
                                                  - Must have a mapping fields_toIterateOver to a
                                                  list of fields to iterate over where each field in
                                                  the list is the field of the
                                                  corresponding keys in that dictionary
            configDict_toInitializeFrom:
            fields_toUpdate:
            kwargs:
        '''
        super().__init__(configDict_toInitializeFrom, fields_toUpdate, kwargs)

        self.fields_toIterateOver = fields_toIterateOver
        self.values_toIterateOver = values_toIterateOver

    def get_allConfigs(self):
        '''

        Returns:

        '''

        iterated_configs = []

        if self.values_toIterateOver is None:
            listOf_listOfValues_toIterateOver = [self.get_dict()[k] for k in self.fields_toIterateOver]
            all_valueSettings = list(itertools.product(*listOf_listOfValues_toIterateOver))
        else:
            all_valueSettings = breadth_first_search(self.values_toIterateOver)

        for value_setting in all_valueSettings:
            updated_fields = dict(zip(self.fields_toIterateOver, value_setting))

            new_config = EvaluationConfig(
                configDict_toInitializeFrom=self.get_dict(),
                fields_toUpdate=updated_fields
            )

            iterated_configs.append(new_config)

        return iterated_configs