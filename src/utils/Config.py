import copy
import ast
import os.path


class Config(object):
    def __init__(self):
        """
        Base config class
        """

    def get_dict(self):
        """

        Returns:

        """
        return copy.deepcopy(self.__dict__)

    def _update_fromDict(self, dict_toUpdateFrom, assert_keyInUpdateDict_isValid):
        """

        Args:
            dict_toUpdateFrom:
            assert_keyInUpdateDict_isValid: If True, then error is thrown if key in the dict_toUpdateFrom does
                not exist in self.config

        Returns:

        """
        updated_attributes = []
        for k, v in dict_toUpdateFrom.items():
            try:
                # For strings that are actually filepaths, literal eval will fail so we have to ignore
                # strings which are filepaths. We check a string is a filepath if a "/" is in string.
                if not (isinstance(v, str) and "/" in v):
                    v = ast.literal_eval(v)
            except ValueError:
                v = v

            if hasattr(self, k):
                setattr(self, k, v)
                updated_attributes.append(k)

            else:
                if assert_keyInUpdateDict_isValid:
                    raise ValueError(f"{k} is not in the config")
        return updated_attributes
