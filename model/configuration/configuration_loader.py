import os
from enum import Enum, auto

import yaml


class Tribe(Enum):
    TRAIN = auto()
    TEST = auto()
    CUSTOM = auto()

class EquationCategory(Enum):
    MODEL = auto()
    POSTSYNAPTIC_EVENT = auto()
    PRESYNAPTIC_EVENT = auto()
    RESET = auto()
    REFRACTORY = auto()
    TRESH_HOLD = auto()
    METHOD = auto()


class DictionaryCategory(Enum):
    STATE = auto()


class ConfigurationLoader:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.equations: {} = None
        self.parse_equations()

    def parse_equations(self):
        local_path = os.path.abspath(os.getcwd())
        path_to_yaml = os.path.join(local_path, self.file_name)
        with open(path_to_yaml) as file:
            self.equations = yaml.safe_load(file)

    def read_equations(self) -> dict:
        return self.equations

    def load_equations_if_preset(self, tribe: Tribe, category: EquationCategory) -> str:
        if category.name not in self.get_tribe(tribe):
            return None
        return self.load_equations(tribe, category)

    def load_equations(self, tribe: Tribe, category: EquationCategory) -> str:
        if category.name not in self.get_tribe(tribe):
            raise Exception(category.name + " not defined for " + tribe.name)
        return "\n".join(self.get_tribe(tribe)[category.name])

    def load_state_if_exist(self, tribe: Tribe):
        if DictionaryCategory.STATE.name in self.get_tribe(tribe):
            return self.load_state(tribe)
        return None

    def load_state(self, tribe: Tribe) -> dict:
        return self.get_tribe(tribe)[DictionaryCategory.STATE.name]

    def get_tribe(self, tribe: Tribe) -> dict:
        if tribe.name not in self.equations:
            raise Exception("Tribe " + tribe.name + " is not defined in equation files")
        return self.equations[tribe.name]
