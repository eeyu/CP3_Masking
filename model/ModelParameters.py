from abc import ABC, abstractmethod
from torch.nn import Module

class ModelParameters(ABC):
    @abstractmethod
    def getName(self) -> str:
        pass

    @abstractmethod
    def instantiate_new_model(self) -> Module:
        pass
    #
    # @abstractmethod
    # def get_configuration_space(self) -> ConfigurationSpace:
    # 	pass
    #
    # @abstractmethod
    # def set_from_configuration(self, config : Configuration):
    # 	pass