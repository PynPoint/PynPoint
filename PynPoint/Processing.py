from abc import ABCMeta, abstractmethod


class PypelineModule:
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self):
        pass


class WritingModule(PypelineModule):

    def __int__(self):
        pass