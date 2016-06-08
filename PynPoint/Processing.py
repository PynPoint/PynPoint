from abc import ABCMeta, abstractmethod, abstractproperty
import os

class PypelineModule:
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in=None):
        assert (type(name_in) == str), "Error: Name needs to be a String"
        self._m_name = name_in

    @property
    def name(self):
        return self._m_name

    @abstractmethod
    def run(self):
        pass


class WritingModule(PypelineModule):

    def __init__(self,
                 name_in=None,
                 output_dir=None):
        # call super for saving the name
        super(WritingModule, self).__init__(name_in)

        # If output_dir is None its location will be the Pypeline default output directory
        assert (os.path.isdir(str(output_dir)) or output_dir is None), 'Error: Output directory for writing module' \
                                                                       ' does not exist - input requested: ' \
                                                                       '%s' % output_dir
        self.m_output_location = output_dir

    # TODO remove this run function
    def run(self):
        print "Working hard"

    # TODO: Ports


class ProcessingModule(PypelineModule):

    def __init__(self,
                 name_in):
        # TODO: Ports
        pass


class ReadingModule(PypelineModule):

    def __init__(self):
        # TODO: Ports
        pass
