# external modules
from abc import ABCMeta, abstractmethod, abstractproperty
import os
import warnings

# own modules
from DataIO import OutputPort

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
    def connect_database(self,
                         data_base_in):
        pass

    @abstractmethod
    def run(self):
        pass


class WritingModule(PypelineModule):
    __metaclass__ = ABCMeta

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

    # TODO: Ports



class ProcessingModule(PypelineModule):

    def __init__(self,
                 name_in):
        # TODO: Ports
        pass


class ReadingModule(PypelineModule):
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in=None,
                 input_dir=None):
        super(ReadingModule, self).__init__(name_in)
        # TODO Documentation

        # If input_dir is None its location will be the Pypeline default input directory
        assert (os.path.isdir(str(input_dir)) or input_dir is None), 'Error: Input directory for reading module' \
                                                                       ' does not exist - input requested: ' \
                                                                       '%s' % input_dir
        self.m_input_location = input_dir
        self._m_out_ports = {}

    def add_output_port(self,
                        tag,
                        default_activation=True):
        # TODO Documentation

        tmp_port = OutputPort(tag,
                              default_activation)
        if tag in self._m_out_ports:
            warnings.warn('Tag already used. Updating..')

        self._m_out_ports[tag] = tmp_port

    def connect_database(self,
                         data_base_in):
        for key, port in self._m_out_ports.iteritems():
            port.set_database_connection(data_base_in)
