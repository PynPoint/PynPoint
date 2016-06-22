"""
Different interfaces for Pypeline Modules.
"""

# external modules
from abc import ABCMeta, abstractmethod
import os
import warnings

# own modules
from PynPoint.DataIO import OutputPort, InputPort


class PypelineModule:
    """
    Abstract interface for the different pipeline Modules
        - Reading Module
        - Writing Module
        - Processing Module
    Each pipeline module has a name as a unique identifier in the Pypeline processing step
    dictionary and has to implement the functions connect_database and run which are used in the
    Pypeline methods.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in):
        """
        Abstract constructor of a Pypeline Module which needs a Module name as identifier, checks
        its type and saves it.
        :param name_in: The name of the Pypeline Module
        :type name_in: String
        :return: None
        """

        assert (isinstance(name_in, str)), "Error: Name needs to be a String"
        self._m_name = name_in

    @property
    def name(self):
        """
        Returns the name of the Pypeline Module. The property makes sure that the internal Module
        name can not be changed.
        :return: The name of the Module
        :rtype: String
        """

        return self._m_name

    @abstractmethod
    def connect_database(self,
                         data_base_in):
        """
        Abstract interface for the function connect_database which is needed to connect the Ports
        of a Pypeline Module with the Pypeline Data Storage
        :param data_base_in: The Data Storage
        """
        pass

    @abstractmethod
    def run(self):
        """
        Abstract interface for the run method of a Pypeline Module which should execute the
        algorithm behind the module.
        """
        pass


class WritingModule(PypelineModule):
    """
    The abstract class WritingModule is a interface for processing steps in the pipeline which do
    not change the content of the internal Data Storage. They only have reading access to the data
    base. Writing Modules can be used to export save or plot data in the .hdf5 data base using a
    different file format. Since Writing Modules are Pypeline Modules they have a name. In addition
    one can specify a directory on the hard drive where the output of the module will be saved.
    If no output directory is given the default Pypline output directory is used. Writing modules
    only have a dictionary of input ports (self._m_input_ports) (the tags / keys of the data to be
    saved or plotted) but no output ports.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in,
                 output_dir=None):
        """
        Abstract constructor of a Writing Module which needs the unique name identifier as input
        (see Pypeline Module). In addition one can specify a output directory where the module will
        save its results. If no output directory is given the Pypline default is used.
        :param name_in: The name of the Writing Module
        :type name_in: String
        :param output_dir: Directory where the results will be saved
        :type output_dir: String (Needs to be a directory, raises error if not)
        :return: None
        """

        # call super for saving the name
        super(WritingModule, self).__init__(name_in)

        # If output_dir is None its location will be the Pypeline default output directory
        assert (os.path.isdir(str(output_dir))
                or output_dir is None), 'Error: Output directory for writing module does not exist'\
                                        ' - input requested: %s' % output_dir
        self.m_output_location = output_dir
        self._m_input_ports = {}

    def add_input_port(self,
                       tag):
        """
        Method which creates a input port and append it to the internal port dictionary. This
        function should be used by classes inhering from Writing Module to make sure that only
        input ports with unique tags are added. The new port can be used by self._m_input_ports[tag]
        :param tag: Tag of the new input port.
        :return: None
        """

        tmp_port = InputPort(tag)
        if tag in self._m_input_ports:
            warnings.warn('Tag already used. Updating..')

        self._m_input_ports[tag] = tmp_port

    def connect_database(self,
                         data_base_in):
        """
        Connects all ports in the internal input port dictionary to the given database.
        :param data_base_in: The input database
        :return: None
        """

        for port in self._m_input_ports.itervalues():
            port.set_database_connection(data_base_in)

    @abstractmethod
    def run(self):
        pass


class ProcessingModule(PypelineModule):
    """
    The abstract class ProcessingModule is a interface for all processing steps in the pipeline
    which capsule a pipeline step with a specific algorithm. They have reading and writing access to
    the data base. Since Writing Modules are Pypeline Modules they have a name. Processing modules
    have a dictionary of input ports (self._m_input_ports) (The data needed for the processing step
    and a dictionary of output ports (self._m_output_ports) (Results of the processing step).
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in):
        """
        Abstract constructor of a ProcessingModule which needs the unique name identifier as input
        (see Pypeline Module).
        :param name_in: The name of the Writing Module
        :type name_in: String
        :return: None
        """

        super(ProcessingModule, self).__init__(name_in)

        self._m_input_ports = {}
        self._m_output_ports = {}

    def add_input_port(self,
                       tag):
        """
        Method which creates a input port and append it to the internal input port dictionary. This
        function should be used by classes inhering from Processing Module to make sure that only
        input ports with unique tags are added. The new port can be used by self._m_input_ports[tag]
        :param tag: Tag of the new input port.
        :return: None
        """

        tmp_port = InputPort(tag)
        if tag in self._m_input_ports:
            warnings.warn('Tag already used. Updating..')

        self._m_input_ports[tag] = tmp_port

    def add_output_port(self,
                        tag):
        """
        Method which creates a output port and append it to the internal output port dictionary.
        This function should be used by classes inhering from Processing Module to make sure that
        only output ports with unique tags are added. The new port can be used by
        self._m_input_ports[tag]
        :param tag: Tag of the new output port.
        :return: None
        """

        tmp_port = OutputPort(tag)
        if tag in self._m_input_ports:
            warnings.warn('Tag already used. Updating..')

        self._m_input_ports[tag] = tmp_port

    def connect_database(self,
                         data_base_in):
        """
        Connects all ports in the internal input and output port dictionary to the given database.
        :param data_base_in: The input database
        :return: None
        """

        for port in self._m_input_ports.itervalues():
            port.set_database_connection(data_base_in)
        for port in self._m_output_ports.itervalues():
            port.set_database_connection(data_base_in)

    @abstractmethod
    def run(self):
        pass


class ReadingModule(PypelineModule):
    """
    The abstract class ReadingModule is a interface for processing steps in the pipeline which do
    not use any information of the internal data storage. They only have writing access to the data
    base which makes the the perfect tool to load data of a different file format to the data base.
    Since Reading Modules are Pypeline Modules they have a name. In addition
    one can specify a directory on the hard drive where the input data of the module is located.
    If no input directory is given the default Pypline input directory is used. Reading modules
    only have a dictionary of output ports (self._m_out_ports) (the tags / keys of the data to be
    saved to the data base) but no input ports.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in,
                 input_dir=None):
        """
        Abstract constructor of a ReadingModule which needs the unique name identifier as input
        (see Pypeline Module). In addition on can specify a input directory where data is located
        which will be loaded by the module. If no directory is given the Pipeline default is used.
        :param name_in: The name of the Reading Module
        :type name_in: String
        :param input_dir: Directory where the input files are located
        :type input_dir: String (Needs to be a directory, raises error if not)
        :return: None
        """

        super(ReadingModule, self).__init__(name_in)

        # If input_dir is None its location will be the Pypeline default input directory
        assert (os.path.isdir(str(input_dir))
                or input_dir is None), 'Error: Input directory for reading module does not exist ' \
                                       '- input requested: %s' % input_dir
        self.m_input_location = input_dir
        self._m_out_ports = {}

    def add_output_port(self,
                        tag,
                        default_activation=True):
        """
        Method which creates a output port and append it to the internal output port dictionary.
        This function should be used by classes inhering from Reading Module to make sure that
        only output ports with unique tags are added. The new port can be used by
        self._m_out_ports[tag]
        :param tag: Tag of the new output port.
        :return: None
        """

        tmp_port = OutputPort(tag,
                              default_activation)
        if tag in self._m_out_ports:
            warnings.warn('Tag already used. Updating..')

        self._m_out_ports[tag] = tmp_port

    def connect_database(self,
                         data_base_in):
        """
        Connects all ports in the internal output port dictionary to the given database.
        :param data_base_in: The input database
        :return: None
        """

        for port in self._m_out_ports.itervalues():
            port.set_database_connection(data_base_in)

    @abstractmethod
    def run(self):
        pass
