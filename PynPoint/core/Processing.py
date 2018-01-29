"""
Different interfaces for Pypeline Modules.
"""

import os
import warnings

from abc import abstractmethod, ABCMeta

import numpy as np

from PynPoint.util.Multiprocessing import LineProcessingCapsule, apply_function
from PynPoint.core.DataIO import OutputPort, InputPort, ConfigPort


class PypelineModule:
    """
    Abstract interface for the different pipeline Modules:

        * Reading Module (:class:`PynPoint.core.Processing.ReadingModule`)
        * Writing Module (:class:`PynPoint.core.Processing.WritingModule`)
        * Processing Module (:class:`PynPoint.core.Processing.ProcessingModule`)

    Each pipeline module has a name as a unique identifier in the Pypeline and has to implement the
    functions connect_database and run which are used in the Pypeline methods.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in):
        """
        Abstract constructor of a Pypeline Module which needs a name as identifier.

        :param name_in: The name of the Pypeline Module
        :type name_in: str
        :return: None
        """

        assert (isinstance(name_in, str)), "Error: Name needs to be a String"
        self._m_name = name_in
        self._m_data_base = None
        self._m_config_port = ConfigPort("config")

    @property
    def name(self):
        """
        Returns the name of the Pypeline Module. This property makes sure that the internal module
        name can not be changed.

        :return: The name of the Module
        :rtype: str
        """

        return self._m_name

    @abstractmethod
    def connect_database(self,
                         data_base_in):
        """
        Abstract interface for the function connect_database which is needed to connect the Ports
        of a PypelineModule with the Pypeline Data Storage.

        :param data_base_in: The Data Storage
        :type data_base_in: DataStorage
        """
        pass

    @abstractmethod
    def run(self):
        """
        Abstract interface for the run method of a Pypeline Module which inheres the actual
        algorithm behind the module.
        """
        pass


class WritingModule(PypelineModule):
    """
    The abstract class WritingModule is a interface for processing steps in the pipeline which do
    not change the content of the internal DataStorage. They only have reading access to the
    central data base. WritingModules can be used to export save or plot data from the .hdf5 data
    base by using a different file format. WritingModules know the directory on the hard drive
    where the output of the module can be saved. If no output directory is given the default
    Pypline output directory is used. WritingModules have a dictionary of input ports
    (self._m_input_ports) but no output ports.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in,
                 output_dir=None):
        """
        Abstract constructor of a Writing Module which needs the unique name identifier as input
        (more information: :class:`PynPoint.core.Processing.PypelineModule`). In addition one can
        specify a output directory where the module will save its results. If no output directory is
        given the Pypline default directory is used. Call this function in all __init__() functions
        inheriting from this class.

        :param name_in: The name of the Writing Module
        :type name_in: str
        :param output_dir: Directory where the results will be saved (Needs to be a directory,
                           raises error if not).
        :type output_dir: str
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
        Method which creates a new InputPort and append it to the internal InputPort dictionary.
        This function should be used by classes inheriting from Processing Module to make sure that
        only InputPort with unique tags are added. The new port can be used by: ::

             Port = self._m_input_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new input port.
        :type tag: str
        :return: The new InputPort
        :rtype: InputPort
        """

        tmp_port = InputPort(tag)
        if tag in self._m_input_ports:
            warnings.warn('Tag already used. Updating..')

        if self._m_data_base is not None:
            tmp_port.set_database_connection(self._m_data_base)

        self._m_input_ports[tag] = tmp_port

        return tmp_port

    def connect_database(self,
                         data_base_in):
        """
        Connects all ports in the internal input and output port dictionaries to the given database.
        This function is called by Pypeline and connects its DataStorage object to all module ports.

        :param data_base_in: The input database
        :type data_base_in: DataStorage
        :return: None
        """

        for port in self._m_input_ports.itervalues():
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    def get_all_input_tags(self):
        """
        Returns a list of all input tags

        :return: list of input tags
        :rtype: list
        """

        return self._m_input_ports.keys()

    @abstractmethod
    def run(self):
        pass


class ProcessingModule(PypelineModule):
    """
    The abstract class ProcessingModule is an interface for all processing steps in the pipeline
    which reads, processes and saves data. Hence they have reading and writing access to the central
    data base using a dictionary of output ports (self._m_output_ports) and a dictionary of input
    ports (self._m_input_ports).
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in):
        """
        Abstract constructor of a ProcessingModule which needs the unique name identifier as input
        (more information: :class:`PynPoint.core.Processing.PypelineModule`). Call this function in
        all __init__() functions inheriting from this class.

        :param name_in: The name of the Processing Module
        :type name_in: str
        """

        super(ProcessingModule, self).__init__(name_in)

        self._m_input_ports = {}
        self._m_output_ports = {}

    def add_input_port(self,
                       tag):
        """
        Method which creates a new InputPort and append it to the internal InputPort dictionary.
        This function should be used by classes inheriting from Processing Module to make sure that
        only InputPort with unique tags are added. The new port can be used by: ::

             Port = self._m_input_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new input port.
        :type tag: str
        :return: The new InputPort
        :rtype: InputPort
        """

        tmp_port = InputPort(tag)
        if tag in self._m_input_ports and tag != "hessian_res_mean" and \
           tag != "hessian_fake" and tag != "contrast_res_mean" and tag != "contrast_fake":
            warnings.warn('Tag '+tag+' already used. Updating..')

        if self._m_data_base is not None:
            tmp_port.set_database_connection(self._m_data_base)

        self._m_input_ports[tag] = tmp_port

        return tmp_port

    def add_output_port(self,
                        tag,
                        default_activation=True):
        """
        Method which creates a new OutputPort and append it to the internal OutputPort dictionary.
        This function should be used by classes inheriting from Processing Module to make sure that
        only OutputPort with unique tags are added. The new port can be used by: ::

             Port = self._m_output_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new output port.
        :type tag: str
        :param default_activation: Activation status of the Port after creation. Deactivated Ports
                                   will not save their results until the are activated.
        :type default_activation: bool
        :return: The new OutputPort
        :rtype: OutputPort
        """

        tmp_port = OutputPort(tag,
                              activate_init=default_activation)

        if tag in self._m_output_ports:
            warnings.warn('Tag already used. Updating..')

        if self._m_data_base is not None:
            tmp_port.set_database_connection(self._m_data_base)

        self._m_output_ports[tag] = tmp_port

        return tmp_port

    def connect_database(self,
                         data_base_in):
        """
        Connects all ports in the internal input and output port dictionaries to the given database.
        This function is called by Pypeline and connects its DataStorage object to all module ports.

        :param data_base_in: The input database
        :type data_base_in: DataStorage
        :return: None
        """

        for port in self._m_input_ports.itervalues():
            port.set_database_connection(data_base_in)
        for port in self._m_output_ports.itervalues():
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    def apply_function_to_line_in_time_multi_processing(self,
                                                        func,
                                                        image_in_port,
                                                        image_out_port,
                                                        func_args=None):
        """
        Applies a given function to all lines in time.
        :param func: The function to be applied
        :param image_in_port: Input Port where the data to be processed is located
        :param image_out_port: Input Port where the results will be stored
        :param func_args: addition arguments needed for the function (can be None)
        :return: None
        """

        # get first line in time
        init_line = image_in_port[:, 0, 0]
        length_of_processed_data = apply_function(init_line, func, func_args).shape[0]

        # we want to replace old values or create a new data set if True
        # if not we want to update the frames
        update = image_out_port.tag == image_in_port.tag
        if update and length_of_processed_data != image_in_port.get_shape()[0]:
            raise ValueError(
                "Input and output port have the same tag while %s is changing "
                "the length of the signal. Use different input and output ports "
                "instead. " % func)

        image_out_port.set_all(np.zeros((length_of_processed_data,
                                         image_in_port.get_shape()[1],
                                         image_in_port.get_shape()[2])),
                               data_dim=3,
                               keep_attributes=False)  # overwrite old existing attributes

        num_processors = self._m_config_port.get_attribute("CPU_COUNT")

        print "Database prepared. Starting analysis with " + str(num_processors) + " processes."

        line_processor = LineProcessingCapsule(image_in_port,
                                               image_out_port,
                                               num_processors,
                                               func,
                                               func_args,
                                               length_of_processed_data)
        line_processor.run()

    @staticmethod
    def apply_function_to_images(func,
                                 image_in_port,
                                 image_out_port,
                                 func_args=None,
                                 num_images_in_memory=100):
        """
        Often a algorithm is applied to all images of a 3D data stack. Hence we have implemented
        this function which applies a given function to all images of a data stack. The function
        needs a port which is linked to the input data, a port which is linked to the output place
        and the actual function. Since the input dataset might be larger than the available memory
        it is possible to set a maximum number of frames that is loaded into the memory.

        **Note** the function *func* is not allowed to change the shape of the images if the input
        and output port have the same tag and num_images_in_memory is not None.

        Have a look at the code of
        :class:`PynPoint.processing_modules.BadPixelCleaning.BadPixelCleaningSigmaFilterModule` for
        an **Example**.


        :raises: ValueError
        :param func: The function which is applied to all images.
                     It needs to have a definition similar to: ::

                         def some_image_function(image_in,
                                                 parameter1,
                                                 parameter2,
                                                 parameter3):

                             # some algorithm here
        :type func: function
        :param image_in_port: InputPort which is linked to the input data
        :type image_in_port: InputPort
        :param image_out_port: OutputPort which is linked to the result place
        :type image_out_port: OutputPort
        :param func_args: Additional arguments which are needed by the function *func*
        :type func_args: tuple
        :param num_images_in_memory: Maximum number of frames which will be loaded to the memory. If
                                     None all frames will be load at once. (This is probably the
                                     fastest but most memory expensive option)
        :type num_images_in_memory: int
        :return: None
        """

        number_of_images = image_in_port.get_shape()[0]

        if num_images_in_memory is None:
            num_images_in_memory = number_of_images

        # check if input and output Port have the same tag

        # we want to replace old values or create a new data set if True
        # if not we want to update the frames
        update = image_out_port.tag == image_in_port.tag

        i = 0
        first_time = True
        while i < number_of_images:
            print "Processing image " + str(i+1) + " of " + str(number_of_images) + " images..."
            if i + num_images_in_memory > number_of_images:
                j = number_of_images
            else:
                j = i + num_images_in_memory

            tmp_frames = image_in_port[i:j]
            tmp_res = []

            # process frames
            # check if additional arguments are given
            if func_args is None:
                for k in range(tmp_frames.shape[0]):
                    tmp_res.append(func(tmp_frames[k]))
            else:
                for k in range(tmp_frames.shape[0]):
                    tmp_res.append(func(tmp_frames[k], * func_args))

            if update:
                try:
                    if num_images_in_memory == number_of_images:
                        image_out_port.set_all(np.array(tmp_res),
                                               keep_attributes=True)
                    else:
                        image_out_port[i:j] = np.array(tmp_res)
                except TypeError:
                    raise ValueError("Input and output port have the same tag while %s is changing "
                                     "the image shape. This is only possible for "
                                     "num_images_in_memory == None. Change num_images_in_memory"
                                     "or choose different port tags." % func)
            elif first_time:
                # The first time we have to reset the eventually existing data
                image_out_port.set_all(np.array(tmp_res))
                first_time = False
            else:
                image_out_port.append(np.array(tmp_res))

            i = j

    def get_all_input_tags(self):
        """
        Returns a list of all input tags

        :return: list of input tags
        :rtype: list
        """

        return self._m_input_ports.keys()

    def get_all_output_tags(self):
        """
        Returns a list of all output tags

        :return: list of output tags
        :rtype: list
        """

        return self._m_output_ports.keys()

    @abstractmethod
    def run(self):
        pass


class ReadingModule(PypelineModule):
    """
    The abstract class ReadingModule is a interface for processing steps in the pipeline which do
    not use any information of the internal data storage. They only have writing access to the data
    base which makes them the perfect tool for loading data of a different file formats into the
    central data base. One can specify a directory on the hard drive where the input data for the
    module is located. If no input directory is given the default Pypline input directory is used.
    Reading modules have a dictionary of output ports (self._m_out_ports) but no input ports.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name_in,
                 input_dir=None):
        """
        Abstract constructor of a ReadingModule which needs the unique name identifier as input
        (more information: :class:`PynPoint.core.Processing.PypelineModule`). In addition one can
        specify a input directory where data is located which will be loaded by the module. If no
        directory is given the Pipeline default directory is used. Call this function in all
        __init__() functions inheriting from this class.

        :param name_in: The name of the Reading Module
        :type name_in: str
        :param input_dir: Directory where the input files are located (Needs to be a directory,
                          raises error if not)
        :type input_dir: str
        :return: None
        """

        super(ReadingModule, self).__init__(name_in)

        # If input_dir is None its location will be the Pypeline default input directory
        assert (os.path.isdir(str(input_dir))
                or input_dir is None), 'Error: Input directory for reading module does not exist ' \
                                       '- input requested: %s' % input_dir
        self.m_input_location = input_dir
        self._m_output_ports = {}

    def add_output_port(self,
                        tag,
                        default_activation=True):
        """
        Method which creates a new OutputPort and append it to the internal OutputPort dictionary.
        This function should be used by classes inheriting from ReadingModule to make sure that
        only OutputPort with unique tags are added. The new port can be used by: ::

             Port = self._m_output_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new output port.
        :type tag: str
        :param default_activation: Activation status of the Port after creation. Deactivated Ports
                                   will not save their results until the are activated.
        :type default_activation: bool
        :return: The new OutputPort
        :rtype: OutputPort
        """

        tmp_port = OutputPort(tag,
                              activate_init=default_activation)

        if tag in self._m_output_ports:
            warnings.warn('Tag already used. Updating..')

        if self._m_data_base is not None:
            tmp_port.set_database_connection(self._m_data_base)

        self._m_output_ports[tag] = tmp_port

        return tmp_port

    def connect_database(self,
                         data_base_in):
        """
        Connects all ports in the internal input and output port dictionaries to the given database.
        This function is called by Pypeline and connects its DataStorage object to all module ports.

        :param data_base_in: The input database
        :type data_base_in: DataStorage
        :return: None
        """

        for port in self._m_output_ports.itervalues():
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    def get_all_output_tags(self):
        """
        Returns a list of all output tags

        :return: list of output tags
        :rtype: list
        """

        return self._m_output_ports.keys()

    @abstractmethod
    def run(self):
        pass
