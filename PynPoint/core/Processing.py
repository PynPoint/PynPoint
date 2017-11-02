"""
Different interfaces for Pypeline Modules.
"""

# external modules
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import multiprocessing

from PynPoint.core.DataIO import OutputPort, InputPort

import warnings


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
    not change the content of the internal DataStorage. They only have reading access to the central
    data base. WritingModules can be used to export save or plot data from the .hdf5 data base by
    using a different file format. WritingModules know directory on the hard drive where the output
    of the module can be saved. If no output directory is given the default Pypline output directory
    is used. WritingModules have a dictionary of inputports (self._m_input_ports) but no output ports.
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


class TaskData(object):
    def __init__(self,
                 data_array,
                 position):
        self.m_data_array = data_array
        self.m_position = position

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
        if tag in self._m_input_ports:
            warnings.warn('Tag already used. Updating..')

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

        self._m_data_base = data_base_in

    @staticmethod
    def apply_function_to_line_in_time_multi_processing(func,
                                                        image_in_port,
                                                        image_out_port,
                                                        func_args=None,
                                                        num_rows_in_memory=40):

        def apply_function(tmp_line_in):
            # process line
            # check if additional arguments are given
            if func_args is None:
                return np.array(func(tmp_line_in))
            else:
                return np.array(func(tmp_line_in, *func_args))

        # get first line in time
        init_line = image_in_port[:, 0, 0]
        length_of_processed_data = apply_function(init_line).shape[0]

        # we want to replace old values or create a new data set if True
        # if not we want to update the frames
        update = image_out_port.tag == image_in_port.tag
        if update and length_of_processed_data != image_in_port.get_shape()[0]:
            raise ValueError(
                "Input and output port have the same tag while %s is changing "
                "the length of the signal. Use different input and output ports "
                "instead. " % func)

        class Reader(multiprocessing.Process):
            def __init__(self,
                         data_in_port_in,
                         data_mutex_in,
                         total_number_of_rows,
                         tasks_queue_in,
                         number_of_processors,
                         num_rows_in_memory_in):
                multiprocessing.Process.__init__(self)
                self.m_total_number_of_rows = total_number_of_rows
                self.m_data_mutex = data_mutex_in
                self.m_task_queue = tasks_queue_in
                self.m_data_in_port = data_in_port_in
                self.m_number_of_processors = number_of_processors
                self.m_number_of_rows_in_memory = num_rows_in_memory_in

            def run(self):

                i = 0
                while i < self.m_total_number_of_rows:
                    # read rows from i to j
                    j = min((i + self.m_number_of_rows_in_memory), self.m_total_number_of_rows)

                    # lock Mutex and read data
                    with self.m_data_mutex:
                        print "reading lines from " + str(i) + " to " + str(j)
                        tmp_data = self.m_data_in_port[:, i:j, :]

                    self.m_task_queue.put(TaskData(tmp_data, (i, j)))
                    i = j

                for i in range(self.m_number_of_processors - 1):
                    # poison pills
                    self.m_task_queue.put(1)

                # Final poison pill
                self.m_task_queue.put(None)

                return

        class LineProcessor(multiprocessing.Process):

            def __init__(self,
                         tasks_queue_in,
                         result_queue_in):

                multiprocessing.Process.__init__(self)
                self.m_task_queue = tasks_queue_in
                self.m_result_queue = result_queue_in

            def run(self):
                proc_name = self.name

                while True:
                    next_task = self.m_task_queue.get()

                    if next_task is 1:
                        # Poison pill means shutdown
                        print '%s: Exiting' % proc_name
                        self.m_task_queue.task_done()
                        break

                    if next_task is None:
                        # got final Poison pill
                        self.m_result_queue.put(None)  # shut down writer process

                        print '%s: Exiting' % proc_name
                        self.m_task_queue.task_done()
                        break

                    print "Process " + proc_name + " got data for row " + str(
                        next_task.m_position) + " and starts processing..."

                    result_arr = np.zeros((length_of_processed_data,
                                          next_task.m_data_array.shape[1],
                                          next_task.m_data_array.shape[2]))
                    for i in range(next_task.m_data_array.shape[1]):
                        for j in range(next_task.m_data_array.shape[2]):
                            tmp_line = next_task.m_data_array[:, i, j]
                            result_arr[:, i, j] = apply_function(tmp_line)

                    result = TaskData(result_arr,
                                      next_task.m_position)

                    self.m_task_queue.task_done()

                    self.m_result_queue.put(result)
                    print "Process " + proc_name + " finished processing!"

                return

        class Writer(multiprocessing.Process):

            def __init__(self,
                         result_queue_in,
                         data_out_port_in,
                         data_mutex_in):
                multiprocessing.Process.__init__(self)
                self.m_result_queue = result_queue_in
                self.m_data_mutex = data_mutex_in
                self.m_data_out_port = data_out_port_in

            def run(self):

                while True:
                    next_result = self.m_result_queue.get()

                    if next_result is None:
                        print "shutting down writer..."
                        self.m_result_queue.task_done()
                        break

                    print "Start writing row " + str(next_result.m_position)

                    with self.m_data_mutex:
                        self.m_data_out_port[:,
                                             next_result.m_position[0] : next_result.m_position[1],
                                             :] = next_result.m_data_array
                    self.m_result_queue.task_done()

        print "Preparing database for analysis ..."

        # TODO: try to create without stalling huge memory
        image_out_port.set_all(np.zeros((length_of_processed_data,
                                        image_in_port.get_shape()[1],
                                        image_in_port.get_shape()[2])),
                               data_dim=3,
                               keep_attributes=False)  # overwrite old existing attributes

        num_processors = multiprocessing.cpu_count()

        number_of_rows = image_in_port.get_shape()[1]
        if num_rows_in_memory is None:
            num_rows_in_memory = int(np.ceil(image_in_port.get_shape()[1]/float(num_processors)))
        else:
            num_rows_in_memory = int(np.ceil(num_rows_in_memory/float(num_processors)))

        print "Database prepared. Starting analysis with " + str(num_processors) + " processes."

        # Establish communication queues

        # buffer twice the data as processes are available
        tasks_queue = multiprocessing.JoinableQueue(maxsize=num_processors)
        result_queue = multiprocessing.JoinableQueue(maxsize=num_processors)

        # data base mutex
        data_mutex = multiprocessing.Lock()

        # create reader
        reader = Reader(data_in_port_in=image_in_port,
                        data_mutex_in=data_mutex,
                        total_number_of_rows=number_of_rows,
                        tasks_queue_in=tasks_queue,
                        number_of_processors=num_processors,
                        num_rows_in_memory_in=num_rows_in_memory)

        # Start consumers
        line_processors = [LineProcessor(tasks_queue_in=tasks_queue,
                                         result_queue_in=result_queue)
                           for i in xrange(num_processors)]

        # create writer
        writer = Writer(result_queue_in=result_queue,
                        data_out_port_in=image_out_port,
                        data_mutex_in=data_mutex)

        # start all processes
        reader.start()

        for processor in line_processors:
            processor.start()

        writer.start()

        # Wait for all of the tasks to finish
        tasks_queue.join()
        result_queue.join()

        for processor in line_processors:
            processor.join()

        writer.join()
        reader.join()

    @staticmethod
    def apply_function_to_line_in_time(func,
                                       image_in_port,
                                       image_out_port,
                                       func_args=None):
        """

        :param func:
        :param image_in_port:
        :type image_in_port: InputPort
        :param image_out_port:
        :type image_out_port: OutputPort
        :param func_args:
        :return:
        """

        # TODO test and documentation

        number_of_lines_i = image_in_port.get_shape()[1]
        number_of_lines_j = image_in_port.get_shape()[2]

        def apply_function(tmp_line_in):
            # process line
            # check if additional arguments are given
            if func_args is None:
                return np.array(func(tmp_line_in))
            else:
                return np.array(func(tmp_line_in, *func_args))

        # get first line in time
        init_line = image_in_port[:, 0, 0]
        length_of_processed_data = apply_function(init_line).shape[0]

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

        for i in range(0, number_of_lines_i):
            print "processed line nr. " + str(i+1) + " of " + str(number_of_lines_i) + " lines."
            for j in range(0, number_of_lines_j):

                tmp_line = image_in_port[:, i, j]
                tmp_res = apply_function(tmp_line)

                if tmp_res.shape[0] != length_of_processed_data:
                    # The processed line has the wrong size -> raise error
                    raise ValueError(
                        "The function %s produces results with different length. This is not "
                        "supported." % func)

                else:
                    image_out_port[:, i, j] = tmp_res

        return

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
        :param func_args: Additional arguments which are needed by the  function *func*
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
            print "processed image " + str(i+1) + " of " + str(number_of_images) + " images"
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

        for port in self._m_output_ports.itervalues():
            port.set_database_connection(data_base_in)

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
