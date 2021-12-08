"""
Interfaces for pipeline modules.
"""

import math
import os
import time
import warnings

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional

import numpy as np

from typeguard import typechecked

from pynpoint.core.dataio import ConfigPort, DataStorage, InputPort, OutputPort
from pynpoint.util.module import progress, update_arguments
from pynpoint.util.multiline import LineProcessingCapsule
from pynpoint.util.multiproc import apply_function
from pynpoint.util.multistack import StackProcessingCapsule


class PypelineModule(metaclass=ABCMeta):
    """
    Abstract interface for the PypelineModule:

        * Reading module (:class:`pynpoint.core.processing.ReadingModule`)
        * Writing module (:class:`pynpoint.core.processing.WritingModule`)
        * Processing module (:class:`pynpoint.core.processing.ProcessingModule`)

    Each :class:`~pynpoint.core.processing.PypelineModule` has a name as a unique identifier in the
    :class:`~pynpoint.core.pypeline.Pypeline` and requires the ``connect_database`` and ``run``
    methods.
    """

    @typechecked
    def __init__(self,
                 name_in: str) -> None:
        """
        Abstract constructor of a :class:`~pynpoint.core.processing.PypelineModule`.

        Parameters
        ----------
        name_in : str
            The name of the :class:`~pynpoint.core.processing.PypelineModule`.

        Returns
        -------
        NoneType
            None
        """

        self._m_name = name_in
        self._m_data_base = None
        self._m_config_port = ConfigPort('config')

    @property
    @typechecked
    def name(self) -> str:
        """
        Returns the name of the :class:`~pynpoint.core.processing.PypelineModule`. This property
        makes sure that the internal module name can not be changed.

        Returns
        -------
        str
            The name of the :class:`~pynpoint.core.processing.PypelineModule`
        """

        return self._m_name

    @abstractmethod
    @typechecked
    def connect_database(self,
                         data_base_in: DataStorage) -> None:
        """
        Abstract interface for the function ``connect_database`` which is needed to connect a
        :class:`~pynpoint.core.dataio.Port` of a :class:`~pynpoint.core.processing.PypelineModule`
        with the :class:`~pynpoint.core.dataio.DataStorage`.

        Parameters
        ----------
        data_base_in : pynpoint.core.dataio.DataStorage
            The central database.
        """

    @abstractmethod
    @typechecked
    def run(self) -> None:
        """
        Abstract interface for the run method of :class:`~pynpoint.core.processing.PypelineModule`
        which inheres the actual algorithm behind the module.
        """


class ReadingModule(PypelineModule, metaclass=ABCMeta):
    """
    The abstract class ReadingModule is an interface for processing steps in the Pypeline which
    have only read access to the central data storage. One can specify a directory on the hard
    drive where the input data for the module is located. If no input directory is given then
    default Pypeline input directory is used. Reading modules have a dictionary of output ports
    (self._m_out_ports) but no input ports.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: Optional[str] = None) -> None:
        """
        Abstract constructor of ReadingModule which needs the unique name identifier as input
        (more information: :class:`pynpoint.core.processing.PypelineModule`). An input directory
        can be specified for the location of the data or else the Pypeline default directory is
        used. This function is called in all *__init__()* functions inheriting from this class.

        Parameters
        ----------
        name_in : str
            The name of the ReadingModule.
        input_dir : str
            Directory where the input files are located.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        assert (os.path.isdir(str(input_dir)) or input_dir is None), 'Input directory for ' \
            'reading module does not exist - input requested: %s.' % input_dir

        self.m_input_location = input_dir
        self._m_output_ports = {}

    @typechecked
    def add_output_port(self,
                        tag: str,
                        activation: bool = True) -> OutputPort:
        """
        Function which creates an OutputPort for a ReadingModule and appends it to the internal
        OutputPort dictionary. This function should be used by classes inheriting from
        ReadingModule to make sure that only output ports with unique tags are added. The new
        port can be used as: ::

             port = self._m_output_ports[tag]

        or by using the returned Port.

        Parameters
        ----------
        tag : str
            Tag of the new output port.
        activation : bool
            Activation status of the Port after creation. Deactivated ports will not save their
            results until they are activated.

        Returns
        -------
        pynpoint.core.dataio.OutputPort
            The new OutputPort for the ReadingModule.
        """

        port = OutputPort(tag, activate_init=activation)

        if tag in self._m_output_ports:
            warnings.warn(f'Tag \'{tag}\' of ReadingModule \'{self._m_name}\' is already used.')

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_output_ports[tag] = port

        return port

    @typechecked
    def connect_database(self,
                         data_base_in: DataStorage) -> None:
        """
        Function used by a ReadingModule to connect all ports in the internal input and output
        port dictionaries to the database. The function is called by Pypeline and connects the
        DataStorage object to all module ports.

        Parameters
        ----------
        data_base_in : pynpoint.core.dataio.DataStorage
            The central database.

        Returns
        -------
        NoneType
            None
        """

        for port in self._m_output_ports.values():
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    @typechecked
    def get_all_output_tags(self) -> List[str]:
        """
        Returns a list of all output tags to the ReadingModule.

        Returns
        -------
        list(str)
            List of output tags.
        """

        return list(self._m_output_ports.keys())

    @abstractmethod
    @typechecked
    def run(self) -> None:
        """
        Abstract interface for the run method of a ReadingModule which inheres the actual
        algorithm behind the module.
        """


class WritingModule(PypelineModule, metaclass=ABCMeta):
    """
    The abstract class WritingModule is an interface for processing steps in the pipeline which
    do not change the content of the internal DataStorage. They only have reading access to the
    central data base. WritingModules can be used to export data from the HDF5 database.
    WritingModules know the directory on the hard drive where the output of the module can be
    saved. If no output directory is given the default Pypeline output directory is used.
    WritingModules have a dictionary of input ports (self._m_input_ports) but no output ports.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 output_dir: Optional[str] = None) -> None:
        """
        Abstract constructor of a WritingModule which needs the unique name identifier as input
        (more information: :class:`pynpoint.core.processing.PypelineModule`). In addition one can
        specify a output directory where the module will save its results. If no output directory is
        given the Pypeline default directory is used. This function is called in all *__init__()*
        functions inheriting from this class.

        Parameters
        ----------
        name_in : str
            The name of the WritingModule.
        output_dir : str
            Directory where the results will be saved.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        assert (os.path.isdir(str(output_dir)) or output_dir is None), 'Output directory for ' \
            'writing module does not exist - input requested: %s.' % output_dir

        self.m_output_location = output_dir
        self._m_input_ports = {}

    @typechecked
    def add_input_port(self,
                       tag: str) -> InputPort:
        """
        Function which creates an InputPort for a WritingModule and appends it to the internal
        InputPort dictionary. This function should be used by classes inheriting from WritingModule
        to make sure that only input ports with unique tags are added. The new port can be used
        as: ::

             port = self._m_input_ports[tag]

        or by using the returned Port.

        Parameters
        ----------
        tag : str
            Tag of the new input port.

        Returns
        -------
        pynpoint.core.dataio.InputPort
            The new InputPort for the WritingModule.
        """

        port = InputPort(tag)

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_input_ports[tag] = port

        return port

    @typechecked
    def connect_database(self,
                         data_base_in: DataStorage) -> None:
        """
        Function used by a WritingModule to connect all ports in the internal input and output
        port dictionaries to the database. The function is called by Pypeline and connects the
        DataStorage object to all module ports.

        Parameters
        ----------
        data_base_in : pynpoint.core.dataio.DataStorage
            The central database.

        Returns
        -------
        NoneType
            None
        """

        for port in self._m_input_ports.values():
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    @typechecked
    def get_all_input_tags(self) -> List[str]:
        """
        Returns a list of all input tags to the WritingModule.

        Returns
        -------
        list(str)
            List of input tags.
        """

        return list(self._m_input_ports.keys())

    @abstractmethod
    @typechecked
    def run(self) -> None:
        """
        Abstract interface for the run method of a WritingModule which inheres the actual
        algorithm behind the module.
        """


class ProcessingModule(PypelineModule, metaclass=ABCMeta):
    """
    The abstract class ProcessingModule is an interface for all processing steps in the pipeline
    which read, process, and store data. Hence processing modules have read and write access to the
    central database through a dictionary of output ports (self._m_output_ports) and a dictionary
    of input ports (self._m_input_ports).
    """

    @typechecked
    def __init__(self,
                 name_in: str) -> None:
        """
        Abstract constructor of a ProcessingModule which needs the unique name identifier as input
        (more information: :class:`pynpoint.core.processing.PypelineModule`). Call this function in
        all __init__() functions inheriting from this class.

        Parameters
        ----------
        name_in : str
             The name of the ProcessingModule.
        """

        super().__init__(name_in)

        self._m_input_ports = {}
        self._m_output_ports = {}

    @typechecked
    def add_input_port(self,
                       tag: str) -> InputPort:
        """
        Function which creates an InputPort for a ProcessingModule and appends it to the internal
        InputPort dictionary. This function should be used by classes inheriting from
        ProcessingModule to make sure that only input ports with unique tags are added. The new
        port can be used as: ::

             port = self._m_input_ports[tag]

        or by using the returned Port.

        Parameters
        ----------
        tag : str
            Tag of the new input port.

        Returns
        -------
        pynpoint.core.dataio.InputPort
            The new InputPort for the ProcessingModule.
        """

        port = InputPort(tag)

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_input_ports[tag] = port

        return port

    @typechecked
    def add_output_port(self,
                        tag: str,
                        activation: bool = True) -> OutputPort:
        """
        Function which creates an :class:`~pynpoint.core.dataio.OutputPort` for a
        :class:`~pynpoint.core.processing.ProcessingModule` and appends it to the internal
        :class:`~pynpoint.core.dataio.OutputPort` dictionary. This function should be used by
        classes inheriting from :class:`~pynpoint.core.processing.ProcessingModule` to make sure
        that only output ports with unique tags are added. The new port can be used as:

        .. code-block:: python

             port = self._m_output_ports[tag]

        or by using the returned :class:`~pynpoint.core.dataio.Port`.

        Parameters
        ----------
        tag : str
            Tag of the new output port.
        activation : bool
            Activation status of the :class:`~pynpoint.core.dataio.Port` after creation.
            Deactivated ports will not save their results until they are activated.

        Returns
        -------
        pynpoint.core.dataio.OutputPort
            The new :class:`~pynpoint.core.dataio.OutputPort` for the
            :class:`~pynpoint.core.processing.ProcessingModule`.
        """

        port = OutputPort(tag, activate_init=activation)

        if tag in self._m_output_ports:
            warnings.warn(f'Tag \'{tag}\' of ProcessingModule \'{self._m_name}\' is already used.')

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_output_ports[tag] = port

        return port

    @typechecked
    def connect_database(self,
                         data_base_in: DataStorage) -> None:
        """
        Function used by a ProcessingModule to connect all ports in the internal input and output
        port dictionaries to the database. The function is called by Pypeline and connects the
        DataStorage object to all module ports.

        Parameters
        ----------
        data_base_in : pynpoint.core.dataio.DataStorage
            The central database.

        Returns
        -------
        NoneType
            None
        """

        for port in self._m_input_ports.values():
            port.set_database_connection(data_base_in)

        for port in self._m_output_ports.values():
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    @typechecked
    def apply_function_in_time(self,
                               func: Callable,
                               image_in_port: InputPort,
                               image_out_port: OutputPort,
                               func_args: Optional[tuple] = None) -> None:
        """
        Applies a function to all pixel lines in time.

        Parameters
        ----------
        func : function
            The input function.
        image_in_port : pynpoint.core.dataio.InputPort
            Input port which is linked to the input data.
        image_out_port : pynpoint.core.dataio.OutputPort
            Output port which is linked to the results.
        func_args : tuple, None
            Additional arguments which are required by the input function. Not used if set to None.

        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')

        init_line = image_in_port[:, 0, 0]

        im_shape = image_in_port.get_shape()

        size = apply_function(init_line, 0, func, func_args).shape[0]

        image_out_port.set_all(data=np.zeros((size, im_shape[1], im_shape[2])),
                               data_dim=3,
                               keep_attributes=False)

        image_in_port.close_port()
        image_out_port.close_port()

        capsule = LineProcessingCapsule(image_in_port=image_in_port,
                                        image_out_port=image_out_port,
                                        num_proc=cpu,
                                        function=func,
                                        function_args=func_args,
                                        data_length=size)

        capsule.run()

    @typechecked
    def apply_function_to_images(self,
                                 func: Callable[..., np.ndarray],
                                 image_in_port: InputPort,
                                 image_out_port: OutputPort,
                                 message: str,
                                 func_args: Optional[tuple] = None) -> None:
        """
        Function which applies a function to all images of an input port. Stacks of images are
        processed in parallel if the CPU and MEMORY attribute are set in the central configuration.
        The number of images per process is equal to the value of MEMORY divided by the value of
        CPU. Note that the function *func* is not allowed to change the shape of the images if the
        input and output port have the same tag and ``MEMORY`` is not set to None.

        Parameters
        ----------
        func : function
            The function which is applied to all images. Its definitions should be similar to::

                def function(image_in,
                             parameter1,
                             parameter2,
                             parameter3)

            The function must return a numpy array.
        image_in_port : pynpoint.core.dataio.InputPort
            Input port which is linked to the input data.
        image_out_port : pynpoint.core.dataio.OutputPort
            Output port which is linked to the results.
        message : str
            Progress message.
        func_args : tuple
            Additional arguments that are required by the input function.

        Returns
        -------
        NoneType
            None
        """

        memory = self._m_config_port.get_attribute('MEMORY')
        cpu = self._m_config_port.get_attribute('CPU')

        nimages = image_in_port.get_shape()[0]

        if memory == 0:
            memory = nimages

        if image_out_port.tag == image_in_port.tag:
            # load all images in the memory at once if the input and output tag are the
            # same or if the MEMORY attribute is set to None in the configuration file
            images = image_in_port.get_all()

            result = []

            start_time = time.time()

            for i in range(nimages):
                progress(i, nimages, message+'...', start_time)

                args = update_arguments(i, nimages, func_args)

                if args is None:
                    result.append(func(images[i, ], i))
                else:
                    result.append(func(images[i, ], i, *args))

            image_out_port.set_all(np.asarray(result), keep_attributes=True)

        elif cpu == 1:
            # process images one-by-one with a single process if CPU is set to 1
            start_time = time.time()

            for i in range(nimages):
                progress(i, nimages, message+'...', start_time)

                args = update_arguments(i, nimages, func_args)

                if args is None:
                    result = func(image_in_port[i, ], i)
                else:
                    result = func(image_in_port[i, ], i, *args)

                if result.ndim == 1:
                    image_out_port.append(result, data_dim=2)
                elif result.ndim == 2:
                    image_out_port.append(result, data_dim=3)

        else:
            # process images in parallel in stacks of MEMORY/CPU images
            print(message, end='')

            args = update_arguments(0, nimages, func_args)

            result = apply_function(image_in_port[0, :, :], 0, func, args)

            result_shape = result.shape

            out_shape = [nimages]
            for item in result_shape:
                out_shape.append(item)

            image_out_port.set_all(data=np.zeros(out_shape),
                                   data_dim=len(result_shape)+1,
                                   keep_attributes=False)

            image_in_port.close_port()
            image_out_port.close_port()

            capsule = StackProcessingCapsule(image_in_port=image_in_port,
                                             image_out_port=image_out_port,
                                             num_proc=cpu,
                                             function=func,
                                             function_args=func_args,
                                             stack_size=math.ceil(memory/cpu),
                                             result_shape=result_shape,
                                             nimages=nimages)

            capsule.run()

            print(' [DONE]')

    @typechecked
    def get_all_input_tags(self) -> List[str]:
        """
        Returns a list of all input tags to the ProcessingModule.

        Returns
        -------
        list(str)
            List of input tags.
        """

        return list(self._m_input_ports.keys())

    @typechecked
    def get_all_output_tags(self) -> List[str]:
        """
        Returns a list of all output tags to the ProcessingModule.

        Returns
        -------
        list(str)
            List of output tags.
        """

        return list(self._m_output_ports.keys())

    @abstractmethod
    @typechecked
    def run(self) -> None:
        """
        Abstract interface for the run method of a
        :class:`~pynpoint.core.processing.ProcessingModule` which inheres the actual
        algorithm behind the module.
        """
