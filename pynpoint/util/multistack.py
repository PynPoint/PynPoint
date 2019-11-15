"""
Utilities for multiprocessing of stacks of images.
"""

import sys
import multiprocessing

from typing import Union, List, Callable

import numpy as np

from typeguard import typechecked

from pynpoint.core.dataio import InputPort, OutputPort
from pynpoint.util.module import update_arguments
from pynpoint.util.multiproc import TaskInput, TaskResult, TaskCreator, TaskProcessor, \
                                    MultiprocessingCapsule, apply_function


class StackReader(TaskCreator):
    """
    Reader of task inputs for :class:`~pynpoint.util.multistack.StackProcessingCapsule`.
    Reads continuously stacks of images of a dataset and puts them into a task queue.
    """

    @typechecked
    def __init__(self,
                 data_port_in: InputPort,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 data_mutex_in: multiprocessing.Lock,
                 num_proc: np.int64,
                 stack_size: int,
                 result_shape: tuple) -> None:
        """
        Parameters
        ----------
        data_port_in : pynpoint.core.dataio.InputPort
            Input port.
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            Tasks queue.
        data_mutex_in : multiprocessing.synchronize.Lock
            A mutex shared with the writer to ensure that no read and write operations happen at
            the same time.
        num_proc : int
            Number of processors.
        stack_size: int
            Number of images per stack.
        result_shape : tuple(int, )
            Shape of the array with the output results (usually a stack of images).

        Returns
        -------
        NoneType
            None
        """

        super(StackReader, self).__init__(data_port_in, tasks_queue_in, data_mutex_in, num_proc)

        self.m_stack_size = stack_size
        self.m_result_shape = result_shape

    @typechecked
    def run(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        with self.m_data_mutex:
            self.m_data_in_port._check_status_and_activate()
            nimages = self.m_data_in_port.get_shape()[0]
            self.m_data_in_port.close_port()

        i = 0
        while i < nimages:
            j = min((i + self.m_stack_size), nimages)

            # lock mutex and read data
            with self.m_data_mutex:
                self.m_data_in_port._check_status_and_activate()
                tmp_data = self.m_data_in_port[i:j, ]  # read images from i to j
                self.m_data_in_port.close_port()

            # first dimension (start, stop, step)
            stack_slice = [(i, j, None)]

            # additional dimensions
            for _ in self.m_result_shape:
                stack_slice.append((None, None, None))

            param = (self.m_result_shape, tuple(stack_slice))
            self.m_task_queue.put(TaskInput(tmp_data, param))

            i = j

        self.create_poison_pills()


class StackTaskProcessor(TaskProcessor):
    """
    Processor of task inputs for :class:`~pynpoint.util.multistack.StackProcessingCapsule`. A
    processor applies a function on a stack of images.
    """

    @typechecked
    def __init__(self,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 result_queue_in: multiprocessing.JoinableQueue,
                 function: Callable,
                 function_args: Union[tuple, None],
                 nimages: int) -> None:
        """
        Parameters
        ----------
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            Tasks queue.
        result_queue_in : multiprocessing.queues.JoinableQueue
            Results queue.
        function : function
            Input function that is applied to the images.
        function_args : tuple, None
            Function arguments.
        nimages : int
            Total number of images.

        Returns
        -------
        NoneType
            None
        """

        super(StackTaskProcessor, self).__init__(tasks_queue_in, result_queue_in)

        self.m_function = function
        self.m_function_args = function_args
        self.m_nimages = nimages

    @typechecked
    def run_job(self,
                tmp_task: TaskInput) -> TaskResult:
        """
        Parameters
        ----------
        tmp_task : pynpoint.util.multiproc.TaskInput
            Task input with the subsets of images and the job parameters.

        Returns
        -------
        pynpoint.util.multiproc.TaskResult
            Task result.
        """

        result_nimages = tmp_task.m_input_data.shape[0]
        result_shape = tmp_task.m_job_parameter[0]

        # first dimension
        full_shape = [result_nimages]

        # additional dimensions
        for item in result_shape:
            full_shape.append(item)

        result_arr = np.zeros(full_shape)

        for i in range(result_nimages):
            # job parameter contains (result_shape, tuple(stack_slice))
            index = tmp_task.m_job_parameter[1][0][0] + i

            args = update_arguments(index, self.m_nimages, self.m_function_args)

            result_arr[i, ] = apply_function(tmp_data=tmp_task.m_input_data[i, ],
                                             func=self.m_function,
                                             func_args=args)

        sys.stdout.write('.')
        sys.stdout.flush()

        return TaskResult(result_arr, tmp_task.m_job_parameter[1])


class StackProcessingCapsule(MultiprocessingCapsule):
    """
    Capsule for parallel processing of stacks of images with the poison pill pattern. A function
    is applied in parallel to each stack of images.
    """

    @typechecked
    def __init__(self,
                 image_in_port: InputPort,
                 image_out_port: OutputPort,
                 num_proc: np.int64,
                 function: Callable,
                 function_args: Union[tuple, None],
                 stack_size: int,
                 result_shape: tuple,
                 nimages: int) -> None:
        """
        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort
            Input port.
        image_out_port : pynpoint.core.dataio.OutputPort
            Output port.
        num_proc : int
            Number of processors.
        function : function
            Input function.
        function_args : tuple, None
            Function arguments.
        stack_size: int
            Number of images per stack.
        result_shape : tuple(int, )
            Shape of the array with output results (usually a stack of images).
        nimages : int
            Total number of images.

        Returns
        -------
        NoneType
            None
        """

        self.m_function = function
        self.m_function_args = function_args
        self.m_stack_size = stack_size
        self.m_result_shape = result_shape
        self.m_nimages = nimages

        super(StackProcessingCapsule, self).__init__(image_in_port, image_out_port, num_proc)

    @typechecked
    def create_processors(self) -> List[StackTaskProcessor]:
        """
        Returns
        -------
        list(pynpoint.util.multiproc.StackTaskProcessor, )
            List with instances of :class:`~pynpoint.util.multiproc.StackTaskProcessor`.
        """

        processors = []

        for _ in range(self.m_num_proc):

            processors.append(StackTaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                                 result_queue_in=self.m_result_queue,
                                                 function=self.m_function,
                                                 function_args=self.m_function_args,
                                                 nimages=self.m_nimages))

        return processors

    @typechecked
    def init_creator(self,
                     image_in_port: InputPort) -> StackReader:
        """
        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort
            Input port from where the subsets of images are read.

        Returns
        -------
        pynpoint.util.multistack.StackReader
            Reader of stacks of images.
        """

        return StackReader(data_port_in=image_in_port,
                           tasks_queue_in=self.m_tasks_queue,
                           data_mutex_in=self.m_data_mutex,
                           num_proc=self.m_num_proc,
                           stack_size=self.m_stack_size,
                           result_shape=self.m_result_shape)
