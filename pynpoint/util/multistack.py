"""
Utilities for multiprocessing of stacks of images.
"""

import sys

import numpy as np

from pynpoint.util.module import update_arguments
from pynpoint.util.multiproc import TaskInput, TaskResult, TaskCreator, TaskProcessor, \
                                    MultiprocessingCapsule, apply_function


class StackReader(TaskCreator):
    """
    Reader of task inputs for :class:`~pynpoint.util.multistack.StackProcessingCapsule`.
    Reads continuously stacks of images of a dataset and puts them into a task queue.
    """

    def __init__(self,
                 data_port_in,
                 tasks_queue_in,
                 data_mutex_in,
                 num_proc,
                 stack_size,
                 result_shape):
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
        result_shape : tuple(int, int, int)
            Shape of the array with the output results (usually a stack of images).

        Returns
        -------
        NoneType
            None
        """

        super(StackReader, self).__init__(data_port_in, tasks_queue_in, data_mutex_in, num_proc)

        self.m_stack_size = stack_size
        self.m_result_shape = result_shape

    def run(self):
        """
        Returns
        -------
        NoneType
            None
        """

        nimages = self.m_data_in_port.get_shape()[0]

        i = 0
        while i < nimages:
            j = min((i + self.m_stack_size), nimages)

            # lock mutex and read data
            with self.m_data_mutex:
                # read images from i to j
                tmp_data = self.m_data_in_port[i:j, ]

            # first dimensiosn (start, stop, step)
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

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in,
                 function,
                 function_args,
                 nimages):
        """
        Parameters
        ----------
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            Tasks queue.
        result_queue_in : multiprocessing.queues.JoinableQueue
            Results queue.
        function : function
            Input function that is applied to the images.
        function_args : tuple, None, optional
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

    def run_job(self,
                tmp_task):
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

    def __init__(self,
                 image_in_port,
                 image_out_port,
                 num_proc,
                 function,
                 function_args,
                 stack_size,
                 result_shape,
                 nimages):
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
        result_shape : tuple(int, int, int)
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

    def create_processors(self):
        """
        Returns
        -------
        list(pynpoint.util.multiproc.StackTaskProcessor, )
            List with instances of :class:`~pynpoint.util.multiproc.StackTaskProcessor`
        """

        processors = []

        for _ in range(self.m_num_proc):

            processors.append(StackTaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                                 result_queue_in=self.m_result_queue,
                                                 function=self.m_function,
                                                 function_args=self.m_function_args,
                                                 nimages=self.m_nimages))

        return processors

    def init_creator(self,
                     image_in_port):
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

        return StackReader(image_in_port,
                           self.m_tasks_queue,
                           self.m_data_mutex,
                           self.m_num_proc,
                           self.m_stack_size,
                           self.m_result_shape)
