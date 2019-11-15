"""
Utilities for multiprocessing of lines in time with the poison pill pattern.
"""

import multiprocessing

from typing import Union, List, Callable

import numpy as np

from typeguard import typechecked

from pynpoint.core.dataio import InputPort, OutputPort
from pynpoint.util.multiproc import TaskInput, TaskResult, TaskCreator, TaskProcessor, \
                                    MultiprocessingCapsule, apply_function


class LineReader(TaskCreator):
    """
    Reader of task inputs for :class:`~pynpoint.util.multiline.LineProcessingCapsule`. Continuously
    read all rows of a dataset and puts them into a task queue.
    """

    @typechecked
    def __init__(self,
                 data_port_in: InputPort,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 data_mutex_in: multiprocessing.Lock,
                 num_proc: np.int64,
                 data_length: int) -> None:
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
        data_length : int
            Length of the processed data.

        Returns
        -------
        NoneType
            None
        """

        super(LineReader, self).__init__(data_port_in, tasks_queue_in, data_mutex_in, num_proc)

        self.m_data_length = data_length

    @typechecked
    def run(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        n_rows = self.m_data_in_port.get_shape()[1]
        row_length = int(np.ceil(self.m_data_in_port.get_shape()[1]/float(self.m_num_proc)))

        i = 0
        while i < n_rows:
            j = min((i + row_length), n_rows)

            # lock mutex and read data
            with self.m_data_mutex:
                self.m_data_in_port._check_status_and_activate()
                tmp_data = self.m_data_in_port[:, i:j, :]  # read rows from i to j
                self.m_data_in_port.close_port()

            param = (self.m_data_length, ((None, None, None), (i, j, None), (None, None, None)))
            self.m_task_queue.put(TaskInput(tmp_data, param))

            i = j

        self.create_poison_pills()


class LineTaskProcessor(TaskProcessor):
    """
    Processor of task inputs for :class:`~pynpoint.util.multiline.LineProcessingCapsule`. A
    processor applies a function on a row of lines in time.
    """

    @typechecked
    def __init__(self,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 result_queue_in: multiprocessing.JoinableQueue,
                 function: Callable,
                 function_args: Union[tuple, None]) -> None:
        """
        Parameters
        ----------
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            Tasks queue.
        result_queue_in : multiprocessing.queues.JoinableQueue
            Results queue.
        function : function
            Input function.
        function_args : tuple, None
            Optional function arguments.

        Returns
        -------
        NoneType
            None
        """

        super(LineTaskProcessor, self).__init__(tasks_queue_in, result_queue_in)

        self.m_function = function
        self.m_function_args = function_args

    @typechecked
    def run_job(self,
                tmp_task: TaskInput) -> TaskResult:
        """
        Parameters
        ----------
        tmp_task : pynpoint.util.multiproc.TaskInput
            Task input with the subsets of lines and the job parameters.

        Returns
        -------
        pynpoint.util.multiproc.TaskResult
            Task result.
        """

        result_arr = np.zeros((tmp_task.m_job_parameter[0],
                               tmp_task.m_input_data.shape[1],
                               tmp_task.m_input_data.shape[2]))

        for i in range(tmp_task.m_input_data.shape[1]):
            for j in range(tmp_task.m_input_data.shape[2]):
                result_arr[:, i, j] = apply_function(tmp_data=tmp_task.m_input_data[:, i, j],
                                                     func=self.m_function,
                                                     func_args=self.m_function_args)

        return TaskResult(result_arr, tmp_task.m_job_parameter[1])


class LineProcessingCapsule(MultiprocessingCapsule):
    """
    Capsule for parallel processing of lines in time with the poison pill pattern. A function is
    applied in parallel to each line in time, for example as in
    :class:`~pynpoint.processing.timedenoising.WaveletTimeDenoisingModule`.
    """

    @typechecked
    def __init__(self,
                 image_in_port: InputPort,
                 image_out_port: OutputPort,
                 num_proc: np.int64,
                 function: Callable,
                 function_args: Union[tuple, None],
                 data_length: int) -> None:
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
            Input function that is applied to the lines.
        function_args : tuple, None, optional
            Function arguments.
        data_length : int
            Length of the processed data.

        Returns
        -------
        NoneType
            None
        """

        self.m_function = function
        self.m_function_args = function_args
        self.m_data_length = data_length

        super(LineProcessingCapsule, self).__init__(image_in_port, image_out_port, num_proc)

    @typechecked
    def create_processors(self) -> List[LineTaskProcessor]:
        """
        Returns
        -------
        list(pynpoint.util.multiproc.LineTaskProcessor, )
            List with instances of :class:`~pynpoint.util.multiproc.LineTaskProcessor`
        """

        processors = []

        for _ in range(self.m_num_proc):

            processors.append(LineTaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                                result_queue_in=self.m_result_queue,
                                                function=self.m_function,
                                                function_args=self.m_function_args))

        return processors

    @typechecked
    def init_creator(self,
                     image_in_port: InputPort) -> LineReader:
        """
        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort
            Input port from where the subsets of lines are read.

        Returns
        -------
        pynpoint.util.multiline.LineReader
            Line reader object.
        """

        return LineReader(image_in_port,
                          self.m_tasks_queue,
                          self.m_data_mutex,
                          self.m_num_proc,
                          self.m_data_length)
