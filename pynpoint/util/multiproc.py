"""
Abstract interfaces for multiprocessing applications with the poison pill pattern.
"""

import multiprocessing

from typing import Union, Callable
from abc import ABCMeta, abstractmethod

import numpy as np

from typeguard import typechecked

from pynpoint.core.dataio import InputPort, OutputPort


class TaskInput:
    """
    Class for tasks that are processed by the :class:`~pynpoint.util.multiproc.TaskProcessor`.
    """

    @typechecked
    def __init__(self,
                 input_data: Union[np.ndarray, np.int64],
                 job_parameter: tuple) -> None:
        """
        Parameters
        ----------
        input_data : int, float, numpy.ndarray
            Input data for by the :class:`~pynpoint.util.multiproc.TaskProcessor`.
        job_parameter : tuple
            Additional data or parameters.

        Returns
        -------
        NoneType
            None
        """

        self.m_input_data = input_data
        self.m_job_parameter = job_parameter


class TaskResult:
    """
    Class for results that can be stored by the :class:`~pynpoint.util.multiproc.TaskWriter`.
    """

    @typechecked
    def __init__(self,
                 data_array: np.ndarray,
                 position: tuple) -> None:
        """
        Parameters
        ----------
        data_array : numpy.ndarray
            Array with the results for a given position.
        position : tuple(tuple(int, int, int), tuple(int, int, int), tuple(int, int, int))
             The position where the results will be stored.

        Returns
        -------
        NoneType
            None
        """

        self.m_data_array = data_array
        self.m_position = position


class TaskCreator(multiprocessing.Process, metaclass=ABCMeta):
    """
    Abstract interface for :class:`~pynpoint.util.multiproc.TaskCreator` classes. A
    :class:`~pynpoint.util.multiproc.TaskCreator` creates instances of
    :class:`~pynpoint.util.multiproc.TaskInput`, which will be processed by the
    :class:`~pynpoint.util.multiproc.TaskProcessor`, and appends them to the central task
    queue. In general there is only one :class:`~pynpoint.util.multiproc.TaskCreator` running
    for a poison pill multiprocessing application. A :class:`~pynpoint.util.multiproc.TaskCreator`
    communicates with to the :class:`~pynpoint.util.multiproc.TaskWriter` in order to avoid
    simultaneously access to the central database.
    """

    @typechecked
    def __init__(self,
                 data_port_in: Union[InputPort, None],
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 data_mutex_in: Union[multiprocessing.Lock, None],
                 num_proc: np.int64) -> None:
        """
        Parameters
        ----------
        data_port_in : pynpoint.core.dataio.InputPort, None
            An input port which links to the data that has to be processed.
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            The central task queue.
        data_mutex_in : multiprocessing.synchronize.Lock, None
            A mutex shared with the writer to ensure that no read and write operations happen at
            the same time.
        num_proc : int
            Maximum number of instances of :class:`~pynpoint.util.multiproc.TaskProcessor` that run
            simultaneously.

        Returns
        -------
        NoneType
            None
        """

        multiprocessing.Process.__init__(self)

        self.m_data_in_port = data_port_in
        self.m_task_queue = tasks_queue_in
        self.m_data_mutex = data_mutex_in
        self.m_num_proc = num_proc

    @typechecked
    def create_poison_pills(self) -> None:
        """
        Creates poison pills for the :class:`~pynpoint.util.multiproc.TaskProcessor` and
        :class:`~pynpoint.util.multiproc.TaskWriter`. A process will shut down if it receives a
        poison pill as a new task. This method should be executed at the end of the
        :func:`~pynpoint.util.multiproc.TaskCreator.run` method.

        Returns
        -------
        NoneType
            None
        """

        for _ in range(self.m_num_proc-1):
            # poison pills
            self.m_task_queue.put(1)

        # final poison pill
        self.m_task_queue.put(None)

    @abstractmethod
    @typechecked
    def run(self) -> None:
        """
        Creates objects of the :class:`~pynpoint.util.multiproc.TaskInput` until all tasks are
        placed in the task queue.

        Returns
        -------
        NoneType
            None
        """


class TaskProcessor(multiprocessing.Process, metaclass=ABCMeta):
    """
    Abstract interface for :class:`~pynpoint.util.multiproc.TaskProcessor` classes. The number of
    instances of :class:`~pynpoint.util.multiproc.TaskProcessor` that run simultaneously in a
    poison pill multiprocessing application can be set with ``CPU`` parameter in the central
    configuration file. A :class:`~pynpoint.util.multiproc.TaskProcessor` takes tasks from a task
    queue, processes the task, and stores the results back into a result queue. The process will
    shut down if the next task is a poison pill. The order in which process finish is not fixed.
    """

    @typechecked
    def __init__(self,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 result_queue_in: multiprocessing.JoinableQueue) -> None:
        """
        Parameters
        ----------
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            The input task queue with instances of :class:`~pynpoint.util.multiproc.TaskInput`.
        result_queue_in : multiprocessing.queues.JoinableQueue
            The result task queue with instances of :class:`~pynpoint.util.multiproc.TaskResult`.

        Returns
        -------
        NoneType
            None
        """

        multiprocessing.Process.__init__(self)

        self.m_task_queue = tasks_queue_in
        self.m_result_queue = result_queue_in

    @typechecked
    def check_poison_pill(self,
                          next_task: Union[TaskInput, int, None]) -> bool:
        """
        Function to check if the next task is a poison pill.

        Parameters
        ----------
        next_task : int, None, pynpoint.util.multiproc.TaskInput
            The next task.

        Returns
        -------
        bool
            True if the next task is a poison pill, False otherwise.
        """

        if next_task == 1:
            # poison pill
            poison_pill = True

            self.m_task_queue.task_done()

        elif next_task is None:
            # final poison pill
            poison_pill = True

            # shut down writer process
            self.m_result_queue.put(None)
            self.m_task_queue.task_done()

        else:
            # no poison pill
            poison_pill = False

        return poison_pill

    @typechecked
    def run(self) -> None:
        """
        Run method to start the :class:`~pynpoint.util.multiproc.TaskProcessor`. The run method
        will continue to process tasks from the input task queue until it receives a poison pill.

        Returns
        -------
        NoneType
            None
        """

        while True:
            next_task = self.m_task_queue.get()

            if self.check_poison_pill(next_task):
                break

            result = self.run_job(next_task)

            self.m_task_queue.task_done()
            self.m_result_queue.put(result)

    @abstractmethod
    @typechecked
    def run_job(self,
                tmp_task: TaskInput) -> None:
        """
        Abstract interface for the :func:`~pynpoint.util.multiproc.TaskProcessor.run_job` method
        which is called from the :func:`~pynpoint.util.multiproc.TaskProcessor.run` method for each
        task individually.

        Parameters
        ----------
        tmp_task : pynpoint.util.multiproc.TaskInput
            Input task.

        Returns
        -------
        NoneType
            None
        """


class TaskWriter(multiprocessing.Process):
    """
    The :class:`~pynpoint.util.multiproc.TaskWriter` receives results from the result queue, which
    have been computed by a :class:`~pynpoint.util.multiproc.TaskProcessor`, and stores the results
    in the central database. The position parameter of the
    :class:`~pynpoint.util.multiproc.TaskResult` is used to slice the result to the correct
    position in the complete output dataset.
    """

    @typechecked
    def __init__(self,
                 result_queue_in: multiprocessing.JoinableQueue,
                 data_out_port_in: Union[OutputPort, None],
                 data_mutex_in: multiprocessing.Lock) -> None:
        """
        Parameters
        ----------
        result_queue_in : multiprocessing.queues.JoinableQueue
            The result queue.
        data_out_port_in : pynpoint.core.dataio.OutputPort, None
            The output port where the results will be stored.
        data_mutex_in : multiprocessing.synchronize.Lock
            A mutex that is shared with the :class:`~pynpoint.util.multiproc.TaskWriter` which
            ensures that read and write operations to the database do not occur simultaneously.

        Returns
        -------
        NoneType
            None
        """

        multiprocessing.Process.__init__(self)

        self.m_result_queue = result_queue_in
        self.m_data_mutex = data_mutex_in
        self.m_data_out_port = data_out_port_in

    @typechecked
    def check_poison_pill(self,
                          next_result: Union[TaskResult, None]) -> int:
        """
        Function to check if the next result is a poison pill.

        Parameters
        ----------
        next_result : None, pynpoint.util.multiproc.TaskResult
            The next result.

        Returns
        -------
        int
            0 -> no poison pill, 1 -> poison pill, 2 -> poison pill but still results in the
            queue (rare error case).
        """

        if next_result is None:
            # check if there are results after the poison pill
            if self.m_result_queue.empty():
                poison_pill = 1

                # shut down the writer
                self.m_result_queue.task_done()

            else:
                poison_pill = 2

                # put back the poison pill for the moment
                self.m_result_queue.put(None)
                self.m_result_queue.task_done()

        else:
            poison_pill = 0

        return poison_pill

    @typechecked
    def run(self) -> None:
        """
        Run method of the :class:`~pynpoint.util.multiproc.TaskWriter`. It is called once when
        it has to start storing the results until it receives a poison pill.

        Returns
        -------
        NoneType
            None
        """

        while True:
            next_result = self.m_result_queue.get()
            poison_pill_case = self.check_poison_pill(next_result)

            if poison_pill_case == 1:
                break

            if poison_pill_case == 2:
                continue

            with self.m_data_mutex:
                self.m_data_out_port._check_status_and_activate()
                self.m_data_out_port[to_slice(next_result.m_position)] = next_result.m_data_array
                self.m_data_out_port.close_port()

            self.m_result_queue.task_done()


class MultiprocessingCapsule(metaclass=ABCMeta):
    """
    Abstract interface for multiprocessing capsules based on the poison pill pattern.
    """

    @typechecked
    def __init__(self,
                 image_in_port: Union[InputPort, None],
                 image_out_port: Union[OutputPort, None],
                 num_proc: np.int64) -> None:
        """
        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort, None
            Port to the input data.
        image_out_port : pynpoint.core.dataio.OutputPort, None
            Port to the place where the output data will be stored.
        num_proc : int
            Number of task processors.

        Returns
        -------
        NoneType
            None
        """

        # buffer twice the data as processes are available
        self.m_tasks_queue = multiprocessing.JoinableQueue(maxsize=num_proc)
        self.m_result_queue = multiprocessing.JoinableQueue(maxsize=num_proc)
        self.m_num_proc = num_proc

        # database mutex
        self.m_data_mutex = multiprocessing.Lock()

        # create reader
        self.m_creator = self.init_creator(image_in_port)

        # create processors
        self.m_task_processors = self.create_processors()

        # create writer
        self.m_writer = self.create_writer(image_out_port)

    @abstractmethod
    @typechecked
    def create_processors(self) -> None:
        """
        Function that is called from the constructor to create a list of instances of
        :class:`~pynpoint.util.multiproc.TaskProcessor`.

        Returns
        -------
        NoneType
            None
        """

    @abstractmethod
    @typechecked
    def init_creator(self,
                     image_in_port: Union[InputPort, None]) -> None:
        """
        Function that is called from the constructor to create a
        :class:`~pynpoint.util.multiproc.TaskCreator`.

        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort, None
            Input port for the task creator.

        Returns
        -------
        NoneType
            None
        """

    @typechecked
    def create_writer(self,
                      image_out_port: Union[OutputPort, None]) -> TaskWriter:
        """
        Function that is called from the constructor to create the
        :class:`~pynpoint.util.multiproc.TaskWriter`.

        Parameters
        ----------
        image_out_port : pynpoint.core.dataio.OutputPort, None
            Output port for the creator.

        Returns
        -------
        pynpoint.util.multiproc.TaskWriter
            Task writer.
        """

        return TaskWriter(self.m_result_queue,
                          image_out_port,
                          self.m_data_mutex)

    @typechecked
    def run(self) -> None:
        """
        Run method that starts the :class:`~pynpoint.util.multiproc.TaskCreator`, the instances
        of :class:`~pynpoint.util.multiproc.TaskProcessor`, and the
        :class:`~pynpoint.util.multiproc.TaskWriter`. They will be shut down when all tasks have
        finished.

        Returns
        -------
        NoneType
            None
        """

        # start all processes
        self.m_creator.start()

        for processor in self.m_task_processors:
            processor.start()

        self.m_writer.start()

        # wait for all tasks to have finished
        self.m_tasks_queue.join()
        self.m_result_queue.join()

        # clean up the processes
        for processor in self.m_task_processors:
            processor.join()

        self.m_writer.join()
        self.m_creator.join()


@typechecked
def apply_function(tmp_data: np.ndarray,
                   func: Callable,
                   func_args: Union[tuple, None]) -> np.ndarray:
    """
    Apply a function with optional arguments to the input data.

    Parameters
    ----------
    tmp_data : numpy.ndarray
        Input data.
    func : function
        Function.
    func_args : tuple, None
        Function arguments.

    Returns
    -------
    numpy.ndarray
        The results of the function.
    """

    if func_args is None:
        result = np.array(func(tmp_data))
    else:
        result = np.array(func(tmp_data, *func_args))

    return result


@typechecked
def to_slice(tuple_slice: tuple) -> tuple:
    """
    Function to convert tuples into slices for a multiprocessing queue.

    Parameters
    ----------
    tuple_slice : tuple
        Tuple to be converted into a slice.

    Returns
    -------
    tuple(slice, slice, slice)
        Tuple with three slices.
    """

    slices = []
    for item in tuple_slice:
        # slice(start, stop step)
        slices.append(slice(item[0], item[1], item[2]))

    return tuple(slices)
