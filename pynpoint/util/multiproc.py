"""
Utilities for poison pill multiprocessing. Provides abstract interfaces as well as an
implementation needed to process lines in time as used in the wavelet time denoising.
"""

from __future__ import absolute_import

import multiprocessing

from abc import ABCMeta, abstractmethod
# from . import multiprocessing

import six
import numpy as np


# ----- General Multiprocessing classes using the poison pill pattern ------

class TaskResult(object):
    """
    Result object which can be stored by the TaskWriter.
    """

    def __init__(self,
                 data_array,
                 position):
        """
        Constructor of TaskResult.

        :param data_array: Some kind of 1/2/3D data which is the result for a given position.
        :type data_array:
        :param position: The position where the result data will be stored
        :type position: slice

        :return: None
        """

        self.m_data_array = data_array
        self.m_position = position


class TaskInput(object):
    """
    Data and parameter capsule for tasks to be processed by the TaskProcessor.
    """

    def __init__(self,
                 input_data,
                 job_parameter):
        """
        Constructor of TaskInput.

        :param input_data: Data needed by the TaskProcessors
        :type input_data:
        :param job_parameter: Additional data or parameters
        :type job_parameter: tuple

        :return: None
        """

        self.m_input_data = input_data
        self.m_job_parameter = job_parameter


class TaskCreator(six.with_metaclass(ABCMeta, multiprocessing.Process)):
    """
    Abstract Interface for all TaskCreator classes. A TaskCreator is supposed to create instances
    of TaskInput which can be processed by a TaskProcessor and appends them to a central task
    queue. In general there is only one TaskCreator running for a poison pill multiprocessing
    application. A TaskCreator needs to communicate to the writer in order to avoid simultaneously
    access to the central database.
    """

    def __init__(self,
                 data_port_in,
                 tasks_queue_in,
                 data_mutex_in,
                 number_of_processors):
        """
        Constructor of TaskCreator. Can only be called by using super from children classes.

        :param data_port_in: A Port which links to data to be processed.
        :type data_port_in:
        :param tasks_queue_in: The central Task queue.
        :type tasks_queue_in: multiprocessing.Queue
        :param data_mutex_in: A mutex shared with the writer to ensure that no read and write
                              operations happen at the same time.
        :type data_mutex_in:
        :param number_of_processors: Maximum number of TaskProcessors running at the same time.
        :type number_of_processors:
        """

        multiprocessing.Process.__init__(self)

        self.m_data_mutex = data_mutex_in
        self.m_task_queue = tasks_queue_in

        self.m_number_of_processors = number_of_processors
        self.m_data_in_port = data_port_in

    @abstractmethod
    def run(self):
        """
        Creates objects of TaskInput until all tasks are in the task queue

        :return: None
        """

        # pass

    def create_poison_pills(self):
        """
        Function which creates the poison pills for TaskProcessor and TaskWriter. If a process
        gets a poison pill as the new task it will shut down. Run the method at the end of the
        run method.

        :return: None
        """

        for _ in six.moves.range(self.m_number_of_processors-1):
            # poison pills
            self.m_task_queue.put(1)

        # final poison pill
        self.m_task_queue.put(None)


class TaskProcessor(six.with_metaclass(ABCMeta, multiprocessing.Process)):
    """
    Abstract interface for a TaskProcessor. There are up to CPU count instances of TaskProcessor
    running at the same time in a poison pill multiprocessing application. There is no guarantee
    which process finishes first. A TaskProcessor takes tasks from a task-queue performs an
    analysis and stores the result back into a result-queue. If the next task is a poison pill it
    is should down.
    """

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in):
        """
        Abstract constructor of TaskProcessor.

        :param tasks_queue_in: The input task queue (contains TaskInput instances).
        :type tasks_queue_in:
        :param result_queue_in: The result task queue (contains TaskResult instances).
        :type result_queue_in:

        :return: None
        """

        multiprocessing.Process.__init__(self)

        self.m_task_queue = tasks_queue_in
        self.m_result_queue = result_queue_in

    def check_poison_pill(self,
                          next_task):
        """
        Run this function to check if the next task is a poison pill. The shut down of the process
        needs to be done in its run method.

        :param next_task: The next task.
        :type next_task:

        :return: True if the next task is a poison pill, else False.
        :rtype: bool
        """

        if next_task == 1:
            # Poison pill means shutdown
            # print '%s: Exiting' % self.name
            self.m_task_queue.task_done()
            return True

        if next_task is None:
            # got final Poison pill
            self.m_result_queue.put(None)  # shut down writer process
            # print '%s: Exiting' % self.name
            self.m_task_queue.task_done()
            return True

        return False

    def run(self):
        """
        The run method is called to start a TaskProcessor. The process will continue to process
        tasks from the input task queue until it gets a poison pill.

        :return: None
        """

        # Process till we get a poison pill
        while True:
            next_task = self.m_task_queue.get()

            if self.check_poison_pill(next_task=next_task):
                break

            result = self.run_job(next_task)

            self.m_task_queue.task_done()
            self.m_result_queue.put(result)

    @abstractmethod
    def run_job(self, tmp_task):
        """
        Abstract interface for the run_job method which is called from run() for each task
        individually.

        :param tmp_task: Input Task
        :type tmp_task:

        :return: A TaskResult
        :rtype:
        """

        # pass


class TaskWriter(multiprocessing.Process):
    """
    The TaskWriter takes results from the result queue computed by a TaskProcessor and stores
    them into the central database. It uses the position parameter of the TaskResult objects in
    order to slice the result to the correct location in global output.
    """

    def __init__(self,
                 result_queue_in,
                 data_out_port_in,
                 data_mutex_in):

        multiprocessing.Process.__init__(self)

        self.m_result_queue = result_queue_in
        self.m_data_mutex = data_mutex_in
        self.m_data_out_port = data_out_port_in

    def check_poison_pill(self, next_result):
        """
        Checks if the next result is a poison pill.

        :param next_result: The next result.
        :type next_result:

        :return: 0 -> no poison pill, 1 -> poison pill, 2 -> poison pill but still results in the
                 queue (rare error case).
        :rtype: int
        """

        if next_result is None:
            # check if no results are after the poison pill
            if self.m_result_queue.empty():
                # print "Shutting down writer..."
                self.m_result_queue.task_done()
                return 1

            # put pack the Poison pill for the moment
            # print "put back poison pill"
            self.m_result_queue.put(None)
            self.m_result_queue.task_done()
            return 2

        return 0

    def run(self):
        """
        The run method of the writer process is called once and will start him to store results
        until it gets a poison pill.

        :return: None
        """

        while True:
            next_result = self.m_result_queue.get()

            # Poison Pill
            poison_pill_case = self.check_poison_pill(next_result)
            if poison_pill_case == 1:
                break
            if poison_pill_case == 2:
                continue

            with self.m_data_mutex:
                self.m_data_out_port[to_slice(next_result.m_position)] = next_result.m_data_array

            self.m_result_queue.task_done()


# ------ Multiprocessing Capsule -------
class MultiprocessingCapsule(six.with_metaclass(ABCMeta, object)):
    """
    Abstract interface for multiprocessing capsules based on the poison pill patter. It consists
    of a TaskCreator, a result writer as well as a list of Task Processors.
    """

    def __init__(self,
                 image_in_port,
                 image_out_port,
                 num_processors):
        """
        Constructor can only be called from children classes by using super.

        :param image_in_port: Port to the input data
        :type image_in_port:
        :param image_out_port: Port to the place where the output will be stored
        :type image_in_port:
        :param num_processors: Maximum number of Task Processors
        :type num_processors:

        :return: None
        """

        # buffer twice the data as processes are available
        self.m_tasks_queue = multiprocessing.JoinableQueue(maxsize=num_processors)
        self.m_result_queue = multiprocessing.JoinableQueue(maxsize=num_processors)
        self.m_num_processors = num_processors

        # data base mutex
        self.m_data_mutex = multiprocessing.Lock()

        # create reader
        self.m_creator = self.init_creator(image_in_port)

        # Start consumers
        self.m_task_processors = self.create_processors()

        # create writer
        self.m_writer = self.create_writer(image_out_port)

    def create_writer(self, image_out_port):
        """
        Called from the constructor to create the writer object.

        :param image_out_port: Output port for the creator.
        :type image_out_port:

        :return: Writer object
        :rtype:
        """

        tmp_writer = TaskWriter(self.m_result_queue,
                                image_out_port,
                                self.m_data_mutex)

        return tmp_writer

    @abstractmethod
    def create_processors(self):
        """
        Called from the constructor to create a list of a TaskProcessor.
        """

        # loop to create Task Processors
        tmp_processors = []
        return tmp_processors

    @abstractmethod
    def init_creator(self, image_in_port):
        """
        Called from the constructor to create a creator object.

        :param image_in_port: Input port for the creator.
        :type image_in_port:

        :return: Creator object.
        :rtype:
        """

        return None

    def run(self):
        """
        The run method starts the Creator, all Task Processors and the Writer Process. Finally it
        will shut down them again after all tasks are done.

        :return: None
        """

        # start all processes
        self.m_creator.start()

        for processor in self.m_task_processors:
            processor.start()

        self.m_writer.start()

        # Wait for all of the tasks to finish
        self.m_tasks_queue.join()
        self.m_result_queue.join()

        # Clean up Processes
        for processor in self.m_task_processors:
            processor.join()

        self.m_writer.join()
        self.m_creator.join()


# ----- Handler to apply a function ------
def apply_function(tmp_data, func, func_args):
    """
    Applies the function func with its arguments func_args to the tmp_data

    :return: the results of the function
    :rtype:
    """

    # process line
    # check if additional arguments are given
    if func_args is None:
        return np.array(func(tmp_data))

    return np.array(func(tmp_data, *func_args))


def to_slice(tuple_slice):
    """
    this function is needed to pickle slices as reburied for multiprocessing queues

    :param tuple_slice: Tuple to be converted to a slice
    :type tuple_slice:

    :return: the slice
    :rtype:
    """

    return (slice(tuple_slice[0][0], tuple_slice[0][1], tuple_slice[0][2]),
            slice(tuple_slice[1][0], tuple_slice[1][1], tuple_slice[1][2]),
            slice(tuple_slice[2][0], tuple_slice[2][1], tuple_slice[2][2]))


# ----- Multiprocessing on lines ------
class LineTaskProcessor(TaskProcessor):
    """
    Line Task Processors are part of the parallel line processing. They take a row of lines in time
    and apply a function to them.
    """

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in,
                 function,
                 function_args):

        super(LineTaskProcessor, self).__init__(tasks_queue_in, result_queue_in)

        self.m_function = function
        self.m_function_args = function_args

    def run_job(self, tmp_task):
        result_arr = np.zeros((tmp_task.m_job_parameter[0],
                               tmp_task.m_input_data.shape[1],
                               tmp_task.m_input_data.shape[2]))

        for i in six.moves.range(tmp_task.m_input_data.shape[1]):
            for j in six.moves.range(tmp_task.m_input_data.shape[2]):
                tmp_line = tmp_task.m_input_data[:, i, j]

                result_arr[:, i, j] = apply_function(tmp_line,
                                                     self.m_function,
                                                     self.m_function_args)

        result = TaskResult(result_arr, tmp_task.m_job_parameter[1])

        return result


class LineReader(TaskCreator):
    """
    Line Reader are part of the parallel line processing. They continuously read all rows of a data
    set and puts them into a task queue.
    """

    def __init__(self,
                 data_port_in,
                 tasks_queue_in,
                 data_mutex_in,
                 number_of_processors,
                 length_of_processed_data):

        super(LineReader, self).__init__(data_port_in,
                                         tasks_queue_in,
                                         data_mutex_in,
                                         number_of_processors)

        self.m_length_of_processed_data = length_of_processed_data

    def run(self):

        total_number_of_rows = self.m_data_in_port.get_shape()[1]
        row_length = int(np.ceil(self.m_data_in_port.get_shape()[1] /
                                 float(self.m_number_of_processors)))

        i = 0
        while i < total_number_of_rows:
            # read rows from i to j
            j = min((i + row_length), total_number_of_rows)

            # lock Mutex and read data
            with self.m_data_mutex:
                # print "Reading lines from " + str(i) + " to " + str(j)
                tmp_data = self.m_data_in_port[:, i:j, :]

            self.m_task_queue.put(TaskInput(tmp_data,
                                            (self.m_length_of_processed_data,
                                             ((None, None, None),
                                              (i, j, None),
                                              (None, None, None)))))
            i = j

        self.create_poison_pills()


class LineProcessingCapsule(MultiprocessingCapsule):
    """
    The central processing class for parallel line processing. Use this class to apply a function
    in time in parallel.
    """

    def __init__(self,
                 image_in_port,
                 image_out_port,
                 num_processors,
                 function,
                 function_args,
                 length_of_processed_data):

        self.m_function = function
        self.m_function_args = function_args
        self.m_length_of_processed_data = length_of_processed_data

        super(LineProcessingCapsule, self).__init__(image_in_port, image_out_port, num_processors)

    def create_processors(self):

        tmp_processors = []

        for _ in six.moves.range(self.m_num_processors):

            tmp_processors.append(LineTaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                                    result_queue_in=self.m_result_queue,
                                                    function=self.m_function,
                                                    function_args=self.m_function_args))

        return tmp_processors

    def init_creator(self, image_in_port):

        reader = LineReader(image_in_port,
                            self.m_tasks_queue,
                            self.m_data_mutex,
                            self.m_num_processors,
                            self.m_length_of_processed_data)

        return reader
