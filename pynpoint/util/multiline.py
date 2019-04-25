"""
Utilities for multiprocessing of lines in time with the poison pill pattern.
"""

import six
import numpy as np

from pynpoint.util.multiproc import TaskInput, TaskResult, TaskCreator, TaskProcessor, \
                                    MultiprocessingCapsule, apply_function


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

    def run_job(self,
                tmp_task):
        """
        Parameters
        ----------
        tmp_task : pynpoint.util.multiproc.TaskInput
            Input task.

        Returns
        -------
        pynpoint.util.multiproc.TaskResult
            Task result.
        """

        result_arr = np.zeros((tmp_task.m_job_parameter[0],
                               tmp_task.m_input_data.shape[1],
                               tmp_task.m_input_data.shape[2]))

        for i in six.moves.range(tmp_task.m_input_data.shape[1]):
            for j in six.moves.range(tmp_task.m_input_data.shape[2]):
                tmp_line = tmp_task.m_input_data[:, i, j]

                result_arr[:, i, j] = apply_function(tmp_line,
                                                     self.m_function,
                                                     self.m_function_args)

        return TaskResult(result_arr, tmp_task.m_job_parameter[1])


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
                 data_length):
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
        number_of_processors : int
            Number of processors.
        data_length : int
            Length of the processed data.

        Returns
        -------
        NoneType
            None
        """

        super(LineReader, self).__init__(data_port_in,
                                         tasks_queue_in,
                                         data_mutex_in,
                                         number_of_processors)

        self.m_data_length = data_length

    def run(self):
        """
        Returns
        -------
        NoneType
            None
        """

        total_number_of_rows = self.m_data_in_port.get_shape()[1]
        row_length = int(np.ceil(self.m_data_in_port.get_shape()[1] /
                                 float(self.m_number_of_processors)))

        i = 0
        while i < total_number_of_rows:
            # read rows from i to j
            j = min((i + row_length), total_number_of_rows)

            # lock mutex and read data
            with self.m_data_mutex:
                # reading lines from i to j
                tmp_data = self.m_data_in_port[:, i:j, :]

            param = (self.m_data_length, ((None, None, None), (i, j, None), (None, None, None)))
            self.m_task_queue.put(TaskInput(tmp_data, param))

            i = j

        self.create_poison_pills()


class LineProcessingCapsule(MultiprocessingCapsule):
    """
    The central processing class for parallel line processing. Use this class to apply a function
    in time in parallel, for example as in
    :class:`~pynpoint.processing.timedenoising.WaveletTimeDenoisingModule`.
    """

    def __init__(self,
                 image_in_port,
                 image_out_port,
                 num_processors,
                 function,
                 function_args,
                 data_length):
        """
        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort
            Input port.
        image_out_port : pynpoint.core.dataio.OutputPort
            Output port.
        num_processors : int
            Number of processors.
        function : function
            Input function.
        function_args :
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

        super(LineProcessingCapsule, self).__init__(image_in_port, image_out_port, num_processors)

    def create_processors(self):
        """
        Returns
        -------
        list(pynpoint.util.multiproc.LineTaskProcessor, )
            List with line task processors.
        """

        tmp_processors = []

        for _ in six.moves.range(self.m_num_processors):

            tmp_processors.append(LineTaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                                    result_queue_in=self.m_result_queue,
                                                    function=self.m_function,
                                                    function_args=self.m_function_args))

        return tmp_processors

    def init_creator(self,
                     image_in_port):
        """
        Parameters
        ----------
        image_in_port : pynpoint.core.dataio.InputPort
            Input port.

        Returns
        -------
        pynpoint.util.multiline.LineReader
            Line reader object.
        """

        return LineReader(image_in_port,
                          self.m_tasks_queue,
                          self.m_data_mutex,
                          self.m_num_processors,
                          self.m_data_length)
