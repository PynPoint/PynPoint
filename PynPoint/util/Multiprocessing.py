import multiprocessing
import numpy as np
from abc import ABCMeta, abstractmethod


# ----- General Multiprocessing classes using the poison pill pattern ------

class TaskResult(object):
    def __init__(self,
                 data_array,
                 position):
        """
        Result object which can be stored by the Writer.
        :param data_array: Some kind of 1/2/3D data which is the result for a given position
        :param position: The position where the result data will be stored
        :type position: slice
        """
        self.m_data_array = data_array
        self.m_position = position


class TaskInput(object):

    def __init__(self,
                 input_data,
                 job_parameter):
        self.m_input_data = input_data
        self.m_job_parameter = job_parameter


class TaskCreator(multiprocessing.Process):
    __metaclass__ = ABCMeta

    def __init__(self,
                 data_port_in,
                 tasks_queue_in,
                 data_mutex_in,
                 number_of_processors):

        multiprocessing.Process.__init__(self)
        self.m_data_mutex = data_mutex_in
        self.m_task_queue = tasks_queue_in

        self.m_number_of_processors = number_of_processors
        self.m_data_in_port = data_port_in

    @abstractmethod
    def run(self):
        """
        Creates TaskInput objects until all tasks are done
        :return:
        """
        pass

    def create_poison_pills(self):

        for i in range(self.m_number_of_processors - 1):
            # poison pills
            self.m_task_queue.put(1)

        # Final poison pill
        self.m_task_queue.put(None)


class TaskProcessor(multiprocessing.Process):
    __metaclass__ = ABCMeta

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in):

        multiprocessing.Process.__init__(self)
        self.m_task_queue = tasks_queue_in
        self.m_result_queue = result_queue_in

    def check_poison_pill(self,
                          next_task):
        process_name = self.name

        if next_task is 1:
            # Poison pill means shutdown
            print '%s: Exiting' % process_name
            self.m_task_queue.task_done()
            return True

        if next_task is None:
            # got final Poison pill
            self.m_result_queue.put(None)  # shut down writer process

            print '%s: Exiting' % process_name
            self.m_task_queue.task_done()
            return True

        return False

    def run(self):

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
        pass


class TaskWriter(multiprocessing.Process):
    def __init__(self,
                 result_queue_in,
                 data_out_port_in,
                 data_mutex_in):

        multiprocessing.Process.__init__(self)
        self.m_result_queue = result_queue_in
        self.m_data_mutex = data_mutex_in
        self.m_data_out_port = data_out_port_in

    def check_poison_pill(self, next_result):
        if next_result is None:
            # check if no results are after the poison pill
            if self.m_result_queue.empty():
                print "Shutting down writer..."
                self.m_result_queue.task_done()
            else:
                # put pack the Poison pill for the moment
                print "put back poison pill"
                self.m_result_queue.task_done()
                self.m_result_queue.put(None)
                return False

            return True
        return False

    def run(self):

        while True:
            next_result = self.m_result_queue.get()

            print self.m_result_queue.empty()

            # Poison Pill
            if self.check_poison_pill(next_result):
                break

            print "writing"
            with self.m_data_mutex:
                self.m_data_out_port[to_slice(next_result.m_position)] = next_result.m_data_array

            self.m_result_queue.task_done()


# ------ Multiprocessing Capsule -------
class MultiprocessingCapsule(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 image_in_port,
                 image_out_port,
                 num_processors):
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
        tmp_writer = TaskWriter(self.m_result_queue,
                                image_out_port,
                                self.m_data_mutex)
        return tmp_writer

    def create_processors(self):
        tmp_processors = [TaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                        result_queue_in=self.m_result_queue)
                                  for i in xrange(self.m_num_processors)]
        return tmp_processors

    @abstractmethod
    def init_creator(self, image_in_port):
        return None

    def run(self):
        # start all processes
        self.m_creator.start()

        for processor in self.m_task_processors:
            processor.start()

        self.m_writer.start()

        # Wait for all of the tasks to finish
        print 1
        self.m_tasks_queue.join()
        print 1.1
        self.m_result_queue.join()

        print 2

        # Clean up Processes
        for processor in self.m_task_processors:
            processor.join()

        print 3

        self.m_writer.join()
        print 4
        self.m_creator.join()

        print 5


# ----- Handler to apply a function ------
def apply_function(tmp_data, func, func_args):
    # process line
    # check if additional arguments are given
    if func_args is None:
        return np.array(func(tmp_data))
    else:
        return np.array(func(tmp_data, *func_args))


def to_slice(tuple_slice):
    """
    this function is needed for pickling slices
    :param tuple_slice:
    :return:
    """
    return (slice(tuple_slice[0][0], tuple_slice[0][1], tuple_slice[0][2]),
            slice(tuple_slice[1][0], tuple_slice[1][1], tuple_slice[1][2]),
            slice(tuple_slice[2][0], tuple_slice[2][1], tuple_slice[2][2]))


# ----- Multiprocessing on lines ------
class LineTaskProcessor(TaskProcessor):

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in,
                 function,
                 function_args):
        super(LineTaskProcessor, self).__init__(tasks_queue_in,
                                                result_queue_in)
        self.m_function = function
        self.m_function_args = function_args

    def run_job(self, tmp_task):
        result_arr = np.zeros((tmp_task.m_job_parameter[0],
                               tmp_task.m_input_data.shape[1],
                               tmp_task.m_input_data.shape[2]))

        for i in range(tmp_task.m_input_data.shape[1]):
            for j in range(tmp_task.m_input_data.shape[2]):
                tmp_line = tmp_task.m_input_data[:, i, j]

                result_arr[:, i, j] = apply_function(tmp_line,
                                                     self.m_function,
                                                     self.m_function_args)

        result = TaskResult(result_arr,
                            tmp_task.m_job_parameter[1])

        return result


class LineReader(TaskCreator):

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
                print "Reading lines from " + str(i) + " to " + str(j)
                tmp_data = self.m_data_in_port[:, i:j, :]

            self.m_task_queue.put(TaskInput(tmp_data,
                                            (self.m_length_of_processed_data,
                                             ((None, None, None),
                                              (i, j, None),
                                              (None, None, None)))))
            i = j

        self.create_poison_pills()


class LineProcessingCapsule(MultiprocessingCapsule):

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
        tmp_processors = [LineTaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                            result_queue_in=self.m_result_queue,
                                            function=self.m_function,
                                            function_args=self.m_function_args)
                          for i in xrange(self.m_num_processors)]
        return tmp_processors

    def init_creator(self, image_in_port):
        reader = LineReader(image_in_port,
                            self.m_tasks_queue,
                            self.m_data_mutex,
                            self.m_num_processors,
                            self.m_length_of_processed_data)
        return reader

