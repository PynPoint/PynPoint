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
    __metaclass__ = ABCMeta

    def __init__(self,
                 input_data,
                 job_parameter):
        self.m_input_data = input_data
        self.m_job_parameter = job_parameter

    @abstractmethod
    def run_job(self):
        pass


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

        print "Process " + process_name + " got data for " + str(next_task.m_position) \
              + " and starts processing..."

        return False

    def run(self):

        # Process till we get a poison pill
        while True:
            next_task = self.m_task_queue.get()

            if self.check_poison_pill(next_task=next_task):
                break

            result = next_task.run_job()

            self.m_task_queue.task_done()
            self.m_result_queue.put(result)

            print "Process " + self.name + " finished processing!"


class TaskWriter(multiprocessing.Process):
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

            # Poison Pill
            if next_result is None:
                print "Shutting down writer..."
                self.m_result_queue.task_done()
                break

            print "Start writing row " + str(next_result.m_position)

            with self.m_data_mutex:
                self.m_data_out_port[next_result.m_position] = next_result.m_data_array

            self.m_result_queue.task_done()

# ------ Mulitprocessing Calsule -------
class MulitporcessingCapluse(object):
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
        self.m_task_processors = [TaskProcessor(tasks_queue_in=self.m_tasks_queue,
                                                result_queue_in=self.m_result_queue)
                                  for i in xrange(num_processors)]

        # create writer
        self.m_writer = TaskWriter(self.m_result_queue,
                                           image_out_port,
                                           self.m_data_mutex)

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
        self.m_tasks_queue.join()
        self.m_result_queue.join()

        # Clean up Processes
        for processor in self.m_task_processors:
            processor.join()

        self.m_writer.join()
        self.m_creator.join()


# ----- Handler to apply a function ------
def apply_function(tmp_data, func, func_args):
    # process line
    # check if additional arguments are given
    if func_args is None:
        return np.array(func(tmp_data))
    else:
        return np.array(func(tmp_data, *func_args))


# ----- Multiprocessing on lines ------
class LineTaskInput(TaskInput):

    def __init__(self,
                 input_data,
                 job_parameter,
                 result_slice):
        # TODO Better documentation
        """

        :param input_data:
        :param job_parameter: tuple (func, func_args)
        :param result_slice:
        """
        super(LineTaskInput, self).__init__(input_data, job_parameter)
        self.m_result_slice = result_slice

    def run_job(self):
        result_arr = np.zeros((length_of_processed_data,
                               self.m_input_data.shape[1],
                               self.m_input_data.shape[2]))

        for i in range(self.m_input_data.shape[1]):
            for j in range(self.m_input_data.shape[2]):
                tmp_line = self.m_input_data[:, i, j]
                result_arr[:, i, j] = apply_function(tmp_line,
                                                     self.m_job_parameter[0],
                                                     self.m_job_parameter[1])

        result = TaskResult(result_arr,
                            self.m_result_slice)

        return result


class LineReader(TaskCreator):

    def __init__(self,
                 data_port_in,
                 tasks_queue_in,
                 data_mutex_in,
                 number_of_processors,
                 func,
                 func_args):
        super(LineReader, self).__init__(data_port_in,
                                         tasks_queue_in,
                                         data_mutex_in,
                                         number_of_processors)
        self.m_function = func
        self.m_function_args = func_args

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

            self.m_task_queue.put(LineTaskInput(tmp_data,
                                                (self.m_function, self.m_function_args),
                                                (slice(None, None, None),
                                                 slice(i, j, None),
                                                 slice(None, None, None))))
            i = j

        self.create_poison_pills()


class LineProcessingCapsule(MulitporcessingCapluse):

    def __init__(self,
                 image_in_port,
                 image_out_port,
                 num_processors,
                 function,
                 function_args):
        super(LineProcessingCapsule, self).__init__(image_in_port, image_out_port, num_processors)
        self.m_function = function
        self.m_function_args = function_args

    def init_creator(self, image_in_port):
        reader = LineReader(image_in_port,
                            self.m_tasks_queue,
                            self.m_data_mutex,
                            self.m_num_processors,
                            self.m_function,
                            self.m_function_args)
        return reader



'''

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


print "Preparing database for analysis ..."

image_out_port.set_all(np.zeros((length_of_processed_data, \
                                 image_in_port.get_shape()[1], \
                                 image_in_port.get_shape()[2])),
                       data_dim=3,
                       keep_attributes=False)  # overwrite old existing attributes

num_processors = multiprocessing.cpu_count()

print "Database prepared. Starting analysis with " + str(num_processors) + " processes."

# Establish communication queues'''



