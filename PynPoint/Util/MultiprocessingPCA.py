"""
Capsule for multiprocessing of the PSF subtraction with PCA. Residuals are created in parallel for
a range of principal components for which the PCA basis is required as input. Note that the
multiprocessing does not run on macOS due to a missing functionality in numpy.
"""

from __future__ import absolute_import

import sys

import numpy as np
from six.moves import range

from PynPoint.Util.Multiprocessing import TaskProcessor, TaskCreator, TaskWriter, TaskResult, \
                                          TaskInput, MultiprocessingCapsule, to_slice
from PynPoint.Util.PSFSubtractionTools import pca_psf_subtraction
from PynPoint.Util.Residuals import combine_residuals


class PcaTaskCreator(TaskCreator):
    """
    The TaskCreator of the PCA multiprocessing. Creates one task for each principal component
    number. Does not require an input port since the data is directly given to the task processors.
    """

    def __init__(self,
                 tasks_queue_in,
                 number_of_processors,
                 pca_numbers):
        """
        Constructor of PcaTaskCreator.

        :param tasks_queue_in: Input task queue.
        :type tasks_queue_in: multiprocessing.queues.JoinableQueue
        :param number_of_processors: Number of processors.
        :type number_of_processors: int
        :param pca_numbers: Principal components for which the residuals are computed.
        :type pca_numbers: numpy.ndarray

        :return: None
        """

        super(PcaTaskCreator, self).__init__(None, tasks_queue_in, None, number_of_processors)

        self.m_pca_numbers = pca_numbers

    def run(self):
        """
        Run method of PcaTaskCreator.

        :return: None
        """

        res_position = 0

        for pca_number in self.m_pca_numbers:
            parameters = (((res_position, res_position+1, None),
                           (None, None, None), (None, None, None)), )

            self.m_task_queue.put(TaskInput(pca_number, parameters))

            res_position += 1

        self.create_poison_pills()


class PcaTaskProcessor(TaskProcessor):
    """
    The TaskProcessor of the PCA multiprocessing is the core of the parallelization. An instance
    of this class will calculate one forward and backward PCA transformation given the pre-trained
    scikit-learn PCA model. It does not get data from the TaskCreator but uses its own copy of the
    input data, which are the same and independent for each task. The following residuals can be
    created:

    * Mean residuals -- requirements[0] = True
    * Median residuals -- requirements[1] = True
    * Noise-weighted residuals -- requirements[2] = True
    * Clipped mean of the residuals -- requirements[3] = True
    """

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in,
                 star_reshape,
                 angles,
                 pca_model,
                 im_shape,
                 indices,
                 requirements=(False, False, False, False)):
        """
        Constructor of PcaTaskProcessor.

        :param tasks_queue_in: Input task queue.
        :type tasks_queue_in: multiprocessing.queues.JoinableQueue
        :param result_queue_in: Input result queue.
        :type result_queue_in: multiprocessing.queues.JoinableQueue
        :param star_reshape: Reshaped (2D) stack of images.
        :type star_reshape: numpy.ndarray
        :param angles: Derotation angles (deg).
        :type angles: numpy.ndarray
        :param pca_model: PCA object with the basis.
        :type pca_model: sklearn.decomposition.pca.PCA
        :param im_shape: Original shape of the stack of images.
        :type im_shape: tuple(int, int, int)
        :param indices: Non-masked image indices.
        :type indices: numpy.ndarray
        :param requirements: Required output residuals.
        :type requirements: tuple(bool, bool, bool, bool)

        :return: None
        """

        super(PcaTaskProcessor, self).__init__(tasks_queue_in, result_queue_in)

        self.m_star_reshape = star_reshape
        self.m_pca_model = pca_model
        self.m_angles = angles
        self.m_im_shape = im_shape
        self.m_indices = indices
        self.m_requirements = requirements

    def run_job(self, tmp_task):
        """
        Run method of PcaTaskProcessor.

        :param tmp_task: Input task.
        :type tmp_task: PynPoint.Util.Multiprocessing.TaskInput

        :return: Output residuals.
        :rtype: PynPoint.Util.Multiprocessing.TaskResult
        """

        residuals, res_rot = pca_psf_subtraction(images=self.m_star_reshape,
                                                 angles=self.m_angles,
                                                 pca_number=tmp_task.m_input_data,
                                                 pca_sklearn=self.m_pca_model,
                                                 im_shape=self.m_im_shape,
                                                 indices=self.m_indices)

        res_output = np.zeros((4, res_rot.shape[1], res_rot.shape[2]))

        if self.m_requirements[0]:
            res_output[0, ] = combine_residuals(method="mean", res_rot=res_rot)

        if self.m_requirements[1]:
            res_output[1, ] = combine_residuals(method="median", res_rot=res_rot)

        if self.m_requirements[2]:
            res_output[2, ] = combine_residuals(method="weighted",
                                                res_rot=res_rot,
                                                residuals=residuals,
                                                angles=self.m_angles)

        if self.m_requirements[3]:
            res_output[3, ] = combine_residuals(method="clipped", res_rot=res_rot)

        sys.stdout.write('.')
        sys.stdout.flush()

        return TaskResult(res_output, tmp_task.m_job_parameter[0])


class PcaTaskWriter(TaskWriter):
    """
    The TaskWriter of the PCA parallelization. Four different ports are used to save the
    results of the task processors (mean, median, weighted, and clipped).
    """

    def __init__(self,
                 result_queue_in,
                 mean_out_port,
                 median_out_port,
                 weighted_out_port,
                 clip_out_port,
                 data_mutex_in,
                 requirements=(False, False, False, False)):
        """
        Constructor of PcaTaskWriter.

        :param result_queue_in: Input result queue.
        :type result_queue_in: multiprocessing.queues.JoinableQueue
        :param mean_out_port: Output port with the mean residuals. Not used if set to None.
        :type mean_out_port: PynPoint.Core.DataIO.OutputPort
        :param median_out_port: Output port with the median residuals. Not used if set to None.
        :type median_out_port: PynPoint.Core.DataIO.OutputPort
        :param weighted_out_port: Output port with the noise-weighted residuals. Not used if set
                                  to None.
        :type weighted_out_port: PynPoint.Core.DataIO.OutputPort
        :param clip_out_port: Output port with the clipped mean residuals. Not used if set to None.
        :type clip_out_port: PynPoint.Core.DataIO.OutputPort
        :param data_mutex_in: A mutual exclusion variable which ensure that no read and write
                           simultaneously occur.
        :type data_mutex_in: multiprocessing.synchronize.Lock
        :param requirements: Required output residuals.
        :type requirements: tuple(bool, bool, bool, bool)

        :return: None
        """

        if mean_out_port is not None:
            data_out_port_in = mean_out_port

        elif median_out_port is not None:
            data_out_port_in = median_out_port

        elif weighted_out_port is not None:
            data_out_port_in = weighted_out_port

        elif clip_out_port is not None:
            data_out_port_in = clip_out_port

        super(PcaTaskWriter, self).__init__(result_queue_in, data_out_port_in, data_mutex_in)

        self.m_mean_out_port = mean_out_port
        self.m_median_out_port = median_out_port
        self.m_weighted_out_port = weighted_out_port
        self.m_clip_out_port = clip_out_port
        self.m_requirements = requirements

    def run(self):
        """
        Run method of PcaTaskWriter. Writes the residuals to the output ports.

        :return: None
        """

        while True:
            next_result = self.m_result_queue.get()
            poison_pill_case = self.check_poison_pill(next_result)

            if poison_pill_case == 1:
                break

            elif poison_pill_case == 2:
                continue

            with self.m_data_mutex:
                if self.m_requirements[0]:
                    self.m_mean_out_port[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[0, :, :]

                if self.m_requirements[1]:
                    self.m_median_out_port[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[1, :, :]

                if self.m_requirements[2]:
                    self.m_weighted_out_port[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[2, :, :]

                if self.m_requirements[3]:
                    self.m_clip_out_port[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[3, :, :]

            self.m_result_queue.task_done()


class PcaMultiprocessingCapsule(MultiprocessingCapsule):
    """
    Capsule for PCA multiprocessing with the poison pill pattern.
    """

    def __init__(self,
                 mean_out_port,
                 median_out_port,
                 weighted_out_port,
                 clip_out_port,
                 num_processors,
                 pca_numbers,
                 pca_model,
                 star_reshape,
                 angles,
                 im_shape,
                 indices):
        """
        Constructor of PcaMultiprocessingCapsule.

        :param mean_out_port: Output port for the mean residuals.
        :type mean_out_port: PynPoint.Core.DataIO.OutputPort
        :param median_out_port: Output port for the median residuals.
        :type median_out_port: PynPoint.Core.DataIO.OutputPort
        :param weighted_out_port: Output port for the noise-weighted residuals.
        :type weighted_out_port: PynPoint.Core.DataIO.OutputPort
        :param clip_out_port: Output port for the mean clipped residuals.
        :type clip_out_port: PynPoint.Core.DataIO.OutputPort
        :param num_processors: Number of processors.
        :type num_processors: int
        :param pca_numbers: Number of principal components.
        :type pca_numbers: numpy.ndarray
        :param pca_model: PCA object with the basis.
        :type pca_model: sklearn.decomposition.pca.PCA
        :param star_reshape: Reshaped (2D) input images.
        :type star_reshape: numpy.ndarray
        :param angles: Derotation angles (deg).
        :type angles: numpy.ndarray
        :param im_shape: Original shape of the input images.
        :type im_shape: tuple(int, int, int)
        :param indices: Non-masked pixel indices.
        :type indices: numpy.ndarray

        :return: None
        """

        self.m_mean_out_port = mean_out_port
        self.m_median_out_port = median_out_port
        self.m_weighted_out_port = weighted_out_port
        self.m_clip_out_port = clip_out_port
        self.m_pca_numbers = pca_numbers
        self.m_pca_model = pca_model
        self.m_star_reshape = star_reshape
        self.m_angles = angles
        self.m_im_shape = im_shape
        self.m_indices = indices

        self.m_requirements = [False, False, False, False]

        if self.m_mean_out_port is not None:
            self.m_requirements[0] = True

        if self.m_median_out_port is not None:
            self.m_requirements[1] = True

        if self.m_weighted_out_port is not None:
            self.m_requirements[2] = True

        if self.m_clip_out_port is not None:
            self.m_requirements[3] = True

        self.m_requirements = tuple(self.m_requirements)

        super(PcaMultiprocessingCapsule, self).__init__(None, None, num_processors)

    def create_writer(self, image_out_port):
        """
        Method to create an instance of PcaTaskWriter.

        :param image_out_port: Input task.
        :type image_out_port: PynPoint.Util.Multiprocessing.TaskInput

        :return: PCA task writer.
        :rtype: PynPoint.Util.MultiprocessingPCA.PcaTaskWriter
        """

        writer = PcaTaskWriter(self.m_result_queue,
                               self.m_mean_out_port,
                               self.m_median_out_port,
                               self.m_weighted_out_port,
                               self.m_clip_out_port,
                               self.m_data_mutex,
                               self.m_requirements)

        return writer

    def init_creator(self, image_in_port):
        """
        Method to create an instance of PcaTaskCreator.

        :param image_in_port: Input task.
        :type image_in_port: PynPoint.Util.Multiprocessing.TaskInput

        :return: PCA task creator.
        :rtype: PynPoint.Util.MultiprocessingPCA.PcaTaskCreator
        """

        creator = PcaTaskCreator(self.m_tasks_queue,
                                 self.m_num_processors,
                                 self.m_pca_numbers)

        return creator

    def create_processors(self):
        """
        Method to create a list of instances of PcaTaskProcessor.

        :return: PCA task processors.
        :rtype: list of PynPoint.Util.MultiprocessingPCA.PcaTaskProcessor
        """

        processors = []

        for _ in range(self.m_num_processors):
            processors.append(PcaTaskProcessor(self.m_tasks_queue,
                                               self.m_result_queue,
                                               self.m_star_reshape,
                                               self.m_angles,
                                               self.m_pca_model,
                                               self.m_im_shape,
                                               self.m_indices,
                                               self.m_requirements))

        return processors
