"""
Capsule for multiprocessing of the PCA-based PSF subtraction. Residuals are created in parallel for
a range of principle components. The PCA basis is required as input. Note that due to a missing
functionality in numpy the multiprocessing does not run on macOS.
"""

import sys

import numpy as np
from scipy import ndimage

from PynPoint.Util.Multiprocessing import TaskProcessor, TaskCreator, TaskWriter, TaskResult, \
                                          TaskInput, MultiprocessingCapsule, to_slice


class PcaTaskCreator(TaskCreator):
    """
    Task Creator of the PCA multiprocessing. This Creator does not need an input port since the data
    is directly given to the Task Processors. It creates one task for each PCA component number
    required.
    """

    def __init__(self,
                 tasks_queue_in,
                 number_of_processors,
                 pca_numbers):
        """
        Constructor of PcaTaskCreator.

        :param tasks_queue_in:
        :type tasks_queue_in:
        :param number_of_processors:
        :type number_of_processors:
        :param pca_numbers:
        :type pca_numbers:

        :return: None
        """

        super(PcaTaskCreator, self).__init__(None, tasks_queue_in, None, number_of_processors)

        self.m_pca_numbers = pca_numbers

    def run(self):

        tmp_result_position = 0

        for pca_number in self.m_pca_numbers:

            self.m_task_queue.put(TaskInput(pca_number,
                                            (((tmp_result_position, tmp_result_position+1, None),
                                              (None, None, None),
                                              (None, None, None)),)))

            tmp_result_position += 1

        self.create_poison_pills()


class PcaTaskProcessor(TaskProcessor):
    """
    The TaskProcessor of the PCA multiprocessing is the core of the parallelization. One instance
    of this class will calculate one forward and backward PCA transformation given the pre-trained
    scikit-learn PCA model. It does not get data from the TaskCreator but uses its own copy of the
    star data, which are the same and independent for each task. Finally the residuals are created:

    * Mean of the residuals -- result_requirements[0] = True
    * Median of the residuals -- result_requirements[1] = True
    * Noise-weighted residuals -- result_requirements[2] = True
    * Clipped mean of the residuals -- result_requirements[3] = True
    * Non-stacked residuals -- result_requirements[4] = True (not implemented for multiprocessing)
    """

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in,
                 star_arr,
                 angles,
                 pca_model,
                 result_requirements=(False, False, False, False)):
        """
        Constructor of PcaTaskProcessor.

        :param tasks_queue_in:
        :type tasks_queue_in:
        :param result_queue_in:
        :type result_queue_in:
        :param angles:
        :type angles:
        :param star_arr:
        :type star_arr:
        :param pca_model:
        :type pca_model: PCA
        :param result_requirements:
        :type result_queue_innts:

        :return: None
        """

        super(PcaTaskProcessor, self).__init__(tasks_queue_in, result_queue_in)

        self.m_star_arr = star_arr
        self.m_pca_model = pca_model
        self.m_angles = angles
        self.m_result_requirements = result_requirements

    def run_job(self, tmp_task):

        star_sklearn = self.m_star_arr.reshape((self.m_star_arr.shape[0],
                                                self.m_star_arr.shape[1]*self.m_star_arr.shape[2]))

        pca_number = tmp_task.m_input_data

        tmp_pca_representation = np.matmul(self.m_pca_model.components_[:pca_number],
                                           star_sklearn.T)

        tmp_pca_representation = np.vstack((tmp_pca_representation,
                                            np.zeros((self.m_pca_model.n_components - pca_number,
                                                      self.m_star_arr.shape[0])))).T

        tmp_psf_images = self.m_pca_model.inverse_transform(tmp_pca_representation)

        tmp_psf_images = tmp_psf_images.reshape((self.m_star_arr.shape[0],
                                                 self.m_star_arr.shape[1],
                                                 self.m_star_arr.shape[2]))

        # subtract the psf model of the star
        tmp_without_psf = self.m_star_arr - tmp_psf_images

        # inverse rotation
        res_array = np.zeros(shape=tmp_without_psf.shape)
        for i, angle in enumerate(self.m_angles):
            res_temp = tmp_without_psf[i, ]
            res_array[i, ] = ndimage.rotate(input=res_temp,
                                            angle=angle,
                                            reshape=False)

        # create residuals
        res_length = 4

        # if self.m_result_requirements[3]:
        #     res_length += res_array.shape[0]

        residual_output = np.zeros((res_length, res_array.shape[1], res_array.shape[2]))

        # 1.) mean
        if self.m_result_requirements[0]:
            tmp_res_rot_mean = np.mean(res_array, axis=0)
            residual_output[0, ] = tmp_res_rot_mean

        # 2.) median
        if self.m_result_requirements[1]:
            tmp_res_rot_median = np.median(res_array, axis=0)
            residual_output[1, ] = tmp_res_rot_median

        # 3.) noise weighted
        if self.m_result_requirements[2]:
            tmp_res_var = np.var(tmp_without_psf, axis=0)

            res_repeat = np.repeat(tmp_res_var[np.newaxis, :, :],
                                   repeats=tmp_without_psf.shape[0],
                                   axis=0)

            res_var = np.zeros(res_repeat.shape)
            for j, angle in enumerate(self.m_angles):
                # ndimage.rotate rotates in clockwise direction for positive angles
                res_var[j, ] = ndimage.rotate(input=res_repeat[j, ],
                                              angle=angle,
                                              reshape=False)

            weight1 = np.divide(res_array, res_var, out=np.zeros_like(res_var),
                                where=(np.abs(res_var) > 1e-100) & (res_var != np.nan))

            weight2 = np.divide(1., res_var, out=np.zeros_like(res_var),
                                where=(np.abs(res_var) > 1e-100) & (res_var != np.nan))

            sum1 = np.sum(weight1, axis=0)
            sum2 = np.sum(weight2, axis=0)

            residual_output[2, ] = np.divide(sum1, sum2, out=np.zeros_like(sum2),
                                             where=(np.abs(sum2) > 1e-100) & (sum2 != np.nan))

        # 4.) clipped mean
        if self.m_result_requirements[3]:
            res_rot_mean_clip = np.zeros(self.m_star_arr[0, ].shape)

            for i in range(res_rot_mean_clip.shape[0]):
                for j in range(res_rot_mean_clip.shape[1]):
                    temp = res_array[:, i, j]

                    if temp.var() > 0.0:
                        no_mean = temp - temp.mean()

                        part1 = no_mean.compress((no_mean < 3.0*np.sqrt(no_mean.var())).flat)
                        part2 = part1.compress((part1 > (-1.0)*3.0*np.sqrt(no_mean.var())).flat)

                        res_rot_mean_clip[i, j] = temp.mean() + part2.mean()

            residual_output[3, ] = res_rot_mean_clip

        # 5.) The de-rotated result images
        # if self.m_result_requirements[4]:
        #     residual_output[4:, :, :] = res_array

        sys.stdout.write('.')
        sys.stdout.flush()

        return TaskResult(residual_output, tmp_task.m_job_parameter[0])


class PcaTaskWriter(TaskWriter):
    """
    The Writer of the PCA parallelization uses three different ports to save the results of the
    Task Processors (mean, median, clipped). If they are not reburied they can be None.
    """

    def __init__(self,
                 result_queue_in,
                 mean_out_port_in,
                 median_out_port_in,
                 weighted_out_port_in,
                 clip_out_port_in,
                 data_mutex_in,
                 result_requirements=(False, False, False)):
        """
        Constructor of PcaTaskWriter.

        :param result_queue_in:
        :type result_queue_in:
        :param mean_out_port:
        :type mean_out_port:
        :param median_out_port:
        :type median_out_port:
        :param weighted_out_port:
        :type weighted_out_port:
        :param clip_out_port:
        :type clip_out_port:
        :param data_mutex_in:
        :type data_mutex_in:
        :param result_requirements:
        :type result_requirements:

        :return: None
        """

        super(PcaTaskWriter, self).__init__(result_queue_in,
                                            mean_out_port_in,
                                            data_mutex_in)

        self.m_median_out_port_in = median_out_port_in
        self.m_clip_out_port_in = clip_out_port_in
        self.m_result_requirements = result_requirements

    def run(self):

        while True:
            next_result = self.m_result_queue.get()
            poison_pill_case = self.check_poison_pill(next_result)

            if poison_pill_case == 1:
                break
            if poison_pill_case == 2:
                continue

            with self.m_data_mutex:
                if self.m_result_requirements[0]:
                    self.m_data_out_port[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[0, :, :]

                if self.m_result_requirements[1]:
                    self.m_median_out_port_in[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[1, :, :]

                if self.m_result_requirements[2]:
                    self.m_clip_out_port_in[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[2, :, :]

                if self.m_result_requirements[3]:
                    self.m_clip_out_port_in[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[3, :, :]

                # if self.m_result_requirements[4]:
                #     raise NotImplementedError("Not yet supported.")

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
                 star_arr,
                 rotations):
        """
        Constructor of PcaMultiprocessingCapsule.

        :param mean_out_port:
        :type mean_out_port:
        :param median_out_port:
        :type median_out_port:
        :param weighted_out_port:
        :type weighted_out_port:
        :param clip_out_port:
        :type clip_out_port:
        :param num_processors:
        :type num_processors:
        :param pca_numbers:
        :type pca_numbers:
        :param star_arr:
        :type star_arr:
        :param rotations:
        :type rotations:

        :return: None
        """

        self.m_mean_out_port = mean_out_port
        self.m_median_out_port = median_out_port
        self.m_weighted_out_port = weighted_out_port
        self.m_clip_out_port = clip_out_port
        self.m_pca_numbers = pca_numbers
        self.m_pca_model = pca_model
        self.m_star_arr = star_arr
        self.m_rotations = rotations

        self.m_result_requirements = [False, False, False, False]

        if self.m_mean_out_port is not None:
            self.m_result_requirements[0] = True

        if self.m_median_out_port is not None:
            self.m_result_requirements[1] = True

        if self.m_weighted_out_port is not None:
            self.m_result_requirements[2] = True

        if self.m_clip_out_port is not None:
            self.m_result_requirements[3] = True

        super(PcaMultiprocessingCapsule, self).__init__(None, None, num_processors)

    def create_writer(self, image_out_port):

        tmp_writer = PcaTaskWriter(self.m_result_queue,
                                   self.m_mean_out_port,
                                   self.m_median_out_port,
                                   self.m_weighted_out_port,
                                   self.m_clip_out_port,
                                   self.m_data_mutex,
                                   self.m_result_requirements)

        return tmp_writer

    def init_creator(self, image_in_port):

        tmp_creator = PcaTaskCreator(self.m_tasks_queue,
                                     self.m_num_processors,
                                     self.m_pca_numbers)

        return tmp_creator

    def create_processors(self):

        tmp_processors = [PcaTaskProcessor(self.m_tasks_queue,
                                           self.m_result_queue,
                                           self.m_star_arr,
                                           self.m_rotations,
                                           self.m_pca_model,
                                           self.m_result_requirements)

                          for _ in xrange(self.m_num_processors)]

        return tmp_processors
