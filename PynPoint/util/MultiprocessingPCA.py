"""
Multiprocessing for PCA PSF Subtraction. It is penalization to run multiple pca component
configurations at the same time. The PCA basis is required as input. Note due to missing
functionality in numpy this does not run on Mac.
"""

from PynPoint.util.Multiprocessing import TaskProcessor, TaskCreator, TaskWriter,\
    MultiprocessingCapsule, to_slice, TaskInput, TaskResult
import numpy as np
from scipy import ndimage


class PcaTaskCreator(TaskCreator):
    """
    Task Creator of the PCA multiprocessing. This Creator does not need an input port since the data
    is directly given to the Task Processors. It creates one task for each pca component number
    required.
    """

    def __init__(self,
                 tasks_queue_in,
                 number_of_processors,
                 pca_numbers):
        super(PcaTaskCreator, self).__init__(None,
                                             tasks_queue_in,
                                             None,
                                             number_of_processors)

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
    The Task Processor of the PCA multiprocessing is the core of the parallization. One instance
    of this class will calculate one forward and backward PCA transformation given the pretrained
    sklearn PCA model. It does not get data from the Task Creator but uses its own copy of the
    star data, which is the same independent of the task. Finally it will create the residual:

    * The mean residual: my default
    * The median if result_requirements[0]  = True
    * The clipped mean if result_requirements[1]  = True
    * The non stacked result frames if result_requirements[2]  = True
    (not implemented for multiprocessing yet)
    """

    def __init__(self,
                 tasks_queue_in,
                 result_queue_in,
                 star_arr,
                 angles,
                 pca_model,
                 result_requirements=(False, False, False)):
        """

        :param tasks_queue_in:
        :param result_queue_in:
        :param star_arr:
        :param pca_model:
        :type pca_model: PCA
        """
        super(PcaTaskProcessor, self).__init__(tasks_queue_in,
                                               result_queue_in)
        self.m_star_arr = star_arr
        self.m_pca_model = pca_model
        self.m_angles = angles
        self.m_result_requirements = result_requirements

    def run_job(self, tmp_task):

        star_sklearn = self.m_star_arr.reshape((self.m_star_arr.shape[0],
                                                self.m_star_arr.shape[1] *
                                                self.m_star_arr.shape[2]))

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
        for i in range(0, len(self.m_angles)):
            res_temp = tmp_without_psf[i, ]
            res_array[i, ] = ndimage.rotate(res_temp,
                                            self.m_angles[i],
                                            reshape=False)

        # create residuals
        res_length = 3

        if self.m_result_requirements[2]:
            res_length += res_array.shape[0]

        residual_output = np.zeros((res_length, res_array.shape[1], res_array.shape[2]))

        # 1.) mean
        tmp_res_rot_mean = np.mean(res_array,
                                   axis=0)

        residual_output[0, :, :] = tmp_res_rot_mean

        # 2.) median
        if self.m_result_requirements[0]:
            tmp_res_rot_median = np.median(res_array,
                                           axis=0)
            residual_output[1, :, :] = tmp_res_rot_median

        # 3.) clipped mean
        if self.m_result_requirements[1]:
            res_rot_mean_clip = np.zeros(self.m_star_arr[0, ].shape)

            for i in range(0, res_rot_mean_clip.shape[0]):
                for j in range(0, res_rot_mean_clip.shape[1]):
                    temp = res_array[:, i, j]
                    if temp.var() > 0.0:
                        no_mean = temp - temp.mean()
                        part1 = no_mean.compress((no_mean < 3.0*np.sqrt(no_mean.var())).flat)
                        part2 = part1.compress((part1 > (-1.0)*3.0*np.sqrt(no_mean.var())).flat)
                        res_rot_mean_clip[i, j] = temp.mean() + part2.mean()

            residual_output[2, :, :] = res_rot_mean_clip

        # 4.) The de-rotated result images
        if self.m_result_requirements[2]:
            residual_output[3:, :, :] = res_array

        print "Created Residual with " + str(pca_number) + " components"

        return TaskResult(residual_output, tmp_task.m_job_parameter[0])


class PcaTaskWriter(TaskWriter):
    """
    The Writer of the PCA palatalization uses three different ports to save the results of the
    Task Processors (mean, median, clipped). If they are not reburied they can be None.
    """

    def __init__(self,
                 result_queue_in,
                 mean_out_port_in,
                 median_out_port_in,
                 clip_out_port_in,
                 data_mutex_in,
                 result_requirements=(False, False, False)):
        super(PcaTaskWriter, self).__init__(result_queue_in,
                                            mean_out_port_in,
                                            data_mutex_in)

        self.m_median_out_port_in = median_out_port_in
        self.m_clip_out_port_in = clip_out_port_in
        self.m_result_requirements = result_requirements

    def run(self):
        while True:
            next_result = self.m_result_queue.get()

            # Poison Pill
            poison_pill_case = self.check_poison_pill(next_result)
            if poison_pill_case == 1:
                break
            if poison_pill_case == 2:
                continue

            with self.m_data_mutex:
                self.m_data_out_port[to_slice(next_result.m_position)] = \
                    next_result.m_data_array[0, :, :]
                if self.m_result_requirements[0]:
                    self.m_median_out_port_in[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[1, :, :]
                if self.m_result_requirements[1]:
                    self.m_clip_out_port_in[to_slice(next_result.m_position)] = \
                        next_result.m_data_array[2, :, :]
                if self.m_result_requirements[2]:
                    raise NotImplementedError("not supported yet")

            self.m_result_queue.task_done()


class PcaMultiprocessingCapsule(MultiprocessingCapsule):
    """
    Capsule for PCA multiprocess using the poison pill pattern.
    """

    def __init__(self,
                 mean_out_port,
                 median_out_port,
                 clip_out_port,
                 num_processors,
                 pca_numbers,
                 pca_model,
                 star_arr,
                 rotations,
                 result_requirements=(False, False, False)):

        self.m_mean_out_port = mean_out_port
        self.m_median_out_port = median_out_port
        self.m_clip_out_port = clip_out_port
        self.m_result_requirements = result_requirements
        self.m_pca_numbers = pca_numbers
        self.m_pca_model = pca_model
        self.m_star_arr = star_arr
        self.m_rotations = rotations

        super(PcaMultiprocessingCapsule, self).__init__(None,
                                                        None,
                                                        num_processors)

    def create_writer(self, image_out_port):
        tmp_writer = PcaTaskWriter(self.m_result_queue,
                                   self.m_mean_out_port,
                                   self.m_median_out_port,
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
