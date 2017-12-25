from Multiprocessing import *
import numpy as np
from sklearn.decomposition import PCA
from scipy import ndimage

class PcaTaskCreator(TaskCreator):

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
            tmp_result_position += 1

            self.m_task_queue.put(TaskInput(pca_number,
                                            (((tmp_result_position, None, None),
                                             (None, None, None),
                                             (None, None, None)),))
                                  )
        self.create_poison_pills()


class PcaTaskProcessor(TaskProcessor):

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

        print 1

        star_sklearn = self.m_star_arr.reshape((self.m_star_arr.shape[0],
                                                self.m_star_arr.shape[1] * self.m_star_arr.shape[2]))

        pca_number = tmp_task.m_input_data

        tmp_pca_representation = np.matmul(self.m_pca_model.components_[:pca_number],
                                           star_sklearn.T)

        tmp_pca_representation = np.vstack((tmp_pca_representation,
                                            np.zeros((self.m_pca_model.n_components - pca_number,
                                                      self.m_star_arr.shape[0])))).T

        print 2

        #tmp_psf_images = np.matmul(tmp_pca_representation[0:200, :], self.m_pca_model.components_[:,0:200])

        tmp_psf_images = self.m_pca_model.inverse_transform(tmp_pca_representation)

        print 3.1

        tmp_psf_images = tmp_psf_images.reshape((self.m_star_arr.shape[0],
                                                 self.m_star_arr.shape[1],
                                                 self.m_star_arr.shape[2]))
        print 3.2

        # subtract the psf model of the star
        tmp_without_psf = self.m_star_arr - tmp_psf_images

        print 3

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
                        a = temp - temp.mean()
                        b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
                        b2 = b1.compress((b1 > (-1.0)*3.0*np.sqrt(a.var())).flat)
                        res_rot_mean_clip[i, j] = temp.mean() + b2.mean()

            residual_output[2, :, :] = res_rot_mean_clip

        # 4.) The de-rotated result images
        if self.m_result_requirements[2]:
            residual_output[3:, :, :] = res_array

        print("Created Residual with " + str(pca_number) + " components")

        return TaskResult(res_array, tmp_task.m_job_parameter[0])


class PcaTaskWriter(TaskWriter):

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
            if next_result is None:
                print "Shutting down writer..."
                self.m_result_queue.task_done()
                break

            print "Start writing row " + str(next_result.m_position)

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
                          for i in xrange(self.m_num_processors)]
        return tmp_processors
