from PynPoint.core.Processing import ProcessingModule
import numpy as np
from sklearn.decomposition import PCA
from scipy import ndimage
from copy import deepcopy
from PynPoint.util import PcaMultiprocessingCapsule
from sys import platform


class PSFSubtractionPCA(ProcessingModule):

    def __init__(self,
                 pca_numbers,
                 name_in="PSF_subtraction",
                 images_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 res_mean_tag="res_mean",
                 res_median_tag=None,
                 res_arr_out_tag=None,
                 res_rot_mean_clip_tag=None,
                 extra_rot=0.0):

        super(PSFSubtractionPCA, self).__init__(name_in)

        # look for the maximum number of components
        self.m_max_PCAs = np.max(pca_numbers)
        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot

        self.m_pca = PCA(n_components=self.m_max_PCAs)

        # add input Ports
        self.m_reference_in_port = self.add_input_port(reference_in_tag)
        self.m_star_in_port = self.add_input_port(images_in_tag)

        # create output if none create special not used port
        self.m_res_mean_out_port = self.add_output_port(str(res_mean_tag))

        # None check
        if res_median_tag is None:
            self.m_res_median_out_port = self.add_output_port(str("no median"))
            self.m_res_median_out_port.deactivate()
        else:
            self.m_res_median_out_port = self.add_output_port(str(res_median_tag))

        if res_rot_mean_clip_tag is None:
            self.m_res_rot_mean_clip_out_port = self.add_output_port(str("no clip"))
            self.m_res_rot_mean_clip_out_port.deactivate()
        else:
            self.m_res_rot_mean_clip_out_port = self.add_output_port(str(res_rot_mean_clip_tag))

        # use a dict to store output ports for the non-stacked residuals
        self.m_res_arr_out_ports = {}
        self.m_res_arr_required = True

        for pca_number in self.m_components:
            # (cast to string for None case)
            # if res_arr_out_tag is None we still get different Tag names like None02
            self.m_res_arr_out_ports[pca_number] = self.add_output_port(str(res_arr_out_tag)
                                                                        + str(pca_number))

        # deactivate not needed array out ports
        if res_arr_out_tag is None:
            self.m_res_arr_required = False
            for port in self.m_res_arr_out_ports.itervalues():
                port.deactivate()

    def _run_multi_processing(self, star_data):
        # do the fit and write out the result
        # clear all result ports
        tmp_output = np.zeros((len(self.m_components),
                               star_data.shape[1],
                               star_data.shape[2]))

        self.m_res_mean_out_port.set_all(tmp_output, keep_attributes=False)
        self.m_res_median_out_port.set_all(tmp_output, keep_attributes=False)
        self.m_res_rot_mean_clip_out_port.set_all(tmp_output, keep_attributes=False)

        print("Start calculating PSF models")

        rotations = - self.m_star_in_port.get_attribute("NEW_PARA")
        rotations += np.ones(rotations.shape[0]) * self.m_extra_rot

        pca_capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                                self.m_res_median_out_port,
                                                self.m_res_rot_mean_clip_out_port,
                                                10,
                                                deepcopy(self.m_components),
                                                deepcopy(self.m_pca),
                                                deepcopy(star_data),
                                                deepcopy(rotations),
                                                result_requirements=(False, False, False))
        pca_capsule.run()

    def _run_single_processing(self, star_sklearn):
        # do the fit and write out the result
        # clear all result ports
        self.m_res_mean_out_port.del_all_data()
        self.m_res_median_out_port.del_all_data()
        self.m_res_rot_mean_clip_out_port.del_all_data()

        print("Start calculating PSF models")
        history = "Using PCAs"
        for pca_number in self.m_components:
            tmp_pca_representation = np.matmul(self.m_pca.components_[:pca_number],
                                               star_sklearn.T)

            tmp_pca_representation = np.vstack((tmp_pca_representation,
                                                np.zeros((self.m_max_PCAs - pca_number,
                                                          star_data.shape[0])))).T

            tmp_psf_images = self.m_pca.inverse_transform(tmp_pca_representation)
            tmp_psf_images = tmp_psf_images.reshape((star_data.shape[0],
                                                     star_data.shape[1],
                                                     star_data.shape[2]))

            # subtract the psf model of the star
            tmp_without_psf = star_data - tmp_psf_images

            # inverse rotation
            delta_para = - self.m_star_in_port.get_attribute("NEW_PARA")
            res_array = np.zeros(shape=tmp_without_psf.shape)
            for i in range(0, len(delta_para)):
                res_temp = tmp_without_psf[i,]
                res_array[i,] = ndimage.rotate(res_temp,
                                               delta_para[i] + self.m_extra_rot,
                                               reshape=False)

            # create residuals
            # 1.) The de-rotated result images
            if self.m_res_arr_required:
                self.m_res_arr_out_ports[pca_number].set_all(res_array)
                self.m_res_arr_out_ports[pca_number].copy_attributes_from_input_port(
                    self.m_star_in_port)
                self.m_res_arr_out_ports[pca_number].add_history_information("PSF_subtraction",
                                                                             history)

            # 2.) mean
            tmp_res_rot_mean = np.mean(res_array,
                                       axis=0)

            self.m_res_mean_out_port.append(tmp_res_rot_mean, data_dim=3)

            # 3.) median
            if self.m_res_median_out_port.m_activate:
                tmp_res_rot_median = np.median(res_array,
                                               axis=0)
                self.m_res_median_out_port.append(tmp_res_rot_median, data_dim=3)

            # 4.) clipped mean
            if self.m_res_rot_mean_clip_out_port.m_activate:
                res_rot_temp = res_array.copy()
                for i in range(0,
                               res_rot_temp.shape[0]):
                    res_rot_temp[i,] -= - tmp_res_rot_mean
                res_rot_var = (res_rot_temp ** 2.).sum(axis=0)
                tmp_res_rot_var = res_rot_var

                self.m_res_rot_mean_clip_out_port.append(tmp_res_rot_var, data_dim=3)

            print("Created Residual with " + str(pca_number) + " components")

    def run(self):
        # get all data and subtract the mean
        star_data = self.m_star_in_port.get_all()
        mean_star = np.mean(star_data, axis=0)
        star_data -= mean_star

        if self.m_reference_in_port.tag == self.m_star_in_port.tag:
            ref_star_data = deepcopy(star_data)

        else:
            ref_star_data = self.m_reference_in_port.get_all()
            mean_ref_star = np.mean(ref_star_data, axis=0)
            ref_star_data -= mean_ref_star

        # Fit the PCA model
        print("Start fitting the PCA model ...")
        ref_star_sklearn = star_data.reshape((ref_star_data.shape[0],
                                              ref_star_data.shape[1] * ref_star_data.shape[2]))
        self.m_pca.fit(ref_star_sklearn)

        # prepare the data for sklearns PCA
        star_sklearn = star_data.reshape((star_data.shape[0],
                                          star_data.shape[1] * star_data.shape[2]))

        # multiprocessing crashed on Mac in combination with numpy
        if platform == "darwin":
            self._run_single_processing(star_sklearn)
        else:
            self._run_multi_processing(star_data)

        # save history for all other ports
        self.m_res_mean_out_port.copy_attributes_from_input_port(self.m_star_in_port)
        self.m_res_median_out_port.copy_attributes_from_input_port(self.m_star_in_port)
        self.m_res_rot_mean_clip_out_port.copy_attributes_from_input_port(self.m_star_in_port)

        self.m_res_mean_out_port.add_history_information("PSF_subtraction", "Using PCAs")
        self.m_res_median_out_port.add_history_information("PSF_subtraction", "Using PCAs")
        self.m_res_rot_mean_clip_out_port.add_history_information("PSF_subtraction", "Using PCAs")

        self.m_res_mean_out_port.close_port()
