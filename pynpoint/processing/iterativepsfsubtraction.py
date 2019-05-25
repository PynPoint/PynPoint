"""
Modules for  iterative PSF subtraction.
"""

from __future__ import absolute_import

import sys

from copy import deepcopy

import numpy as np

from sklearn.decomposition import PCA

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.multipca import PcaMultiprocessingCapsule
from pynpoint.util.psf import pca_psf_subtraction, iterative_pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals

class IterativePcaPsfSubtractionModule(ProcessingModule):
    """
    Module for PSF subtraction with an extention of principal component analysis (PCA),
    done by iteratively removing the residual image from the initial data cube, as
    described in Pairet et al 2018 "Referenceless algorithm for circumstellar disk imaging."
    The residuals are calculated in parallel for the selected numbers of principal components.
     This may require a large amount of memory in case the stack of input images is very large.
    The number of processes can be set with the CPU keyword in the configuration file.
    """

    def __init__(self,
                 pca_numbers,
                 pca_number_init,
                 interval=1,
                 name_in="it_psf_subtraction",
                 images_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 res_mean_tag="res_mean",
                 res_median_tag=None,
                 res_weighted_tag=None,
                 res_rot_mean_clip_tag=None,
                 res_arr_out_tag=None,
                 extra_rot=0.,
                 subtract_mean=True):
        """
        Constructor of PcaPsfSubtractionModule.

        :param pca_numbers: Number of principal components used for the PSF model. Can be a single
                            value or a list of integers.
        :type pca_numbers: int
        :param pca_number_init: Number of initial principal component, where the iteration is commenced. 
                                Should be a single value.
        :type pca_number_init: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param images_in_tag: Tag of the database entry with the science images that are read
                              as input.
        :type images_in_tag: str
        :param reference_in_tag: Tag of the database entry with the reference images that are
                                 read as input.
        :type reference_in_tag: str
        :param res_mean_tag: Tag of the database entry with the mean collapsed residuals. Not
                             calculated if set to *None*.
        :type res_mean_tag: str
        :param res_median_tag: Tag of the database entry with the median collapsed residuals. Not
                               calculated if set to *None*.
        :type res_median_tag: str
        :param res_weighted_tag: Tag of the database entry with the noise-weighted residuals
                                 (see Bottom et al. 2017). Not calculated if set to *None*.
        :type res_weighted_tag: str
        :param res_rot_mean_clip_tag: Tag of the database entry of the clipped mean residuals. Not
                                      calculated if set to *None*.
        :type res_rot_mean_clip_tag: str
        :param res_arr_out_tag: Tag of the database entry with the image residuals from the PSF
                                subtraction. If a list of PCs is provided in *pca_numbers* then
                                multiple tags will be created in the central database. Not
                                calculated if set to *None*. Not supported with multiprocessing.
        :type res_arr_out_tag: str
        :param extra_rot: Additional rotation angle of the images (deg).
        :type extra_rot: float
        :param subtract_mean: The mean of the science and reference images is subtracted from
                              the corresponding stack, before the PCA basis is constructed and
                              fitted.
        :type subtract_mean: bool

        :return: None
        """
        super(IterativePcaPsfSubtractionModule, self).__init__(name_in)
        self.m_pca_number_init = pca_number_init        
        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean

        #self.m_pca = PCA(n_components=np.amax(self.m_components), svd_solver="arpack")

        self.m_reference_in_port = self.add_input_port(reference_in_tag)
        self.m_star_in_port = self.add_input_port(images_in_tag)

        if res_mean_tag is None:
            self.m_res_mean_out_port = None
        else:
            self.m_res_mean_out_port = self.add_output_port(res_mean_tag)

        if res_median_tag is None:
            self.m_res_median_out_port = None
        else:
            self.m_res_median_out_port = self.add_output_port(res_median_tag)

        if res_weighted_tag is None:
            self.m_res_weighted_out_port = None
        else:
            self.m_res_weighted_out_port = self.add_output_port(res_weighted_tag)

        if res_rot_mean_clip_tag is None:
            self.m_res_rot_mean_clip_out_port = None
        else:
            self.m_res_rot_mean_clip_out_port = self.add_output_port(res_rot_mean_clip_tag)

        if res_arr_out_tag is None:
            self.m_res_arr_out_ports = None
        else:
            self.m_res_arr_out_ports = {}
            for pca_number in self.m_components:
                self.m_res_arr_out_ports[pca_number] = self.add_output_port(res_arr_out_tag
                                                                            +str(pca_number))


    def _run_single_processing(self, star_reshape, im_shape, indices):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.

        :return: None
        """

        parang = -1.*self.m_star_in_port.get_attribute("PARANG") + self.m_extra_rot

        pca_number_init = self.m_pca_number_init
        
        pca_numbers = self.m_components
        
        residuals_list, res_rot_list = iterative_pca_psf_subtraction(images=star_reshape,
                                                     angles=parang,
                                                     pca_numbers=pca_numbers,
                                                     pca_number_init=pca_number_init,
                                                     indices=indices)

        for i, res_rot in enumerate(res_rot_list):
            #progress(i, len(self.m_components), "Creating residuals...")

            history = "max iter PC number = "+str(np.amax(self.m_components))

            # 1.) derotated residuals
            if self.m_res_arr_out_ports is not None:
                self.m_res_arr_out_ports[pca_numbers[i]].set_all(res_rot)
                self.m_res_arr_out_ports[pca_numbers[i]].copy_attributes_from_input_port(
                    self.m_star_in_port)
                self.m_res_arr_out_ports[pca_numbers[i]].add_history_information( \
                    "IterativePcaPsfSubtractionModule", history)

            # 2.) mean residuals
            if self.m_res_mean_out_port is not None:
                stack = combine_residuals(method="mean", res_rot=res_rot)
                self.m_res_mean_out_port.append(stack, data_dim=3)

            # 3.) median residuals
            if self.m_res_median_out_port is not None:
                stack = combine_residuals(method="median", res_rot=res_rot)
                self.m_res_median_out_port.append(stack, data_dim=3)

            # 4.) noise-weighted residuals
            if self.m_res_weighted_out_port is not None:
                stack = combine_residuals(method="weighted",
                                          res_rot=res_rot,
                                          residuals=residuals_list[i],
                                          angles=parang)

                self.m_res_weighted_out_port.append(stack, data_dim=3)

            # 5.) clipped mean residuals
            if self.m_res_rot_mean_clip_out_port is not None:
                stack = combine_residuals(method="clipped", res_rot=res_rot)
                self.m_res_rot_mean_clip_out_port.append(stack, data_dim=3)

        sys.stdout.write("Creating residuals... [DONE]\n")
        sys.stdout.flush()

    def _clear_output_ports(self):
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.del_all_data()
            self.m_res_mean_out_port.del_all_attributes()

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.del_all_data()
            self.m_res_median_out_port.del_all_attributes()

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.del_all_data()
            self.m_res_weighted_out_port.del_all_attributes()

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.del_all_data()
            self.m_res_rot_mean_clip_out_port.del_all_attributes()

        if self.m_res_arr_out_ports is not None:
            for pca_number in self.m_components:
                self.m_res_arr_out_ports[pca_number].del_all_data()
                self.m_res_arr_out_ports[pca_number].del_all_attributes()

    def run(self):
        """
        Run method of the module. Subtracts the mean of the image stack from all images, reshapes
        the stack of images into a 2D array, uses singular value decomposition to construct the
        orthogonal basis set, calculates the PCA coefficients for each image, subtracts the PSF
        model, and writes the residuals as output.

        :return: None
        """

        cpu = self._m_config_port.get_attribute("CPU")

        if cpu > 1 and self.m_res_arr_out_ports is not None:
            warnings.warn("Multiprocessing not possible if 'res_arr_out_tag' is not set to None.")

        self._clear_output_ports()

        # get all data
        star_data = self.m_star_in_port.get_all()
        im_shape = star_data.shape

        # select the first image and get the unmasked image indices
        im_star = star_data[0, ].reshape(-1)
        indices = np.where(im_star != 0.)[0]

        # reshape the star data and select the unmasked pixels
        star_reshape = star_data.reshape(im_shape[0], im_shape[1]*im_shape[2])
        star_reshape = star_reshape[:, indices]

        if self.m_reference_in_port.tag == self.m_star_in_port.tag:
            ref_reshape = deepcopy(star_reshape)

        else:
            ref_data = self.m_reference_in_port.get_all()
            ref_shape = ref_data.shape

            if ref_shape[-2:] != im_shape[-2:]:
                raise ValueError("The image size of the science data and the reference data "
                                 "should be identical.")

            # reshape reference data and select the unmasked pixels
            ref_reshape = ref_data.reshape(ref_shape[0], ref_shape[1]*ref_shape[2])
            ref_reshape = ref_reshape[:, indices]

        # subtract mean from science data, if required
        if self.m_subtract_mean:
            mean_star = np.mean(star_reshape, axis=0)
            star_reshape -= mean_star

        # subtract mean from reference data
        mean_ref = np.mean(ref_reshape, axis=0)
        ref_reshape -= mean_ref

        # create the PCA basis
        sys.stdout.write("Constructing PSF model...")
        sys.stdout.flush()


        # add mean of reference array as 1st PC and orthogonalize it with respect to the PCA basis
        """
        if not self.m_subtract_mean:
            mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

            q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                 self.m_pca.components_[:-1, ])).T)

            self.m_pca.components_ = q_ortho.T
        """
        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()



        sys.stdout.write("Running in single process...")

        self._run_single_processing(star_data, im_shape, indices)
        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        history = "max PC number = "+str(np.amax(self.m_components))

        # save history for all other ports
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_mean_out_port.add_history("IterativePcaPsfSubtractionModule", history)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_median_out_port.add_history("IterativePcaPsfSubtractionModule", history)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_weighted_out_port.add_history("IterativePcaPsfSubtractionModule",
                                                                 history)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_rot_mean_clip_out_port.add_history("IterativePcaPsfSubtractionModule",
                                                                      history)

        self.m_star_in_port.close_port()
