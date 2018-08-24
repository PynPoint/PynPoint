"""
Modules for PSF subtraction with principle component analysis.
"""

from copy import deepcopy
from sys import platform, stdout

import numpy as np

from scipy import ndimage
from sklearn.decomposition import PCA

from PynPoint.Util.ModuleTools import progress
from PynPoint.Util.MultiprocessingPCA import PcaMultiprocessingCapsule
from PynPoint.Core.Processing import ProcessingModule


class PcaPsfSubtractionModule(ProcessingModule):
    """
    Module for PSF subtraction with principal component analysis (PCA). The multiprocessing
    implementation is only supported for Linux and Windows. Mac only runs in single processing
    due to a bug in the numpy package. Note that the calculation of the residuals with multi-
    processing may require a large amount of memory in case the stack of input images is very
    large. In that case, CPU could be set to a smaller number (or even 1) in the configuration
    file.
    """

    def __init__(self,
                 pca_numbers,
                 name_in="PSF_subtraction",
                 images_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 res_mean_tag="res_mean",
                 res_median_tag=None,
                 res_arr_out_tag=None,
                 res_rot_mean_clip_tag=None,
                 extra_rot=0.,
                 subtract_mean=True,
                 **kwargs):
        """
        Constructor of PcaPsfSubtractionModule.

        :param pca_numbers: Number of PCA components used for the PSF model. Can be a single value
                            or a list of integers. A list of PCAs will be processed (if supported)
                            using multiprocessing.
        :type pca_numbers: int
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
        :param res_arr_out_tag: Tag of the database entry with the image residuals from the PSF
                                subtraction. If a list of PCs is provided in *pca_numbers* then
                                multiple tags will be created in the central database. Not
                                calculated if set to *None*.
        :type res_arr_out_tag: str
        :param res_rot_mean_clip_tag: Tag of the database entry of the clipped mean residuals. Not
                                      calculated if set to *None*.
        :type res_rot_mean_clip_tag: str
        :param extra_rot: Additional rotation angle of the images (deg).
        :type extra_rot: float
        :param subtract_mean: The mean of the science and reference images is subtracted from
                              the corresponding stack, before the PCA basis is constructed and fitted.
        :type subtract_mean: bool
        :param \**kwargs:
            See below.

        :Keyword arguments:
            **basis_out_tag** (*str*) -- Tag of the database entry with the basis set.

            **verbose** (*bool*) -- Print progress to the standard output.

        :return: None
        """

        super(PcaPsfSubtractionModule, self).__init__(name_in)

        if "verbose" in kwargs:
            self.m_verbose = kwargs["verbose"]
        else:
            self.m_verbose = True

        if "basis_out_tag" in kwargs:
            self.m_basis_out_port = self.add_output_port(kwargs["basis_out_tag"])
        else:
            self.m_basis_out_port = None

        # look for the maximum number of components
        self.m_max_pacs = np.max(pca_numbers)
        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean

        self.m_pca = PCA(n_components=self.m_max_pacs, svd_solver="arpack")

        # add input ports
        self.m_reference_in_port = self.add_input_port(reference_in_tag)
        self.m_star_in_port = self.add_input_port(images_in_tag)

        # add output ports
        if res_mean_tag is None:
            self.m_res_mean_out_port = None
        else:
            self.m_res_mean_out_port = self.add_output_port(res_mean_tag)

        if res_median_tag is None:
            self.m_res_median_out_port = None
        else:
            self.m_res_median_out_port = self.add_output_port(res_median_tag)

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

    def _run_multi_processing(self, star_data):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using multiprocessing.

        :return: None
        """

        tmp_output = np.zeros((len(self.m_components), star_data.shape[1], star_data.shape[2]))

        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.set_all(tmp_output, keep_attributes=False)

        cpu = self._m_config_port.get_attribute("CPU")

        rotations = -1.*self.m_star_in_port.get_attribute("PARANG")
        rotations += np.ones(rotations.shape[0]) * self.m_extra_rot

        pca_capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                                self.m_res_median_out_port,
                                                self.m_res_rot_mean_clip_out_port,
                                                cpu,
                                                deepcopy(self.m_components),
                                                deepcopy(self.m_pca),
                                                deepcopy(star_data),
                                                deepcopy(rotations))

        pca_capsule.run()

    def _run_single_processing(self, star_sklearn, star_data):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.

        :return: None
        """

        for i, pca_number in enumerate(self.m_components):

            if self.m_verbose:
                progress(i, len(self.m_components), "Creating residuals...")

            tmp_pca_representation = np.matmul(self.m_pca.components_[:pca_number],
                                               star_sklearn.T)

            tmp_pca_representation = np.vstack((tmp_pca_representation,
                                                np.zeros((self.m_max_pacs - pca_number,
                                                          star_data.shape[0])))).T

            tmp_psf_images = self.m_pca.inverse_transform(tmp_pca_representation)
            tmp_psf_images = tmp_psf_images.reshape((star_data.shape[0],
                                                     star_data.shape[1],
                                                     star_data.shape[2]))

            # subtract the psf model of the star
            tmp_without_psf = star_data - tmp_psf_images

            # inverse rotation
            delta_para = -1.*self.m_star_in_port.get_attribute("PARANG")
            res_array = np.zeros(shape=tmp_without_psf.shape)
            for j, angle in enumerate(delta_para):
                res_temp = tmp_without_psf[j, ]
                # ndimage.rotate rotates in clockwise direction for positive angles
                res_array[j, ] = ndimage.rotate(res_temp, angle+self.m_extra_rot, reshape=False)

            # create residuals
            # 1.) The de-rotated result images
            if self.m_res_arr_out_ports is not None:
                self.m_res_arr_out_ports[pca_number].set_all(res_array)
                self.m_res_arr_out_ports[pca_number].copy_attributes_from_input_port(
                    self.m_star_in_port)
                self.m_res_arr_out_ports[pca_number].add_history_information("PSF subtraction",
                                                                             "PCA")

            # 2.) mean
            if self.m_res_mean_out_port is not None:
                tmp_res_rot_mean = np.mean(res_array, axis=0)
                self.m_res_mean_out_port.append(tmp_res_rot_mean, data_dim=3)

            # 3.) median
            if self.m_res_median_out_port is not None:
                tmp_res_rot_median = np.median(res_array, axis=0)
                self.m_res_median_out_port.append(tmp_res_rot_median, data_dim=3)

            # 4.) clipped mean
            if self.m_res_rot_mean_clip_out_port is not None:
                res_rot_temp = res_array.copy()

                if self.m_res_mean_out_port is None:
                    tmp_res_rot_mean = np.mean(res_array, axis=0)

                for j in range(res_rot_temp.shape[0]):
                    res_rot_temp[j, ] -= -tmp_res_rot_mean

                res_rot_var = (res_rot_temp**2.).sum(axis=0)
                tmp_res_rot_var = res_rot_var

                self.m_res_rot_mean_clip_out_port.append(tmp_res_rot_var, data_dim=3)

        if self.m_verbose:
            stdout.write("Creating residuals... [DONE]\n")
            stdout.flush()

    def _clear_output_ports(self):
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.del_all_data()
            self.m_res_mean_out_port.del_all_attributes()

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.del_all_data()
            self.m_res_median_out_port.del_all_attributes()

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

        self._clear_output_ports()

        # get all data
        star_data = self.m_star_in_port.get_all()

        if self.m_reference_in_port.tag == self.m_star_in_port.tag:
            ref_star_data = deepcopy(star_data)

        else:
            ref_star_data = self.m_reference_in_port.get_all()

        # subtract mean from science data, if required
        if self.m_subtract_mean:
            mean_star = np.mean(star_data, axis=0)
            star_data -= mean_star

        # subtract mean from reference data
        mean_ref_star = np.mean(ref_star_data, axis=0)
        ref_star_data -= mean_ref_star

        # Fit the PCA model
        if self.m_verbose:
            stdout.write("Constructing PSF model...")
            stdout.flush()

        ref_star_sklearn = ref_star_data.reshape((ref_star_data.shape[0],
                                                  ref_star_data.shape[1] * ref_star_data.shape[2]))
        self.m_pca.fit(ref_star_sklearn)

        # add mean of reference array as first principal component and orthogonalize it wrt the PCA basis
        if not self.m_subtract_mean:
            mean_ref_star_sklearn = mean_ref_star.reshape((1,
                                                           ref_star_data.shape[1] * ref_star_data.shape[2]))
            
            Q, _ = np.linalg.qr(np.vstack((mean_ref_star_sklearn, self.m_pca.components_[:-1,])).T)
            self.m_pca.components_ = Q.T

        if self.m_verbose:
            stdout.write(" [DONE]\n")
            stdout.flush()

        if self.m_basis_out_port is not None:
            basis = self.m_pca.components_.reshape((self.m_pca.components_.shape[0],
                                                    star_data.shape[1], star_data.shape[2]))
            self.m_basis_out_port.set_all(basis)

        # prepare the data for sklearns PCA
        star_sklearn = star_data.reshape((star_data.shape[0],
                                          star_data.shape[1] * star_data.shape[2]))

        cpu = self._m_config_port.get_attribute("CPU")

        # multiprocessing crashed on Mac in combination with numpy
        if platform == "darwin" or self.m_res_arr_out_ports is not None or cpu == 1:
            self._run_single_processing(star_sklearn, star_data)

        else:
            if self.m_verbose:
                stdout.write("Creating residuals...")
                stdout.flush()

            self._run_multi_processing(star_data)

            if self.m_verbose:
                stdout.write(" [DONE]\n")
                stdout.flush()

        # save history for all other ports
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_mean_out_port.add_history_information("PSF subtraction", "PCA")

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_median_out_port.add_history_information("PSF subtraction", "PCA")

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_rot_mean_clip_out_port.add_history_information("PSF subtraction", "PCA")

        self.m_star_in_port.close_port()
