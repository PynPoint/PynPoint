"""
Modules for PSF subtraction with principal component analysis.
"""

import sys
from copy import deepcopy

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
                 res_weighted_tag=None,
                 res_rot_mean_clip_tag=None,
                 res_arr_out_tag=None,
                 basis_out_tag=None,
                 extra_rot=0.,
                 subtract_mean=True):
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
        :param basis_out_tag: Tag of the database entry with the basis set. Not stored if set to
                              None.
        :type basis_out_tag: str
        :param extra_rot: Additional rotation angle of the images (deg).
        :type extra_rot: float
        :param subtract_mean: The mean of the science and reference images is subtracted from
                              the corresponding stack, before the PCA basis is constructed and
                              fitted.
        :type subtract_mean: bool

        :return: None
        """

        super(PcaPsfSubtractionModule, self).__init__(name_in)

        self.m_max_pc = np.max(pca_numbers)
        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean

        self.m_pca = PCA(n_components=self.m_max_pc, svd_solver="arpack")

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

        if basis_out_tag is None:
            self.m_basis_out_port = None
        else:
            self.m_basis_out_port = self.add_output_port(basis_out_tag)

    def _run_multi_processing(self, star_reshape, im_shape, indices):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using multiprocessing.

        :return: None
        """

        tmp_output = np.zeros((len(self.m_components), im_shape[1], im_shape[2]))

        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.set_all(tmp_output, keep_attributes=False)

        cpu = self._m_config_port.get_attribute("CPU")

        rotations = -1.*self.m_star_in_port.get_attribute("PARANG") + self.m_extra_rot

        pca_capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                                self.m_res_median_out_port,
                                                self.m_res_weighted_out_port,
                                                self.m_res_rot_mean_clip_out_port,
                                                cpu,
                                                deepcopy(self.m_components),
                                                deepcopy(self.m_pca),
                                                deepcopy(star_reshape),
                                                deepcopy(rotations),
                                                deepcopy(im_shape),
                                                deepcopy(indices))

        pca_capsule.run()

    def _run_single_processing(self, star_reshape, im_shape, indices):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.

        :return: None
        """

        for i, pc_number in enumerate(self.m_components):
            progress(i, len(self.m_components), "Creating residuals...")

            # create pca representation
            pca_rep = np.matmul(self.m_pca.components_[:pc_number], star_reshape.T)
            pca_rep = np.vstack((pca_rep, np.zeros((self.m_max_pc - pc_number, im_shape[0])))).T

            # create PSF model
            psf_model = self.m_pca.inverse_transform(pca_rep)

            # create original array size
            residuals = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))

            # subtract the psf model
            residuals[:, indices] = star_reshape - psf_model

            # reshape to the original image size
            residuals = residuals.reshape(im_shape)

            # inverse rotation
            parang = -1.*self.m_star_in_port.get_attribute("PARANG")

            res_array = np.zeros(residuals.shape)
            for j, angle in enumerate(parang):
                # ndimage.rotate rotates in clockwise direction for positive angles
                res_array[j, ] = ndimage.rotate(input=residuals[j, ],
                                                angle=angle+self.m_extra_rot,
                                                reshape=False)

            history = "max PC number = "+str(self.m_max_pc)

            # create residuals
            # 1.) derotated residuals
            if self.m_res_arr_out_ports is not None:
                self.m_res_arr_out_ports[pc_number].set_all(res_array)
                self.m_res_arr_out_ports[pc_number].copy_attributes_from_input_port(
                    self.m_star_in_port)
                self.m_res_arr_out_ports[pc_number].add_history_information( \
                    "PcaPsfSubtractionModule", history)

            # 2.) mean
            if self.m_res_mean_out_port is not None:
                tmp_res_rot_mean = np.mean(res_array, axis=0)
                self.m_res_mean_out_port.append(tmp_res_rot_mean, data_dim=3)

            # 3.) median
            if self.m_res_median_out_port is not None:
                tmp_res_rot_median = np.median(res_array, axis=0)
                self.m_res_median_out_port.append(tmp_res_rot_median, data_dim=3)

            # 4.) noise weighted
            if self.m_res_weighted_out_port is not None:
                tmp_res_var = np.var(residuals, axis=0)

                res_repeat = np.repeat(tmp_res_var[np.newaxis, :, :],
                                       repeats=residuals.shape[0],
                                       axis=0)

                res_var = np.zeros(res_repeat.shape)
                for j, angle in enumerate(parang):
                    # ndimage.rotate rotates in clockwise direction for positive angles
                    res_var[j, ] = ndimage.rotate(input=res_repeat[j, ],
                                                  angle=angle+self.m_extra_rot,
                                                  reshape=False)

                weight1 = np.divide(res_array, res_var, out=np.zeros_like(res_var),
                                    where=(np.abs(res_var) > 1e-100) & (res_var != np.nan))

                weight2 = np.divide(1., res_var, out=np.zeros_like(res_var),
                                    where=(np.abs(res_var) > 1e-100) & (res_var != np.nan))

                sum1 = np.sum(weight1, axis=0)
                sum2 = np.sum(weight2, axis=0)

                res_rot_weighted = np.divide(sum1, sum2, out=np.zeros_like(sum2),
                                             where=(np.abs(sum2) > 1e-100) & (sum2 != np.nan))

                self.m_res_weighted_out_port.append(res_rot_weighted, data_dim=3)

            # 5.) clipped mean
            if self.m_res_rot_mean_clip_out_port is not None:
                res_rot_mean_clip = np.zeros(res_array[0, ].shape)

                for j in range(res_rot_mean_clip.shape[0]):
                    for k in range(res_rot_mean_clip.shape[1]):
                        temp = res_array[:, j, k]

                        if temp.var() > 0.0:
                            no_mean = temp - temp.mean()

                            part1 = no_mean.compress((no_mean < 3.0*np.sqrt(no_mean.var())).flat)
                            part2 = part1.compress((part1 > (-1.0)*3.0*np.sqrt(no_mean.var())).flat)

                            res_rot_mean_clip[j, k] = temp.mean() + part2.mean()

                self.m_res_rot_mean_clip_out_port.append(res_rot_mean_clip, data_dim=3)

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
            for pc_number in self.m_components:
                self.m_res_arr_out_ports[pc_number].del_all_data()
                self.m_res_arr_out_ports[pc_number].del_all_attributes()

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

        self.m_pca.fit(ref_reshape)

        # add mean of reference array as 1st PC and orthogonalize it with respect to the PCA basis
        if not self.m_subtract_mean:
            mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

            q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                 self.m_pca.components_[:-1, ])).T)

            self.m_pca.components_ = q_ortho.T

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        if self.m_basis_out_port is not None:
            pc_size = self.m_pca.components_.shape[0]

            basis = np.zeros((pc_size, im_shape[1]*im_shape[2]))
            basis[:, indices] = self.m_pca.components_
            basis = basis.reshape((pc_size, im_shape[1], im_shape[2]))

            self.m_basis_out_port.set_all(basis)

        cpu = self._m_config_port.get_attribute("CPU")

        # multiprocessing crashes on osx due to a bug in numpy
        if sys.platform == "darwin" or self.m_res_arr_out_ports is not None or cpu == 1:
            self._run_single_processing(star_reshape, im_shape, indices)

        else:
            sys.stdout.write("Creating residuals")
            sys.stdout.flush()

            self._run_multi_processing(star_reshape, im_shape, indices)

            sys.stdout.write(" [DONE]\n")
            sys.stdout.flush()

        history = "max PC number = "+str(self.m_max_pc)

        # save history for all other ports
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_mean_out_port.add_history_information("PcaPsfSubtractionModule", history)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_median_out_port.add_history_information("PcaPsfSubtractionModule", history)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_weighted_out_port.add_history_information("PcaPsfSubtractionModule",
                                                                 history)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.copy_attributes_from_input_port(self.m_star_in_port)
            self.m_res_rot_mean_clip_out_port.add_history_information("PcaPsfSubtractionModule",
                                                                      history)

        self.m_star_in_port.close_port()
