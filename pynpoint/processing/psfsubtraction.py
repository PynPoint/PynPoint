"""
Pipeline modules for PSF subtraction.
"""

import sys
import time
import math
import warnings

from copy import deepcopy

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.multipca import PcaMultiprocessingCapsule
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


class PcaPsfSubtractionModule(ProcessingModule):
    """
    Pipeline module for PSF subtraction with principal component analysis (PCA). The residuals are
    calculated in parallel for the selected numbers of principal components. This may require
    a large amount of memory in case the stack of input images is very large. The number of
    processes can be set with the CPU keyword in the configuration file.
    """

    def __init__(self,
                 pca_numbers,
                 name_in='psf_subtraction',
                 images_in_tag='im_arr',
                 reference_in_tag='ref_arr',
                 res_mean_tag=None,
                 res_median_tag=None,
                 res_weighted_tag=None,
                 res_rot_mean_clip_tag=None,
                 res_arr_out_tag=None,
                 basis_out_tag=None,
                 extra_rot=0.,
                 subtract_mean=True):
        """
        Parameters
        ----------
        pca_numbers : list(int, ), tuple(int, ), or numpy.ndarray
            Number of principal components used for the PSF model. Can be a single value or a tuple
            with integers.
        name_in : str
            Unique name of the module instance.
        images_in_tag : str
            Tag of the database entry with the science images that are read as input
        reference_in_tag : str
            Tag of the database entry with the reference images that are read as input.
        res_mean_tag : str
            Tag of the database entry with the mean collapsed residuals. Not calculated if set to
            None.
        res_median_tag : str
            Tag of the database entry with the median collapsed residuals. Not calculated if set
            to None.
        res_weighted_tag : str
            Tag of the database entry with the noise-weighted residuals (see Bottom et al. 2017).
            Not calculated if set to None.
        res_rot_mean_clip_tag : str
            Tag of the database entry of the clipped mean residuals. Not calculated if set to
            None.
        res_arr_out_tag : str
            Tag of the database entry with the image residuals from the PSF subtraction. If a list
            of PCs is provided in *pca_numbers* then multiple tags will be created in the central
            database. Not calculated if set to None. Not supported with multiprocessing.
        basis_out_tag : str
            Tag of the database entry with the basis set. Not stored if set to None.
        extra_rot : float
            Additional rotation angle of the images (deg).
        subtract_mean : bool
            The mean of the science and reference images is subtracted from the corresponding
            stack, before the PCA basis is constructed and fitted.

        Returns
        -------
        NoneType
            None
        """

        super(PcaPsfSubtractionModule, self).__init__(name_in)

        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean

        self.m_pca = PCA(n_components=np.amax(self.m_components), svd_solver='arpack')

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

        Returns
        -------
        NoneType
            None
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

        angles = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

        capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                            self.m_res_median_out_port,
                                            self.m_res_weighted_out_port,
                                            self.m_res_rot_mean_clip_out_port,
                                            self._m_config_port.get_attribute('CPU'),
                                            deepcopy(self.m_components),
                                            deepcopy(self.m_pca),
                                            deepcopy(star_reshape),
                                            deepcopy(angles),
                                            im_shape,
                                            indices)

        capsule.run()

    def _run_single_processing(self, star_reshape, im_shape, indices):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.

        Returns
        -------
        NoneType
            None
        """
        start_time = time.time()
        for i, pca_number in enumerate(self.m_components):
            progress(i, len(self.m_components), 'Creating residuals...', start_time)

            parang = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

            residuals, res_rot = pca_psf_subtraction(images=star_reshape,
                                                     angles=parang,
                                                     pca_number=pca_number,
                                                     pca_sklearn=self.m_pca,
                                                     im_shape=im_shape,
                                                     indices=indices)

            hist = f'max PC number = {np.amax(self.m_components)}'

            # 1.) derotated residuals
            if self.m_res_arr_out_ports is not None:
                self.m_res_arr_out_ports[pca_number].set_all(res_rot)
                self.m_res_arr_out_ports[pca_number].copy_attributes(self.m_star_in_port)
                self.m_res_arr_out_ports[pca_number].add_history('PcaPsfSubtractionModule', hist)

            # 2.) mean residuals
            if self.m_res_mean_out_port is not None:
                stack = combine_residuals(method='mean', res_rot=res_rot)
                self.m_res_mean_out_port.append(stack, data_dim=3)

            # 3.) median residuals
            if self.m_res_median_out_port is not None:
                stack = combine_residuals(method='median', res_rot=res_rot)
                self.m_res_median_out_port.append(stack, data_dim=3)

            # 4.) noise-weighted residuals
            if self.m_res_weighted_out_port is not None:
                stack = combine_residuals(method='weighted',
                                          res_rot=res_rot,
                                          residuals=residuals,
                                          angles=parang)

                self.m_res_weighted_out_port.append(stack, data_dim=3)

            # 5.) clipped mean residuals
            if self.m_res_rot_mean_clip_out_port is not None:
                stack = combine_residuals(method='clipped', res_rot=res_rot)
                self.m_res_rot_mean_clip_out_port.append(stack, data_dim=3)

        sys.stdout.write('Creating residuals... [DONE]\n')
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

        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')

        if cpu > 1 and self.m_res_arr_out_ports is not None:
            warnings.warn(f'Multiprocessing not possible if \'res_arr_out_tag\' is not set '
                          f'to None.')

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
                raise ValueError('The image size of the science data and the reference data '
                                 'should be identical.')

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
        sys.stdout.write('Constructing PSF model...')
        sys.stdout.flush()

        self.m_pca.fit(ref_reshape)

        # add mean of reference array as 1st PC and orthogonalize it with respect to the PCA basis
        if not self.m_subtract_mean:
            mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

            q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                 self.m_pca.components_[:-1, ])).T)

            self.m_pca.components_ = q_ortho.T

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        if self.m_basis_out_port is not None:
            pc_size = self.m_pca.components_.shape[0]

            basis = np.zeros((pc_size, im_shape[1]*im_shape[2]))
            basis[:, indices] = self.m_pca.components_
            basis = basis.reshape((pc_size, im_shape[1], im_shape[2]))

            self.m_basis_out_port.set_all(basis)

        if cpu == 1 or self.m_res_arr_out_ports is not None:
            self._run_single_processing(star_reshape, im_shape, indices)

        else:
            sys.stdout.write('Creating residuals')
            sys.stdout.flush()

            self._run_multi_processing(star_reshape, im_shape, indices)

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

        history = f'max PC number = {np.amax(self.m_components)}'

        # save history for all other ports
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_mean_out_port.add_history('PcaPsfSubtractionModule', history)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_median_out_port.add_history('PcaPsfSubtractionModule', history)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_weighted_out_port.add_history('PcaPsfSubtractionModule', history)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.copy_attributes(self.m_star_in_port)
            self.m_res_rot_mean_clip_out_port.add_history('PcaPsfSubtractionModule', history)

        self.m_star_in_port.close_port()


class ClassicalADIModule(ProcessingModule):
    """
    Pipeline module for PSF subtraction with classical ADI by subtracting a median-combined
    reference image. A rotation threshold can be set for a fixed separation to prevent
    self-subtraction.
    """

    def __init__(self,
                 threshold,
                 nreference=5,
                 residuals='median',
                 extra_rot=0.,
                 name_in='cadi',
                 image_in_tag='im_arr',
                 res_out_tag='residuals',
                 stack_out_tag='stacked'):
        """
        Parameters
        ----------
        threshold : tuple(float, float, float)
            Tuple with the separation for which the angle threshold is optimized (arcsec), FWHM of
            the PSF (arcsec), and the threshold (FWHM) for the selection of the reference images.
            No threshold is used if set to None.
        nreference : int
            Number of reference image, closest in line to the science image. All images are used if
            *threshold* is None or *nreference* is None.
        residuals : str
            Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        extra_rot : float
            Additional rotation angle of the images (deg).
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        res_out_tag : str
            Tag of the database entry with the residuals of the PSF subtraction that are written
            as output.
        stack_out_tag : str
            Tag of the database entry with the stacked residuals that are written as output.

        Returns
        -------
        NoneType
            None
        """

        super(ClassicalADIModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_res_inout_port = self.add_input_port(res_out_tag)

        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_stack_out_port = self.add_output_port(stack_out_tag)

        self.m_threshold = threshold
        self.m_nreference = nreference
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals

        self.m_count = 0

    def run(self):
        """
        Run method of the module. Selects for each image the reference images closest in line while
        taking into account a rotation threshold for a fixed separation, median-combines the
        references images, and subtracts the reference image from each image separately.
        Alternatively, a single, median-combined reference image can be created and subtracted from
        all images. All images are used if the rotation condition can not be met. Both the
        individual residuals (before derotation) and the stacked residuals are stored.

        Returns
        -------
        NoneType
            None
        """

        def _subtract_psf(image,
                          parang_thres,
                          nref,
                          reference):

            if parang_thres:
                ang_diff = np.abs(parang[self.m_count]-parang)
                index_thres = np.where(ang_diff > parang_thres)[0]

                if index_thres.size == 0:
                    reference = self.m_image_in_port.get_all()
                    warnings.warn('No images meet the rotation threshold. Creating a reference '
                                  'PSF from the median of all images instead.')

                else:
                    if nref:
                        index_diff = np.abs(self.m_count - index_thres)
                        index_near = np.argsort(index_diff)[:nref]
                        index_sort = np.sort(index_thres[index_near])
                        reference = self.m_image_in_port[index_sort, :, :]

                    else:
                        reference = self.m_image_in_port[index_thres, :, :]

                reference = np.median(reference, axis=0)

            self.m_count += 1

            return image-reference

        parang = -1.*self.m_image_in_port.get_attribute('PARANG') + self.m_extra_rot

        if self.m_threshold:
            parang_thres = 2.*math.atan2(self.m_threshold[2]*self.m_threshold[1],
                                         2.*self.m_threshold[0])
            parang_thres = math.degrees(parang_thres)
            reference = None

        else:
            parang_thres = None
            reference = self.m_image_in_port.get_all()
            reference = np.median(reference, axis=0)

        self.apply_function_to_images(_subtract_psf,
                                      self.m_image_in_port,
                                      self.m_res_out_port,
                                      'Running ClassicalADIModule',
                                      func_args=(parang_thres, self.m_nreference, reference))

        im_res = self.m_res_inout_port.get_all()

        res_rot = np.zeros(im_res.shape)
        for i, item in enumerate(parang):
            res_rot[i, ] = rotate(im_res[i, ], item, reshape=False)

        stack = combine_residuals(self.m_residuals,
                                  res_rot,
                                  residuals=im_res,
                                  angles=parang)

        self.m_stack_out_port.set_all(stack)

        if self.m_threshold:
            history = f'threshold [deg] = {parang_thres:.2f}'
        else:
            history = 'threshold [deg] = None'

        self.m_res_out_port.copy_attributes(self.m_image_in_port)
        self.m_res_out_port.add_history('ClassicalADIModule', history)

        self.m_stack_out_port.copy_attributes(self.m_image_in_port)
        self.m_stack_out_port.add_history('ClassicalADIModule', history)

        self.m_res_out_port.close_port()
