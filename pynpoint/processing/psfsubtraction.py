"""
Pipeline modules for PSF subtraction.
"""

import time
import math
import warnings

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.apply_func import subtract_psf
from pynpoint.util.module import progress
from pynpoint.util.multipca import PcaMultiprocessingCapsule
from pynpoint.util.residuals import combine_residuals
from pynpoint.util.postproc import postprocessor
from pynpoint.util.sdi import scaling_factors


class PcaPsfSubtractionModule(ProcessingModule):
    """
    Pipeline module for PSF subtraction with principal component analysis (PCA). The module can
    be used for ADI, RDI (see ``subtract_mean`` parameter), SDI, and ASDI. The residuals are
    calculated in parallel for the selected numbers of principal components. This may require
    a large amount of memory in case the stack of input images is very large. The number of
    processes can therefore be set with the ``CPU`` keyword in the configuration file.
    """

    __author__ = 'Markus Bonse, Tomas Stolker, Sven Kiefer'

    @typechecked
    def __init__(self,
                 name_in: str,
                 images_in_tag: str,
                 reference_in_tag: str,
                 res_mean_tag: Optional[str] = None,
                 res_median_tag: Optional[str] = None,
                 res_weighted_tag: Optional[str] = None,
                 res_rot_mean_clip_tag: Optional[str] = None,
                 res_arr_out_tag: Optional[str] = None,
                 basis_out_tag: Optional[str] = None,
                 pca_numbers: Union[range,
                                    List[int],
                                    np.ndarray,
                                    Tuple[range, range],
                                    Tuple[List[int], List[int]],
                                    Tuple[np.ndarray, np.ndarray]] = range(1, 21),
                 extra_rot: float = 0.,
                 subtract_mean: bool = True,
                 processing_type: str = 'ADI') -> None:
        """
        Parameters
        ----------
        name_in : str
            Name tag of the pipeline module.
        images_in_tag : str
            Database entry with the images from which the PSF model will be subtracted.
        reference_in_tag : str
            Database entry with the reference images from which the PSF model is created. Usually
            ``reference_in_tag`` is the same as ``images_in_tag``, but a different dataset can be
            used as reference images in case of RDI.
        res_mean_tag : str, None
            Database entry where the the mean-collapsed residuals will be stored. The residuals are
            not calculated and stored if set to None.
        res_median_tag : str, None
            Database entry where the the median-collapsed residuals will be stored. The residuals
            are not calculated and stored if set to None.
        res_weighted_tag : str, None
            Database entry where the the noise-weighted residuals will be stored (see Bottom et al.
            2017). The residuals are not calculated and stored if set to None.
        res_rot_mean_clip_tag : str, None
            Tag of the database entry of the clipped mean residuals. Not calculated if set to
            None.
        res_arr_out_tag : str, None
            Database entry where the derotated, but not collapsed, residuals are stored. The number
            of principal components is was used is appended to the ``res_arr_out_tag``. The
            residuals are not stored if set to None. This parameter is not supported with
            multiprocessing (i.e. ``CPU`` > 1). For IFS data and if the processing type is either
            ADI+SDI or SDI+ADI the residuals can only be calculated if exactly 1 principal component
            for each ADI and SDI is given with the pca_numbers parameter.
        basis_out_tag : str, None
            Database entry where the principal components are stored. The data is not stored if set
            to None. Only supported for imaging data with ``processing_type='ADI'``.
        pca_numbers : range, list(int), np.ndarray, tuple(range, range), tuple[list(int),
                      list(int)), tuple(np.ndarray, np.ndarray))
            Number of principal components that are used for the PSF model. With ADI or SDI, a
            single list/range/array needs to be provided while for SDI+ADI or ADI+SDI a tuple is
            required with twice a list/range/array.
        extra_rot : float
            Additional rotation angle of the images (deg).
        subtract_mean : bool
            The mean of the science and reference images is subtracted from the corresponding
            stack, before the PCA basis is constructed and fitted. Set the argument to ``False``
            for RDI, that is, in case ``reference_in_tag`` is different from ``images_in_tag``
            and there is no or limited field rotation. The parameter is only supported with
            ``processing_type='ADI'``.
        processing_type : str
            Post-processing type:
                - ADI: Angular differential imaging. Can be used both on imaging and IFS datasets.
                  This argument is also used for RDI, in which case the ``PARANG`` attribute should
                  contain zeros a derotation angles (e.g. with
                  :func:`~pynpoint.core.pypeline.Pypeline.set_attribute` or
                  :class:`~pynpoint.readwrite.attr_writing.ParangWritingModule`). The collapsed
                  residuals are stored as 3D dataset with one image per principal component.
                - SDI: Spectral differential imaging. Can only be applied on IFS datasets. The
                  collapsed residuals are stored as $D dataset with one image per wavelength and
                  principal component.
                - SDI+ADI: Spectral and angular differential imaging. Can only be applied on IFS
                  datasets. The collapsed residuals are stored as 5D datasets with one image per
                  wavelength and each of the principal components.
                - ADI+SDI: Angular and spectral differential imaging. Can only be applied on IFS
                  datasets. The collapsed residuals are stored as 5D datasets with one image per
                  wavelength and each of the principal components.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_pca_numbers = pca_numbers

        if isinstance(pca_numbers, tuple):
            self.m_components = (np.sort(np.atleast_1d(pca_numbers[0])),
                                 np.sort(np.atleast_1d(pca_numbers[1])))

        else:
            self.m_components = np.sort(np.atleast_1d(pca_numbers))
            self.m_pca = PCA(n_components=np.amax(self.m_components), svd_solver='arpack')

        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean
        self.m_processing_type = processing_type

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
            if isinstance(self.m_components, tuple):
                self.m_res_arr_out_ports = self.add_output_port(res_arr_out_tag)
            else:
                self.m_res_arr_out_ports = {}

                for pca_number in self.m_components:
                    self.m_res_arr_out_ports[pca_number] = self.add_output_port(
                        res_arr_out_tag + str(pca_number))

        if basis_out_tag is None:
            self.m_basis_out_port = None
        else:
            self.m_basis_out_port = self.add_output_port(basis_out_tag)

        if self.m_processing_type in ['ADI', 'SDI']:
            if not isinstance(self.m_components, (range, list, np.ndarray)):
                raise ValueError(f'The post-processing type \'{self.m_processing_type}\' requires '
                                 f'a single range/list/array as argument for \'pca_numbers\'.')

        elif self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
            if not isinstance(self.m_components, tuple):
                raise ValueError(f'The post-processing type \'{self.m_processing_type}\' requires '
                                 f'a tuple for with twice a range/list/array as argument for '
                                 f'\'pca_numbers\'.')

            if res_arr_out_tag is not None and len(self.m_components[0]) + \
                    len(self.m_components[1]) != 2:
                raise ValueError(f'If the post-processing type \'{self.m_processing_type}\' '
                                 'is selected, residuals can only be calculated if no more than '
                                 '1 principal component for ADI and SDI is given.')
        else:
            raise ValueError('Please select a valid post-processing type.')

    @typechecked
    def _run_multi_processing(self,
                              star_reshape: np.ndarray,
                              im_shape: tuple,
                              indices: Optional[np.ndarray]) -> None:
        """
        Internal function to create the residuals, derotate the images, and write the output
        using multiprocessing.
        """

        cpu = self._m_config_port.get_attribute('CPU')
        parang = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

        if self.m_ifs_data:
            if 'WAVELENGTH' in self.m_star_in_port.get_all_non_static_attributes():
                wavelength = self.m_star_in_port.get_attribute('WAVELENGTH')

            else:
                raise ValueError('The wavelengths are not found. These should be stored '
                                 'as the \'WAVELENGTH\' attribute.')

            scales = scaling_factors(wavelength)

        else:
            scales = None

        if self.m_processing_type in ['ADI', 'SDI']:
            pca_first = self.m_components
            pca_secon = [-1]  # Not used

        elif self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
            pca_first = self.m_components[0]
            pca_secon = self.m_components[1]

        if self.m_ifs_data:
            if self.m_processing_type in ['ADI', 'SDI']:
                res_shape = (len(pca_first), len(wavelength), im_shape[-2], im_shape[-1])

            elif self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
                res_shape = (len(pca_first), len(pca_secon), len(wavelength),
                             im_shape[-2], im_shape[-1])

        else:
            res_shape = (len(self.m_components), im_shape[1], im_shape[2])

        tmp_output = np.zeros(res_shape)

        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.set_all(tmp_output, keep_attributes=False)

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.set_all(tmp_output, keep_attributes=False)

        self.m_star_in_port.close_port()
        self.m_reference_in_port.close_port()

        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.close_port()

        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.close_port()

        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.close_port()

        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.close_port()

        if self.m_basis_out_port is not None:
            self.m_basis_out_port.close_port()

        capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                            self.m_res_median_out_port,
                                            self.m_res_weighted_out_port,
                                            self.m_res_rot_mean_clip_out_port,
                                            cpu,
                                            deepcopy(self.m_components),
                                            deepcopy(self.m_pca),
                                            deepcopy(star_reshape),
                                            deepcopy(parang),
                                            deepcopy(scales),
                                            im_shape,
                                            indices,
                                            self.m_processing_type)

        capsule.run()

    @typechecked
    def _run_single_processing(self,
                               star_reshape: np.ndarray,
                               im_shape: tuple,
                               indices: Optional[np.ndarray]) -> None:
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.
        """

        start_time = time.time()

        # Get the parallactic angles
        parang = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

        if self.m_ifs_data:
            # Get the wavelengths
            if 'WAVELENGTH' in self.m_star_in_port.get_all_non_static_attributes():
                wavelength = self.m_star_in_port.get_attribute('WAVELENGTH')

            else:
                raise ValueError('The wavelengths are not found. These should be stored '
                                 'as the \'WAVELENGTH\' attribute.')

            # Calculate the wavelength ratios
            scales = scaling_factors(wavelength)

        else:
            scales = None

        if self.m_processing_type in ['ADI', 'SDI']:
            pca_first = self.m_components
            pca_secon = [-1]  # Not used

        elif self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
            pca_first = self.m_components[0]
            pca_secon = self.m_components[1]

        # Setup output arrays

        out_array_res = np.zeros(im_shape)

        if self.m_ifs_data:
            if self.m_processing_type in ['ADI', 'SDI']:
                res_shape = (len(pca_first), len(wavelength), im_shape[-2], im_shape[-1])

            elif self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
                res_shape = (len(pca_first), len(pca_secon), len(wavelength),
                             im_shape[-2], im_shape[-1])

        else:
            res_shape = (len(pca_first), im_shape[-2], im_shape[-1])

        out_array_mean = np.zeros(res_shape)
        out_array_medi = np.zeros(res_shape)
        out_array_weig = np.zeros(res_shape)
        out_array_clip = np.zeros(res_shape)

        # loop over all different combination of pca_numbers and applying the reductions
        for i, pca_1 in enumerate(pca_first):
            for j, pca_2 in enumerate(pca_secon):
                progress(i+j, len(pca_first)+len(pca_secon), 'Creating residuals...', start_time)

                # process images
                residuals, res_rot = postprocessor(images=star_reshape,
                                                   angles=parang,
                                                   scales=scales,
                                                   pca_number=(pca_1, pca_2),
                                                   pca_sklearn=self.m_pca,
                                                   im_shape=im_shape,
                                                   indices=indices,
                                                   processing_type=self.m_processing_type)

                # 1.) derotated residuals
                if self.m_res_arr_out_ports is not None:
                    if not self.m_ifs_data:
                        self.m_res_arr_out_ports[pca_1].set_all(res_rot)
                        self.m_res_arr_out_ports[pca_1].copy_attributes(self.m_star_in_port)
                        self.m_res_arr_out_ports[pca_1].add_history(
                            'PcaPsfSubtractionModule', f'max PC number = {pca_first}')

                    else:
                        out_array_res = residuals

                # 2.) mean residuals
                if self.m_res_mean_out_port is not None:
                    if self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
                        out_array_mean[i, j] = combine_residuals(method='mean',
                                                                 res_rot=res_rot,
                                                                 angles=parang)

                    else:
                        out_array_mean[i] = combine_residuals(method='mean',
                                                              res_rot=res_rot,
                                                              angles=parang)

                # 3.) median residuals
                if self.m_res_median_out_port is not None:
                    if self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
                        out_array_medi[i, j] = combine_residuals(method='median',
                                                                 res_rot=res_rot,
                                                                 angles=parang)

                    else:
                        out_array_medi[i] = combine_residuals(method='median',
                                                              res_rot=res_rot,
                                                              angles=parang)

                # 4.) noise-weighted residuals
                if self.m_res_weighted_out_port is not None:
                    if self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
                        out_array_weig[i, j] = combine_residuals(method='weighted',
                                                                 res_rot=res_rot,
                                                                 residuals=residuals,
                                                                 angles=parang)

                    else:
                        out_array_weig[i] = combine_residuals(method='weighted',
                                                              res_rot=res_rot,
                                                              residuals=residuals,
                                                              angles=parang)

                # 5.) clipped mean residuals
                if self.m_res_rot_mean_clip_out_port is not None:
                    if self.m_processing_type in ['SDI+ADI', 'ADI+SDI']:
                        out_array_clip[i, j] = combine_residuals(method='clipped',
                                                                 res_rot=res_rot,
                                                                 angles=parang)

                    else:
                        out_array_clip[i] = combine_residuals(method='clipped',
                                                              res_rot=res_rot,
                                                              angles=parang)

        # Configurate data output according to the processing type
        # 1.) derotated residuals
        if self.m_res_arr_out_ports is not None and self.m_ifs_data:
            if pca_secon[0] == -1:
                history = f'max PC number = {pca_first}'

            else:
                history = f'max PC number = {pca_first} / {pca_secon}'

            # squeeze out_array_res to reduce dimensionallity as the residuals of
            # SDI+ADI and ADI+SDI are always of the form (1, 1, ...)
            squeezed = np.squeeze(out_array_res)

            if isinstance(self.m_components, tuple):
                self.m_res_arr_out_ports.set_all(squeezed, data_dim=squeezed.ndim)
                self.m_res_arr_out_ports.copy_attributes(self.m_star_in_port)
                self.m_res_arr_out_ports.add_history('PcaPsfSubtractionModule', history)

            else:
                for i, pca in enumerate(self.m_components):
                    self.m_res_arr_out_ports[pca].append(squeezed[i])
                    self.m_res_arr_out_ports[pca].add_history('PcaPsfSubtractionModule', history)

        # 2.) mean residuals
        if self.m_res_mean_out_port is not None:
            self.m_res_mean_out_port.set_all(out_array_mean,
                                             data_dim=out_array_mean.ndim)

        # 3.) median residuals
        if self.m_res_median_out_port is not None:
            self.m_res_median_out_port.set_all(out_array_medi,
                                               data_dim=out_array_medi.ndim)

        # 4.) noise-weighted residuals
        if self.m_res_weighted_out_port is not None:
            self.m_res_weighted_out_port.set_all(out_array_weig,
                                                 data_dim=out_array_weig.ndim)

        # 5.) clipped mean residuals
        if self.m_res_rot_mean_clip_out_port is not None:
            self.m_res_rot_mean_clip_out_port.set_all(out_array_clip,
                                                      data_dim=out_array_clip.ndim)

    @typechecked
    def run(self) -> None:
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

        print('Input parameters:')
        print(f'   - Post-processing type: {self.m_processing_type}')
        print(f'   - Number of principal components: {self.m_pca_numbers}')
        print(f'   - Subtract mean: {self.m_subtract_mean}')
        print(f'   - Extra rotation (deg): {self.m_extra_rot}')

        cpu = self._m_config_port.get_attribute('CPU')

        if cpu > 1 and self.m_res_arr_out_ports is not None:
            warnings.warn('Multiprocessing not possible if \'res_arr_out_tag\' is not set '
                          'to None.')

        # Read the data
        star_data = self.m_star_in_port.get_all()
        im_shape = star_data.shape

        # Parse input processing types to internal processing types
        if star_data.ndim == 3:
            self.m_ifs_data = False

        elif star_data.ndim == 4:
            self.m_ifs_data = True

        else:
            raise ValueError(f'The input data has {star_data.ndim} dimensions while only 3 or 4 '
                             f' are supported by the pipeline module.')

        if self.m_processing_type == 'ADI' and not self.m_ifs_data:
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
            print('Constructing PSF model...', end='')
            self.m_pca.fit(ref_reshape)

            # add mean of reference array as 1st PC and orthogonalize it with respect to
            # the other principal components
            if not self.m_subtract_mean:
                mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

                q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                     self.m_pca.components_[:-1, ])).T)

                self.m_pca.components_ = q_ortho.T

            print(' [DONE]')

            if self.m_basis_out_port is not None:
                pc_size = self.m_pca.components_.shape[0]

                basis = np.zeros((pc_size, im_shape[1]*im_shape[2]))
                basis[:, indices] = self.m_pca.components_
                basis = basis.reshape((pc_size, im_shape[1], im_shape[2]))

                self.m_basis_out_port.set_all(basis)

        else:
            # This setup is used for SDI processes. No preparations are possible because SDI/ADI
            # combinations are case specific and need to be conducted in pca_psf_subtraction.
            self.m_pca = None
            indices = None
            star_reshape = star_data

        # Running a single processing PCA analysis
        if cpu == 1 or self.m_res_arr_out_ports is not None:
            self._run_single_processing(star_reshape, im_shape, indices)

        # Running multiprocessed PCA analysis
        else:
            print('Creating residuals', end='')
            self._run_multi_processing(star_reshape, im_shape, indices)
            print(' [DONE]')

        # write history
        if isinstance(self.m_components, tuple):
            history = f'max PC number = {np.amax(self.m_components[0])} / ' \
                      f'{np.amax(self.m_components[1])}'

        else:
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

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 res_out_tag: str,
                 stack_out_tag: str,
                 threshold: Optional[Tuple[float, float, float]],
                 nreference: Optional[int] = None,
                 residuals: str = 'median',
                 extra_rot: float = 0.) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        res_out_tag : str
            Tag of the database entry with the residuals of the PSF subtraction that are written
            as output.
        stack_out_tag : str
            Tag of the database entry with the stacked residuals that are written as output.
        threshold : tuple(float, float, float), None
            Tuple with the separation for which the angle threshold is optimized (arcsec), FWHM of
            the PSF (arcsec), and the threshold (FWHM) for the selection of the reference images.
            No threshold is used if set to None.
        nreference : int, None
            Number of reference images, closest in line to the science image. All images are used if
            *threshold* is None or *nreference* is None.
        residuals : str
            Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        extra_rot : float
            Additional rotation angle of the images (deg).

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_stack_out_port = self.add_output_port(stack_out_tag)

        self.m_threshold = threshold
        self.m_nreference = nreference
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals

    @typechecked
    def run(self) -> None:
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

        parang = -1.*self.m_image_in_port.get_attribute('PARANG') + self.m_extra_rot

        nimages = self.m_image_in_port.get_shape()[0]

        if self.m_threshold:
            parang_thres = 2.*math.atan2(self.m_threshold[2]*self.m_threshold[1],
                                         2.*self.m_threshold[0])
            parang_thres = math.degrees(parang_thres)
            reference = None

        else:
            parang_thres = None
            reference = self.m_image_in_port.get_all()
            reference = np.median(reference, axis=0)

        ang_diff = np.zeros((nimages, parang.shape[0]))

        for i in range(nimages):
            ang_diff[i, :] = np.abs(parang[i] - parang)

        self.apply_function_to_images(subtract_psf,
                                      self.m_image_in_port,
                                      self.m_res_out_port,
                                      'Classical ADI',
                                      func_args=(parang_thres,
                                                 self.m_nreference,
                                                 reference,
                                                 ang_diff,
                                                 self.m_image_in_port))

        self.m_res_in_port = self.add_input_port(self.m_res_out_port._m_tag)
        im_res = self.m_res_in_port.get_all()

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
