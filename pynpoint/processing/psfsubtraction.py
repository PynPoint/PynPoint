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
from pynpoint.util.module import progress
from pynpoint.util.multipca import PcaMultiprocessingCapsule
from pynpoint.util.residuals import combine_residuals
from pynpoint.util.ifs import sdi_scaling, scaling_calculation, \
                              i_want_to_seperate_wavelengths
from pynpoint.util.sdi import postprocessor

class PcaPsfSubtractionModule(ProcessingModule):
    """
    Pipeline module for PSF subtraction with principal component analysis (PCA). The residuals are
    calculated in parallel for the selected numbers of principal components. This may require
    a large amount of memory in case the stack of input images is very large. The number of
    processes can be set with the CPU keyword in the configuration file.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

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
            Unique name of the module instance.
        images_in_tag : str
            Tag of the database entry with the science images that are read as input
        reference_in_tag : str
            Tag of the database entry with the reference images that are read as input.
        res_mean_tag : str, None
            Tag of the database entry with the mean collapsed residuals. Not calculated if set to
            None.
        res_median_tag : str, None
            Tag of the database entry with the median collapsed residuals. Not calculated if set
            to None.
        res_weighted_tag : str, None
            Tag of the database entry with the noise-weighted residuals (see Bottom et al. 2017).
            Not calculated if set to None.
        res_rot_mean_clip_tag : str, None
            Tag of the database entry of the clipped mean residuals. Not calculated if set to
            None.
        res_arr_out_tag : str, None
            Tag of the database entry with the derotated image residuals from the PSF subtraction.
            The tag name of `res_arr_out_tag` is appended with the number of principal components
            that was used. Not calculated if set to None. Not supported with multiprocessing.
        basis_out_tag : str, None
            Tag of the database entry with the basis set. Not stored if set to None.
        pca_numbers : range, list(int, ), numpy.ndarray
            Number of principal components used for the PSF model. Can be a single value or a tuple
            with integers.
        extra_rot : float
            Additional rotation angle of the images (deg).
        subtract_mean : bool
            The mean of the science and reference images is subtracted from the corresponding
            stack, before the PCA basis is constructed and fitted..
        processing_type : str
            Type of post processing:
                ADI: Applying ADI with a PCA reduction using pca_number of principal
                     components. Creates one final image.
                SDI: Applying SDI with a PCA reduction using pca_number of principal
                     components. Creates one image per wavelength.
                ADI+SDI: First applies ADI with a PCA reduction using pca_number of principal
                     components, then applies SDI with a PCA reduction using pca_number of
                     prinzipal components. Creates one image per wavelength.
                SDI+ADI: First applies SDI with a PCA reduction using pca_number of prinzipal
                     components, then applies ADI with a PCA reduction using pca_number of
                     prinzipal components. Creates one image per wavelength.

        Returns
        -------
        NoneType
            None
        """

        super(PcaPsfSubtractionModule, self).__init__(name_in)
        
        if type(pca_numbers) is tuple:
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
            if type(self.m_components) is tuple:
                self.m_res_arr_out_ports = self.add_output_port(res_arr_out_tag)
            else:
                self.m_res_arr_out_ports = {}
                for pca_number in self.m_components:
                    self.m_res_arr_out_ports[pca_number] = self.add_output_port(res_arr_out_tag +
                                                                                str(pca_number))
            
        if basis_out_tag is None:
            self.m_basis_out_port = None
        else:
            self.m_basis_out_port = self.add_output_port(basis_out_tag)

    @typechecked
    def _run_multi_processing(self,
                              star_reshape: np.ndarray,
                              im_shape: tuple,
                              indices: Union[np.ndarray, None]) -> None:
        """
        Internal function to create the residuals, derotate the images, and write the output
        using multiprocessing.
        """

        cpu = self._m_config_port.get_attribute('CPU')
        angles = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

        pixscale = self.m_star_in_port.get_attribute('PIXSCALE')
        lam = self.m_star_in_port.get_attribute('WAVELENGTH')

        if lam is None:
            lam = [1]

        scales = scaling_calculation(pixscale, lam)
        
        
        # Set up the pca numbers for correct handling. The first number will be used for the first
        # PCA step, the second for the subsequent one. If only one step is required, the second pca
        # number list is kept empty.
        if self.m_processing_type in ['Wsap', 'Tsap', 'Wasp', 'Tasp']:
            if type(self.m_components) is not tuple:
                raise ValueError('The selected processing type requires a tuple for pca_number.')
            pca_first = self.m_components[0]
            pca_secon = self.m_components[1]

        else:
            if type(self.m_components) is tuple:
                raise Warning('The selected processing type does not require a tuple for pca_number.' +
                              'To prevent ambiguity, only the first entery of the tuple is used.')
                pca_first = self.m_components[0]
            else:
                pca_first = self.m_components

            # default value for second pca_number: unsused for all further purposes
            pca_secon = [-1]

        if self.m_processing_type is 'Oadi':
            tmp_output = np.zeros((len(self.m_components), im_shape[1], im_shape[2]))
        else:
            if i_want_to_seperate_wavelengths(self.m_processing_type):
                tmp_output = np.zeros((len(pca_first), len(pca_secon), len(lam), im_shape[-2], im_shape[-1]))
            else:
                tmp_output = np.zeros((len(pca_first), len(pca_secon), im_shape[-2], im_shape[-1]))
            
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

        if self.m_res_arr_out_ports is not None:
            for pca_number in self.m_components:
                self.m_res_arr_out_ports[pca_number].close_port()

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
                                            deepcopy(angles),
                                            deepcopy(scales),
                                            im_shape,
                                            indices,
                                            self.m_processing_type)

        capsule.run()

    @typechecked
    def _run_single_processing(self,
                               star_reshape: np.ndarray,
                               im_shape: tuple,
                               indices: Union[np.ndarray, None]) -> None:
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.
        """

        start_time = time.time()

        # calculate parangs
        parang = -1.*self.m_star_in_port.get_attribute('PARANG') + self.m_extra_rot

        # calculate scaling factors
        pixscale = self.m_star_in_port.get_attribute('PIXSCALE')
        lam = self.m_star_in_port.get_attribute('WAVELENGTH')
        if lam is None:
            lam = [1]
        scales = scaling_calculation(pixscale, lam)

        # Set up the pca numbers for correct handling. The first number will be used for the first
        # PCA step, the second for the subsequent one. If only one step is required, the second pca
        # number list is kept empty.
        if self.m_processing_type in ['Wsap', 'Tsap', 'Wasp', 'Tasp']:
            if type(self.m_components) is not tuple:
                raise ValueError('The selected processing type requires a tuple for pca_number.')
            pca_first = self.m_components[0]
            pca_secon = self.m_components[1]

        else:
            if type(self.m_components) is tuple:
                raise Warning('The selected processing type does not require a tuple for pca_number.' +
                              'To prevent ambiguity, only the first entery of the tuple is used.')
                pca_first = self.m_components[0]
            else:
                pca_first = self.m_components

            # default value for second pca_number: unsused for all further purposes
            pca_secon = [-1]

        # set up output arrays
        if i_want_to_seperate_wavelengths(self.m_processing_type):
            out_array_resi = np.zeros(im_shape)
            out_array_mean = np.zeros((len(pca_first), len(pca_secon), len(lam), im_shape[-2], im_shape[-1]))
            out_array_medi = np.zeros((len(pca_first), len(pca_secon), len(lam), im_shape[-2], im_shape[-1]))
            out_array_weig = np.zeros((len(pca_first), len(pca_secon), len(lam), im_shape[-2], im_shape[-1]))
            out_array_clip = np.zeros((len(pca_first), len(pca_secon), len(lam), im_shape[-2], im_shape[-1]))
        else:
            out_array_resi = np.zeros(im_shape)
            out_array_mean = np.zeros((len(pca_first), len(pca_secon), im_shape[-2], im_shape[-1]))
            out_array_medi = np.zeros((len(pca_first), len(pca_secon), im_shape[-2], im_shape[-1]))
            out_array_weig = np.zeros((len(pca_first), len(pca_secon), im_shape[-2], im_shape[-1]))
            out_array_clip = np.zeros((len(pca_first), len(pca_secon), im_shape[-2], im_shape[-1]))
            

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
                    if len(pca_first)+len(pca_secon) == 2:
                        out_array_resi = residuals
                    else:
                        print('Residuals can only be printed if no more than 1 pca number for each ' +
                              'reduction step is selected. With your pca numbers, no residuals are saved.')

                # 2.) mean residuals
                if self.m_res_mean_out_port is not None:
                    out_array_mean[i, j] = combine_residuals(method='mean', 
                                                             res_rot=res_rot,
                                                             processing_type=self.m_processing_type)

                # 3.) median residuals
                if self.m_res_median_out_port is not None:
                    out_array_medi[i, j] = combine_residuals(method='median', 
                                                             res_rot=res_rot,
                                                             processing_type=self.m_processing_type)

                # 4.) noise-weighted residuals
                if self.m_res_weighted_out_port is not None:
                    out_array_weig[i, j] = combine_residuals(method='weighted',
                                                             res_rot=res_rot,
                                                             residuals=residuals,
                                                             angles=parang,
                                                             processing_type=self.m_processing_type)

                # 5.) clipped mean residuals
                if self.m_res_rot_mean_clip_out_port is not None:
                    out_array_clip[i, j] = combine_residuals(method='clipped', 
                                                             res_rot=res_rot,
                                                             processing_type=self.m_processing_type)
    
        # Configurate data output. The arrays are squeezed becuase all dimensions which should not
        # be output have length 1 and therefore can be removed by squeezing
        # 1.) derotated residuals
        if self.m_res_arr_out_ports is not None and len(pca_first)+len(pca_secon) == 2:
            if pca_secon[0] == -1:
                hist = f'max PC number = {pca_first}'
            else:
                hist = f'max PC number = {pca_first} / {pca_secon}'
            squeezed = np.squeeze(out_array_resi)
            
            if type(self.m_components) is tuple:
                self.m_res_arr_out_ports.set_all(squeezed, data_dim=squeezed.ndim)
                self.m_res_arr_out_ports.copy_attributes(self.m_star_in_port)
                self.m_res_arr_out_ports.add_history('PcaPsfSubtractionModule', hist)
            else:
                for p, pca in enumerate(self.m_components):
                    self.m_res_arr_out_ports[pca].append(squeezed[p])
                    self.m_res_arr_out_ports[pca].add_history('PcaPsfSubtractionModule', hist)
            

        # 2.) mean residuals
        if self.m_res_mean_out_port is not None:
            squeezed = np.squeeze(out_array_mean)
            self.m_res_mean_out_port.set_all(squeezed, data_dim=squeezed.ndim)

        # 3.) median residuals
        if self.m_res_median_out_port is not None:
            squeezed = np.squeeze(out_array_medi)
            self.m_res_median_out_port.set_all(squeezed, data_dim=squeezed.ndim)

        # 4.) noise-weighted residuals
        if self.m_res_weighted_out_port is not None:
            squeezed = np.squeeze(out_array_weig)
            self.m_res_weighted_out_port.set_all(squeezed, data_dim=squeezed.ndim)

        # 5.) clipped mean residuals
        if self.m_res_rot_mean_clip_out_port is not None:
            squeezed = np.squeeze(out_array_clip)
            self.m_res_rot_mean_clip_out_port.set_all(squeezed, data_dim=squeezed.ndim)

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

        cpu = self._m_config_port.get_attribute('CPU')

        if cpu > 1 and self.m_res_arr_out_ports is not None:
            warnings.warn(f'Multiprocessing not possible if \'res_arr_out_tag\' is not set '
                          f'to None.')

        # Parse processing_type input to postporcesser support
        valid_pt = ['ADI', 'SDI', 'ADI+SDI', 'SDI+ADI', 'Oadi',
                    'Tnan', 'Wnan', 'Tadi', 'Wadi', 'Tsdi', 'Wsdi',
                    'Tsaa', 'Wsaa', 'Tsap', 'Wsap', 'Tasp', 'Wasp']
        
        # Check if a valid processing type was selected
        if self.m_processing_type not in valid_pt:
            er_msg = ("Invalid processing type " + self.m_processing_type + "; needs to be one of the following: "
                      + str(valid_pt))
            raise ValueError(er_msg)
            
        if self.m_processing_type == 'ADI':
            self.m_processing_type = 'Oadi'
        if self.m_processing_type == 'SDI':
            self.m_processing_type = 'Wsdi'
        if self.m_processing_type == 'ADI+SDI':
            self.m_processing_type = 'Wasp'
        if self.m_processing_type == 'SDI+ADI':
            self.m_processing_type = 'Wsap'

        # get all data
        star_data = self.m_star_in_port.get_all()
        im_shape = star_data.shape
        
        # Check if the data of images_in_tags has the required dimensionallity
        if self.m_processing_type == 'Oadi':
            if star_data.ndim != 3:
                raise ValueError('The dimension of the images_in_tags data should be 3')
        else:
            if star_data.ndim != 4:
                raise ValueError('The dimension of the images_in_tags data should be 4')
            if self.m_star_in_port.get_attribute('WAVELENGTH') is None:
                raise ValueError('The Wavelength information of the images_in_tag is required but was not found.')
        
        if self.m_processing_type is 'Oadi':
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
    
            # add mean of reference array as 1st PC and orthogonalize it with respect to the PCA basis
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
        
        # This set up is used for SDI processes. No preparations are possible because SDI/ADI
        # combinations are case specific and need to be conducted within the pca_psf_subtraction
        # function.
        else:
            self.m_pca = None
            indices = None
            star_reshape = star_data
            if self.m_basis_out_port is not None:
                print('Calculating the  PCA basis of SDI processes is ambiguous and ' +
                      'therefore is skipped.')

        # Running a single processing PCA analysis 
        if cpu == 1 or self.m_res_arr_out_ports is not None:
            self._run_single_processing(star_reshape, im_shape, indices)
            
        # Running multiprocessed PCA analysis
        else:
            print('Creating residuals', end='')
            self._run_multi_processing(star_reshape, im_shape, indices)
            print(' [DONE]')
        
        if type(self.m_components) is tuple:
            history = f'max PC number = {np.amax(self.m_components[0])} / {np.amax(self.m_components[1])}'
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
                 threshold: Union[Tuple[float, float, float], None],
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

        super(ClassicalADIModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_stack_out_port = self.add_output_port(stack_out_tag)

        self.m_threshold = threshold
        self.m_nreference = nreference
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals

        self.m_count = 0

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

        @typechecked
        def _subtract_psf(image: np.ndarray,
                          parang_thres: Optional[float],
                          nref: Optional[int],
                          reference: Optional[np.ndarray] = None) -> np.ndarray:

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
                                      'Classical ADI',
                                      func_args=(parang_thres, self.m_nreference, reference))

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
