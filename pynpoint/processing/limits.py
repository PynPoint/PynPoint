"""
Pipeline modules for estimating detection limits.
"""

import os
import sys
import math
import time
import warnings
import multiprocessing as mp

from typing import Tuple, List

import numpy as np

from scipy.interpolate import griddata
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.limits import contrast_limit
from pynpoint.util.module import progress
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


class ContrastCurveModule(ProcessingModule):
    """
    Pipeline module to calculate contrast limits for a given sigma level or false positive
    fraction, with a correction for small sample statistics. Positions are processed in
    parallel if ``CPU`` is set to a value larger than 1 in the configuration file.
    """

    __author__ = 'Tomas Stolker, Jasper Jonker, Benedikt Schmidhuber'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 contrast_out_tag: str,
                 separation: Tuple[float, float, float] = (0.1, 1., 0.01),
                 angle: Tuple[float, float, float] = (0., 360., 60.),
                 threshold: Tuple[str, float] = ('sigma', 5.),
                 psf_scaling: float = 1.,
                 aperture: float = 0.05,
                 pca_number: int = 20,
                 cent_size: float = None,
                 edge_size: float = None,
                 extra_rot: float = 0.,
                 residuals: str = 'median',
                 snr_inject: float = 100.,
                 **kwargs: float) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that contains the stack with images.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake planet.
            Can be either a single image or a stack of images equal in size to *image_in_tag*.
        contrast_out_tag : str
            Tag of the database entry that contains the separation, azimuthally averaged contrast
            limits, the azimuthal variance of the contrast limits, and the threshold of the false
            positive fraction associated with sigma.
        separation : tuple(float, float, float)
            Range of separations (arcsec) where the contrast is calculated. Should be specified as
            (lower limit, upper limit, step size). Apertures that fall within the mask radius or
            beyond the image size are removed.
        angle : tuple(float, float, float)
            Range of position angles (deg) where the contrast is calculated. Should be specified as
            (lower limit, upper limit, step size), measured counterclockwise with respect to the
            vertical image axis, i.e. East of North.
        threshold : tuple(str, float)
            Detection threshold for the contrast curve, either in terms of 'sigma' or the false
            positive fraction (FPF). The value is a tuple, for example provided as ('sigma', 5.)
            or ('fpf', 1e-6). Note that when sigma is fixed, the false positive fraction will
            change with separation. Also, sigma only corresponds to the standard deviation of a
            normal distribution at large separations (i.e., large number of samples).
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should have a positive value.
        aperture : float
            Aperture radius (arcsec).
        pca_number : int
            Number of principal components used for the PSF subtraction.
        cent_size : float, None
            Central mask radius (arcsec). No mask is used when set to None.
        edge_size : float, None
            Outer edge radius (arcsec) beyond which pixels are masked. No outer mask is used when
            set to None. If the value is larger than half the image size then it will be set to
            half the image size.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        snr_inject : float
            Signal-to-noise ratio of the injected planet signal that is used to measure the amount
            of self-subtraction.

        Returns
        -------
        NoneType
            None
        """

        super(ContrastCurveModule, self).__init__(name_in)

        if 'sigma' in kwargs:
            warnings.warn('The \'sigma\' parameter has been deprecated. Please use the '
                          '\'threshold\' parameter instead.', DeprecationWarning)

        if 'norm' in kwargs:
            warnings.warn('The \'norm\' parameter has been deprecated. It is not recommended to '
                          'normalize the images before PSF subtraction.', DeprecationWarning)

        if 'accuracy' in kwargs:
            warnings.warn('The \'accuracy\' parameter has been deprecated. The parameter is no '
                          'longer required.', DeprecationWarning)

        if 'magnitude' in kwargs:
            warnings.warn('The \'magnitude\' parameter has been deprecated. The parameter is no '
                          'longer required.', DeprecationWarning)

        if 'ignore' in kwargs:
            warnings.warn('The \'ignore\' parameter has been deprecated. The parameter is no '
                          'longer required.', DeprecationWarning)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_contrast_out_port = self.add_output_port(contrast_out_tag)

        self.m_separation = separation
        self.m_angle = angle
        self.m_psf_scaling = psf_scaling
        self.m_threshold = threshold
        self.m_aperture = aperture
        self.m_pca_number = pca_number
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals
        self.m_snr_inject = snr_inject

        if self.m_angle[0] < 0. or self.m_angle[0] > 360. or self.m_angle[1] < 0. or \
           self.m_angle[1] > 360. or self.m_angle[2] < 0. or self.m_angle[2] > 360.:

            raise ValueError('The angular positions of the fake planets should lie between '
                             '0 deg and 360 deg.')

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. An artificial planet is injected (based on the noise level) at a
        given separation and position angle. The amount of self-subtraction is then determined and
        the contrast limit is calculated for a given sigma level or false positive fraction. A
        correction for small sample statistics is applied for both cases. Note that if the sigma
        level is fixed, the false positive fraction changes with separation, following the
        Student's t-distribution (see Mawet et al. 2014 for details).

        Returns
        -------
        NoneType
            None
        """

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError(f'The number of frames in psf_in_tag {psf.shape} does not match with '
                             f'the number of frames in image_in_tag {images.shape}. The '
                             f'DerotateAndStackModule can be used to average the PSF frames '
                             f'(without derotating) before applying the ContrastCurveModule.')

        cpu = self._m_config_port.get_attribute('CPU')
        working_place = self._m_config_port.get_attribute('WORKING_PLACE')

        parang = self.m_image_in_port.get_attribute('PARANG')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        self.m_image_in_port.close_port()
        self.m_psf_in_port.close_port()

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        self.m_aperture /= pixscale

        pos_r = np.arange(self.m_separation[0]/pixscale,
                          self.m_separation[1]/pixscale,
                          self.m_separation[2]/pixscale)

        pos_t = np.arange(self.m_angle[0]+self.m_extra_rot,
                          self.m_angle[1]+self.m_extra_rot,
                          self.m_angle[2])

        if self.m_cent_size is None:
            index_del = np.argwhere(pos_r-self.m_aperture <= 0.)
        else:
            index_del = np.argwhere(pos_r-self.m_aperture <= self.m_cent_size)

        pos_r = np.delete(pos_r, index_del)

        if self.m_edge_size is None or self.m_edge_size > images.shape[1]/2.:
            index_del = np.argwhere(pos_r+self.m_aperture >= images.shape[1]/2.)
        else:
            index_del = np.argwhere(pos_r+self.m_aperture >= self.m_edge_size)

        pos_r = np.delete(pos_r, index_del)

        positions = []
        for sep in pos_r:
            for ang in pos_t:
                positions.append((sep, ang))

        result = []
        async_results = []

        # Create temporary files
        tmp_im_str = os.path.join(working_place, 'tmp_images.npy')
        tmp_psf_str = os.path.join(working_place, 'tmp_psf.npy')

        np.save(tmp_im_str, images)
        np.save(tmp_psf_str, psf)

        mask = create_mask(images.shape[-2:], (self.m_cent_size, self.m_edge_size))

        _, im_res = pca_psf_subtraction(images=images*mask,
                                        angles=-1.*parang+self.m_extra_rot,
                                        pca_number=self.m_pca_number)

        noise = combine_residuals(method=self.m_residuals, res_rot=im_res)

        pool = mp.Pool(cpu)

        for pos in positions:
            async_results.append(pool.apply_async(contrast_limit,
                                                  args=(tmp_im_str,
                                                        tmp_psf_str,
                                                        noise,
                                                        mask,
                                                        parang,
                                                        self.m_psf_scaling,
                                                        self.m_extra_rot,
                                                        self.m_pca_number,
                                                        self.m_threshold,
                                                        self.m_aperture,
                                                        self.m_residuals,
                                                        self.m_snr_inject,
                                                        pos)))

        pool.close()

        start_time = time.time()

        # wait for all processes to finish
        while mp.active_children():
            # number of finished processes
            nfinished = sum([i.ready() for i in async_results])

            progress(nfinished, len(positions), 'Calculating detection limits...', start_time)

            # check if new processes have finished every 5 seconds
            time.sleep(5)

        if nfinished != len(positions):
            sys.stdout.write('\r                                                      ')
            sys.stdout.write('\rCalculating detection limits... [DONE]\n')
            sys.stdout.flush()

        # get the results for every async_result object
        for item in async_results:
            result.append(item.get())

        pool.terminate()

        os.remove(tmp_im_str)
        os.remove(tmp_psf_str)

        result = np.asarray(result)

        # Sort the results first by separation and then by angle
        indices = np.lexsort((result[:, 1], result[:, 0]))
        result = result[indices]

        result = result.reshape((pos_r.size, pos_t.size, 4))

        mag_mean = np.nanmean(result, axis=1)[:, 2]
        mag_var = np.nanvar(result, axis=1)[:, 2]
        res_fpf = result[:, 0, 3]

        limits = np.column_stack((pos_r*pixscale, mag_mean, mag_var, res_fpf))

        self.m_image_in_port._check_status_and_activate()
        self.m_contrast_out_port._check_status_and_activate()

        self.m_contrast_out_port.set_all(limits, data_dim=2)

        history = f'{self.m_threshold[0]} = {self.m_threshold[1]}'
        self.m_contrast_out_port.add_history('ContrastCurveModule', history)
        self.m_contrast_out_port.copy_attributes(self.m_image_in_port)
        self.m_contrast_out_port.close_port()


class MassLimitsModule(ProcessingModule):
    """
    Pipeline module to calculate mass limits from the contrast limits and any isochrones model grid
    downloaded from https://phoenix.ens-lyon.fr/Grids/.
    """

    __author__ = 'Benedikt Schmidhuber, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 contrast_in_tag: str,
                 mass_out_tag: str,
                 model_file: str,
                 star_prop: dict,
                 instr_filter: str = 'L\'') -> None:

        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        contrast_in_tag : str
            Tag of the database entry that contains the contrast curve data, as computed with the
            :class:`~pynpoint.processing.limits.ContrastCurveModule`.
        mass_out_tag : str
            Tag of the database entry with the output data containing the separation, the mass
            limits, and the upper and lower one sigma deviation as calculated for the azimuthal
            variance on the contrast limits.
        model_file: str
            Path to the file containing the model data. Must be in the same format as the
            grids found on https://phoenix.ens-lyon.fr/Grids/. Any of the isochrones files from
            this website can be used.
        star_prop : dict
            Dictionary containing host star properties. Must have the following keys:

                - ``magnitude`` - Apparent magnitude, in the same band as the `instr_filter`.
                - ``distance`` - Distance in parsec.
                - ``age`` - Age of the system in the Myr.

        instr_filter: str
            Instrument filter in the same format as listed in the `model_file`.

        Returns
        -------
        NoneType
            None
        """

        super(MassLimitsModule, self).__init__(name_in)

        self.m_star_age = star_prop['age']/1000.  # [Myr]
        self.m_star_abs = star_prop['magnitude'] - 5.*math.log10(star_prop['distance']/10.)

        self.m_instr_filter = instr_filter
        self.m_model_file = model_file

        if not os.path.exists(self.m_model_file):
            raise ValueError('The path does not appear to be an existing file. Please check the'
                             'path. If you are unsure about the path, pass the absolute path to the'
                             'model file.')

        self.m_contrast_in_port = self.add_input_port(contrast_in_tag)
        self.m_mass_out_port = self.add_output_port(mass_out_tag)

    @staticmethod
    @typechecked
    def read_model(model_file_path: str) -> Tuple[List[float], List[np.ndarray], List[str]]:
        """
        Reads the data from the model file and structures it. Returns an array of available model
        ages and a list of model data for each age.

        Parameters
        -------
        model_file: str
            Path to the file containing the model data.

        Returns
        -------
        list(float, )
            List with all the ages from the model grid.
        list(numpy.ndarray, )
            List with all the isochrone data, so the length is the same as the number of ages.
        list(str, )
            List with all the column names from the model grid.
        """

        # read in all the data, selecting out empty lines or '---' lines
        data = []
        with open(model_file_path) as file:
            for line in file:
                if ('---' in line) or line == '\n':
                    continue
                else:
                    data += [list(filter(None, line.rstrip().split(' ')))]

        # initialize list of ages
        ages = []
        # initialize the header
        header = []
        # initialize a new data list, where the data is separated by age
        isochrones = []

        k = -1
        for _line in data:
            if '(Gyr)' in _line:
                # get time line
                ages += [float(_line[-1])]
                isochrones += [[]]
                k += 1

            elif 'lg(g)' in _line:
                # get header line
                header = ['M/Ms', 'Teff(K)'] + _line[1:]

            else:
                # save the data
                isochrones[k] += [_line]

        for index, _ in enumerate(isochrones):
            isochrones[index] = np.array(isochrones[index], dtype=float)

        return ages, isochrones, header

    @staticmethod
    @typechecked
    def interpolate_model(age_eval: np.ndarray,
                          mag_eval: np.ndarray,
                          filter_index: int,
                          model_age: List[float],
                          model_data: List[np.ndarray]) -> np.ndarray:
        """
        Interpolates the grid based model data.

        Parameters
        ----------
        age_eval : numpy.ndarray
            Age at which the system is evaluated. Must be of the same shape as `mag_eval`.
        mag_eval : numpy.ndarray
            Absolute magnitude for which the system is evaluated. Must be of the same shape as
            `age_eval`.
        filter_index: int
            Column index where the filter is located.
        model_age: list(float, )
            List of ages which are given by the model.
        model_data: list(numpy.ndarray, )
            List of arrays containing the model data.

        Returns
        -------
        griddata : numpy.ndarray
            Interpolated values for the given evaluation points (age_eval, mag_eval). Has the
            same shape as age_eval and mag_eval.
        """

        grid_points = np.array([])
        grid_values = np.array([])

        # create array of available points
        for age_index, age_item in enumerate(model_age):
            iso_mag = model_data[age_index][:, filter_index]
            iso_age = np.ones_like(iso_mag) * age_item
            iso_mass = model_data[age_index][:, 0]

            grid_points = np.append(grid_points, np.column_stack((iso_age, iso_mag)))
            grid_values = np.append(grid_values, iso_mass)

        grid_points = grid_points.reshape(-1, 2)
        interp = np.column_stack((age_eval, mag_eval))

        return griddata(grid_points, grid_values, interp, method='cubic', rescale=True)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Calculates the mass limits from the contrast limits (as
        calculated with the :class:`~pynpoint.processing.limits.ContrastCurveModule`) and the
        isochrones of an evolutionary model. The age and the absolute magnitude of the isochrones
        are linearly interpolated such that the mass limits can be calculated for a given contrast
        limits  (which is converted in an absolute magnitude with the apparent magnitude and
        distance of the central star).

        Returns
        -------
        NoneType
            None
        """

        model_age, model_data, model_header = self.read_model(self.m_model_file)

        assert self.m_instr_filter in model_header, 'The selected filter was not found in the ' \
                                                    'list of available filters from the model.'

        # find the column index of the filter
        # simple argwhere gives empty list?!
        filter_index = np.argwhere([self.m_instr_filter == j for j in model_header])[0]
        filter_index = int(filter_index)

        contrast_data = self.m_contrast_in_port.get_all()

        separation = contrast_data[:, 0]
        contrast = contrast_data[:, 1]
        contrast_std = np.sqrt(contrast_data[:, 2])

        age_eval = self.m_star_age*np.ones_like(contrast)
        mag_eval = self.m_star_abs+contrast

        print('Interpolating isochrones...', end='')

        mass = self.interpolate_model(age_eval=age_eval,
                                      mag_eval=mag_eval,
                                      filter_index=filter_index,
                                      model_age=model_age,
                                      model_data=model_data)

        mass_upper = self.interpolate_model(age_eval=age_eval,
                                            mag_eval=mag_eval-contrast_std,
                                            filter_index=filter_index,
                                            model_age=model_age,
                                            model_data=model_data) - mass

        mass_lower = self.interpolate_model(age_eval=age_eval,
                                            mag_eval=mag_eval+contrast_std,
                                            filter_index=filter_index,
                                            model_age=model_age,
                                            model_data=model_data) - mass

        mass_limits = np.column_stack((separation, mass, mass_upper, mass_lower))
        self.m_mass_out_port.set_all(mass_limits, data_dim=2)

        print(' [DONE]')

        history = f'filter = {self.m_instr_filter}'
        self.m_mass_out_port.add_history('MassLimitsModule', history)
        self.m_mass_out_port.copy_attributes(self.m_contrast_in_port)
        self.m_mass_out_port.close_port()
