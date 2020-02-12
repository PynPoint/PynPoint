"""
Pipeline modules for photometric and astrometric measurements.
"""

import sys
import time
import warnings

from typing import Union, Tuple, List
from multiprocessing import Pool

import numpy as np
import emcee

from typeguard import typechecked
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from photutils import aperture_photometry, CircularAperture

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.analysis import fake_planet, merit_function, false_alarm, gaussian_noise
from pynpoint.util.image import create_mask, polar_to_cartesian, cartesian_to_polar, \
                                center_subpixel, rotate_coordinates
from pynpoint.util.mcmc import lnprob
from pynpoint.util.module import progress, memory_frames
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


class FakePlanetModule(ProcessingModule):
    """
    Pipeline module to inject a positive or negative artificial planet into a stack of images.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 image_out_tag: str,
                 position: Tuple[float, float],
                 magnitude: float,
                 psf_scaling: float = 1.,
                 interpolation: str = 'spline') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with images that are read as input.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake planet.
            Can be either a single image (2D) or a cube (3D) with the dimensions equal to
            *image_in_tag*.
        image_out_tag : str
            Tag of the database entry with images that are written as output.
        position : tuple(float, float)
            Angular separation (arcsec) and position angle (deg) of the fake planet. Angle is
            measured in counterclockwise direction with respect to the upward direction (i.e.,
            East of North).
        magnitude : float
            Magnitude of the fake planet with respect to the star.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). A negative value will inject a negative planet signal.
        interpolation : str
            Type of interpolation that is used for shifting the images (spline, bilinear, or fft).

        Returns
        -------
        NoneType
            None
        """

        super(FakePlanetModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_interpolation = interpolation

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Shifts the PSF template to the location of the fake planet
        with an additional correction for the parallactic angle and an optional flux scaling.
        The stack of images with the injected planet signal is stored.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute('MEMORY')
        parang = self.m_image_in_port.get_attribute('PARANG')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        self.m_position = (self.m_position[0]/pixscale, self.m_position[1])

        im_shape = self.m_image_in_port.get_shape()
        psf_shape = self.m_psf_in_port.get_shape()

        if psf_shape[0] != 1 and psf_shape[0] != im_shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number '
                             'of frames in image_in_tag. The DerotateAndStackModule can be '
                             'used to average the PSF frames (without derotating) before applying '
                             'the FakePlanetModule.')

        if psf_shape[-2:] != im_shape[-2:]:
            raise ValueError(f'The images in \'{self.m_image_in_port.tag}\' should have the same '
                             f'dimensions as the images images in \'{self.m_psf_in_port.tag}\'.')

        frames = memory_frames(memory, im_shape[0])

        start_time = time.time()
        for j, _ in enumerate(frames[:-1]):
            progress(j, len(frames[:-1]), 'Injecting artificial planets...', start_time)

            images = self.m_image_in_port[frames[j]:frames[j+1]]
            angles = parang[frames[j]:frames[j+1]]

            if psf_shape[0] == 1:
                psf = self.m_psf_in_port.get_all()
            else:
                psf = self.m_psf_in_port[frames[j]:frames[j+1]]

            im_fake = fake_planet(images=images,
                                  psf=psf,
                                  parang=angles,
                                  position=self.m_position,
                                  magnitude=self.m_magnitude,
                                  psf_scaling=self.m_psf_scaling,
                                  interpolation='spline')

            self.m_image_out_port.append(im_fake, data_dim=3)

        history = f'(sep, angle, mag) = ({self.m_position[0]*pixscale:.2f}, ' \
                  f'{self.m_position[1]:.2f}, {self.m_magnitude:.2f})'

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('FakePlanetModule', history)
        self.m_image_out_port.close_port()


class SimplexMinimizationModule(ProcessingModule):
    """
    Pipeline module to measure the flux and position of a planet by injecting negative fake planets
    and minimizing a figure of merit.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 res_out_tag: str,
                 flux_position_tag: str,
                 position: Tuple[int, int],
                 magnitude: float,
                 psf_scaling: float = -1.,
                 merit: str = 'hessian',
                 aperture: float = 0.1,
                 sigma: float = 0.0,
                 tolerance: float = 0.1,
                 pca_number: Union[int, range, List[int]] = 10,
                 cent_size: float = None,
                 edge_size: float = None,
                 extra_rot: float = 0.,
                 residuals: str = 'median',
                 reference_in_tag: str = None,
                 offset: float = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        psf_in_tag : str
            Tag of the database entry with the reference PSF that is used as fake planet. Can be
            either a single image or a stack of images equal in size to ``image_in_tag``.
        res_out_tag : str
            Tag of the database entry with the image residuals that are written as output. The
            residuals are stored for each step of the minimization. The last image contains the
            best-fit residuals.
        flux_position_tag : str
            Tag of the database entry with the flux and position results that are written as output.
            Each step of the minimization stores the x position (pix), y position (pix), separation
            (arcsec), angle (deg), contrast (mag), and the chi-square value. The last row contains
            the best-fit results.
        position : tuple(int, int)
            Approximate position (x, y) of the planet (pix). This is also the location where the
            figure of merit is calculated within an aperture of radius ``aperture``.
        magnitude : float
            Approximate magnitude of the planet relative to the star.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be negative in order to inject negative fake planets.
        merit : str
            Figure of merit for the minimization. Can be 'hessian', to minimize the sum of the
            absolute values of the determinant of the Hessian matrix, or 'poisson', to minimize the
            sum of the absolute pixel values, assuming a Poisson distribution for the noise
            (Wertz et al. 2017), or 'gaussian', to minimize the ratio of the squared pixel values
            and the variance of the pixels within an annulus but excluding the aperture area.
        aperture : float
            Aperture radius (arcsec) at the position specified at *position*.
        sigma : float
            Standard deviation (arcsec) of the Gaussian kernel which is used to smooth the images
            before the figure of merit is calculated (in order to reduce small pixel-to-pixel
            variations).
        tolerance : float
            Absolute error on the input parameters, position (pix) and contrast (mag), that is used
            as acceptance level for convergence. Note that only a single value can be specified
            which is used for both the position and flux so tolerance=0.1 will give a precision of
            0.1 mag and 0.1 pix. The tolerance on the output (i.e., the chi-square value) is set to
            np.inf so the condition is always met.
        pca_number : int, range, list(int, )
            Number of principal components (PCs) used for the PSF subtraction. Can be either a
            single value or a range/list of values. In the latter case, the `res_out_tag` and
            `flux_position_tag` contain a 3 digit number with the number of PCs.
        cent_size : float
            Radius of the central mask (arcsec). No mask is used when set to None. Masking is done
            after the artificial planet is injected.
        edge_size : float
            Outer radius (arcsec) beyond which pixels are masked. No outer mask is used when set to
            None. The radius will be set to half the image size if the argument is larger than half
            the image size. Masking is done after the artificial planet is injected.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        reference_in_tag : str, None
            Tag of the database entry with the reference images that are read as input. The data of
            the ``image_in_tag`` itself is used as reference data for the PSF subtraction if set to
            None. Note that the mean is not subtracted from the data of ``image_in_tag`` and
            ``reference_in_tag`` in case the ``reference_in_tag`` is used, to allow for flux and
            position measurements in the context of RDI.
        offset : float, None
            Offset (pix) by which the injected negative PSF may deviate from ``position``. No
            constraint on the position is applied if set to None.

        Returns
        -------
        NoneType
            None
        """

        super(SimplexMinimizationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        if reference_in_tag is None:
            self.m_reference_in_port = None
        else:
            self.m_reference_in_port = self.add_input_port(reference_in_tag)

        self.m_res_out_port = []
        self.m_flux_pos_port = []

        if isinstance(pca_number, int):
            self.m_res_out_port.append(self.add_output_port(res_out_tag))
            self.m_flux_pos_port.append(self.add_output_port(flux_position_tag))

        else:
            for item in pca_number:
                self.m_res_out_port.append(self.add_output_port(res_out_tag+f'{item:03d}'))
                self.m_flux_pos_port.append(self.add_output_port(flux_position_tag+f'{item:03d}'))

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals
        self.m_offset = offset

        if isinstance(pca_number, int):
            self.m_pca_number = [pca_number]
        else:
            self.m_pca_number = pca_number

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. The position and contrast of a planet is measured by injecting
        negative copies of the PSF template and applying a simplex method (Nelder-Mead) for
        minimization of a figure of merit at the planet location.

        Returns
        -------
        NoneType
            None
        """

        for item in self.m_res_out_port:
            item.del_all_data()
            item.del_all_attributes()

        for item in self.m_flux_pos_port:
            item.del_all_data()
            item.del_all_attributes()

        parang = self.m_image_in_port.get_attribute('PARANG')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        aperture = (self.m_position[1], self.m_position[0], self.m_aperture/pixscale)

        self.m_sigma /= pixscale

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        psf = self.m_psf_in_port.get_all()
        images = self.m_image_in_port.get_all()

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number '
                             'of frames in image_in_tag. The DerotateAndStackModule can be '
                             'used to average the PSF frames (without derotating) before applying '
                             'the SimplexMinimizationModule.')

        center = center_subpixel(psf)

        if self.m_reference_in_port is not None and self.m_merit != 'poisson':
            raise NotImplementedError('The reference_in_tag can only be used in combination with '
                                      'the \'poisson\' figure of merit.')

        def _objective(arg, count, n_components, sklearn_pca, noise):
            pos_y = arg[0]
            pos_x = arg[1]
            mag = arg[2]

            if self.m_offset is not None:
                if pos_x < self.m_position[0] - self.m_offset or \
                        pos_x > self.m_position[0] + self.m_offset:
                    return np.inf

                if pos_y < self.m_position[1] - self.m_offset or \
                        pos_y > self.m_position[1] + self.m_offset:
                    return np.inf

            sep_ang = cartesian_to_polar(center, pos_y, pos_x)

            fake = fake_planet(images=images,
                               psf=psf,
                               parang=parang,
                               position=(sep_ang[0], sep_ang[1]),
                               magnitude=mag,
                               psf_scaling=self.m_psf_scaling)

            mask = create_mask(fake.shape[-2:], (self.m_cent_size, self.m_edge_size))

            if self.m_reference_in_port is None:
                im_res_rot, im_res_derot = pca_psf_subtraction(images=fake*mask,
                                                               angles=-1.*parang+self.m_extra_rot,
                                                               pca_number=n_components,
                                                               pca_sklearn=sklearn_pca,
                                                               im_shape=None,
                                                               indices=None)

            else:
                im_reshape = np.reshape(fake*mask, (im_shape[0], im_shape[1]*im_shape[2]))

                im_res_rot, im_res_derot = pca_psf_subtraction(images=im_reshape,
                                                               angles=-1.*parang+self.m_extra_rot,
                                                               pca_number=n_components,
                                                               pca_sklearn=sklearn_pca,
                                                               im_shape=im_shape,
                                                               indices=None)

            res_stack = combine_residuals(method=self.m_residuals,
                                          res_rot=im_res_derot,
                                          residuals=im_res_rot,
                                          angles=parang)

            self.m_res_out_port[count].append(res_stack, data_dim=3)

            chi_square = merit_function(residuals=res_stack[0, ],
                                        merit=self.m_merit,
                                        aperture=aperture,
                                        sigma=self.m_sigma,
                                        noise=noise)

            position = rotate_coordinates(center, (pos_y, pos_x), -self.m_extra_rot)

            res = np.asarray([position[1],
                              position[0],
                              sep_ang[0]*pixscale,
                              (sep_ang[1]-self.m_extra_rot) % 360.,
                              mag,
                              chi_square])

            self.m_flux_pos_port[count].append(res, data_dim=2)

            sys.stdout.write('\rSimplex minimization... ')
            sys.stdout.write(f'{n_components} PC - chi^2 = {chi_square:.8E}')
            sys.stdout.flush()

            return chi_square

        pos_init = rotate_coordinates(center,
                                      (self.m_position[1], self.m_position[0]),  # (y, x)
                                      self.m_extra_rot)

        for i, n_components in enumerate(self.m_pca_number):
            sys.stdout.write(f'\rSimplex minimization... {n_components} PC ')
            sys.stdout.flush()

            if self.m_reference_in_port is None:
                sklearn_pca = None

            else:
                ref_data = self.m_reference_in_port.get_all()

                im_shape = images.shape
                ref_shape = ref_data.shape

                if ref_shape[1:] != im_shape[1:]:
                    raise ValueError('The image size of the science data and the reference data '
                                     'should be identical.')

                # reshape reference data and select the unmasked pixels
                ref_reshape = ref_data.reshape(ref_shape[0], ref_shape[1]*ref_shape[2])

                mean_ref = np.mean(ref_reshape, axis=0)
                ref_reshape -= mean_ref

                # create the PCA basis
                sklearn_pca = PCA(n_components=n_components, svd_solver='arpack')
                sklearn_pca.fit(ref_reshape)

                # add mean of reference array as 1st PC and orthogonalize it to the PCA basis
                mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

                q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                     sklearn_pca.components_[:-1, ])).T)

                sklearn_pca.components_ = q_ortho.T

            if self.m_merit in ('poisson', 'hessian'):
                noise = None

            elif self.m_merit == 'gaussian':
                noise = gaussian_noise(images=images,
                                       parang=parang,
                                       cent_size=self.m_cent_size,
                                       edge_size=self.m_edge_size,
                                       pca_number=n_components,
                                       residuals=self.m_residuals,
                                       aperture=aperture)

            minimize(fun=_objective,
                     x0=[pos_init[0], pos_init[1], self.m_magnitude],
                     args=(i, n_components, sklearn_pca, noise),
                     method='Nelder-Mead',
                     tol=None,
                     options={'xatol': self.m_tolerance, 'fatol': float('inf')})

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        history = f'merit = {self.m_merit}'

        for item in self.m_flux_pos_port:
            item.copy_attributes(self.m_image_in_port)
            item.add_history('SimplexMinimizationModule', history)

        for item in self.m_res_out_port:
            item.copy_attributes(self.m_image_in_port)
            item.add_history('SimplexMinimizationModule', history)

        self.m_res_out_port[0].close_port()


class FalsePositiveModule(ProcessingModule):
    """
    Pipeline module to calculate the signal-to-noise ratio (SNR) and false positive fraction (FPF)
    at a specified location in an image by using the Student's t-test (Mawet et al. 2014).
    Optionally, the SNR can be optimized with the aperture position as free parameter.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 snr_out_tag: str,
                 position: Tuple[float, float],
                 aperture: float = 0.1,
                 ignore: bool = False,
                 optimize: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input. The SNR/FPF is
            calculated for each image in the dataset.
        snr_out_tag : str
            Tag of the database entry that is written as output. The output format is: (x position
            (pix), y position (pix), separation (arcsec), position angle (deg), SNR, FPF). The
            position angle is measured in counterclockwise direction with respect to the upward
            direction (i.e., East of North).
        position : tuple(float, float)
            The x and y position (pix) where the SNR and FPF is calculated. Note that the bottom
            left of the image is defined as (-0.5, -0.5) so there is a -1.0 offset with respect
            to the DS9 coordinate system. Aperture photometry corrects for the partial inclusion
            of pixels at the boundary.
        aperture : float
            Aperture radius (arcsec).
        ignore : bool
            Ignore the two neighboring apertures that may contain self-subtraction from the planet.
        optimize : bool
            Optimize the SNR. The aperture position is stored in the `snr_out_tag`. The size of the
            aperture is kept fixed.

        Keyword arguments
        -----------------
        tolerance : float
            The absolute tolerance on the position for the optimization to end. Default is set
            to 0.01 (pix).
        offset : float, None
            Offset (pix) by which the aperture may deviate from ``position`` when
            ``optimize=True`` (default: None).

        Returns
        -------
        NoneType
            None
        """

        if 'tolerance' in kwargs:
            self.m_tolerance = kwargs['tolerance']
        else:
            self.m_tolerance = 1e-2

        if 'offset' in kwargs:
            self.m_offset = kwargs['offset']
        else:
            self.m_offset = None

        if 'bounds' in kwargs:
            warnings.warn('The \'bounds\' keyword argument has been deprecated. Please use '
                          '\'offset\' instead (e.g. offset=3.0).', DeprecationWarning)

        super(FalsePositiveModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_snr_out_port = self.add_output_port(snr_out_tag)

        self.m_position = position
        self.m_aperture = aperture
        self.m_ignore = ignore
        self.m_optimize = optimize

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Calculates the SNR and FPF for a specified position in a post-
        processed image with the Student's t-test (Mawet et al. 2014). This approach assumes
        Gaussian noise but accounts for small sample statistics.

        Returns
        -------
        NoneType
            None
        """

        def _snr_optimize(arg):
            pos_x, pos_y = arg

            if self.m_offset is not None:
                if pos_x < self.m_position[0] - self.m_offset or \
                        pos_x > self.m_position[0] + self.m_offset:
                    snr = 0.

                elif pos_y < self.m_position[1] - self.m_offset or \
                        pos_y > self.m_position[1] + self.m_offset:
                    snr = 0.

                else:
                    snr = None

            if self.m_offset is None or snr is None:
                _, _, snr, _ = false_alarm(image=image,
                                           x_pos=pos_x,
                                           y_pos=pos_y,
                                           size=self.m_aperture,
                                           ignore=self.m_ignore)

            return -snr

        self.m_snr_out_port.del_all_data()
        self.m_snr_out_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        self.m_aperture /= pixscale

        nimages = self.m_image_in_port.get_shape()[0]

        start_time = time.time()

        for j in range(nimages):
            progress(j, nimages, 'Calculating S/N and FPF...', start_time)

            image = self.m_image_in_port[j, ]
            center = center_subpixel(image)

            if self.m_optimize:
                result = minimize(fun=_snr_optimize,
                                  x0=[self.m_position[0], self.m_position[1]],
                                  method='Nelder-Mead',
                                  tol=None,
                                  options={'xatol': self.m_tolerance, 'fatol': float('inf')})

                _, _, snr, fpf = false_alarm(image=image,
                                             x_pos=result.x[0],
                                             y_pos=result.x[1],
                                             size=self.m_aperture,
                                             ignore=self.m_ignore)

                x_pos, y_pos = result.x[0], result.x[1]

            else:
                _, _, snr, fpf = false_alarm(image=image,
                                             x_pos=self.m_position[0],
                                             y_pos=self.m_position[1],
                                             size=self.m_aperture,
                                             ignore=self.m_ignore)

                x_pos, y_pos = self.m_position[0], self.m_position[1]

            sep_ang = cartesian_to_polar(center, y_pos, x_pos)
            result = np.column_stack((x_pos, y_pos, sep_ang[0]*pixscale, sep_ang[1], snr, fpf))

            self.m_snr_out_port.append(result, data_dim=2)

        history = f'aperture [arcsec] = {self.m_aperture*pixscale:.2f}'
        self.m_snr_out_port.copy_attributes(self.m_image_in_port)
        self.m_snr_out_port.add_history('FalsePositiveModule', history)
        self.m_snr_out_port.close_port()


class MCMCsamplingModule(ProcessingModule):
    """
    Pipeline module to measure the separation, position angle, and contrast of a planet with
    injection of negative artificial planets and sampling of the posterior distributions with
    emcee, an affine invariant Markov chain Monte Carlo (MCMC) ensemble sampler.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 chain_out_tag: str,
                 param: Tuple[float, float, float],
                 bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 nwalkers: int = 100,
                 nsteps: int = 200,
                 psf_scaling: float = -1.,
                 pca_number: int = 20,
                 aperture: Union[float, Tuple[int, int, float]] = 0.1,
                 mask: Union[Tuple[float, float], Tuple[None, float],
                             Tuple[float, None], Tuple[None, None]] = None,
                 extra_rot: float = 0.,
                 merit: str = 'gaussian',
                 residuals: str = 'median',
                 **kwargs: Union[float, Tuple[float, float, float]]) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with images that are read as input.
        psf_in_tag : str
            Tag of the database entry with the reference PSF that is used as fake planet. Can be
            either a single image (2D) or a cube (3D) with the dimensions equal to *image_in_tag*.
        chain_out_tag : str
            Tag of the database entry with the Markov chain that is written as output. The shape
            of the array is (nsteps, nwalkers, 3). The mean acceptance fraction and the integrated
            autocorrelation time are stored as attributes to this dataset.
        param : tuple(float, float, float)
            The approximate separation (arcsec), angle (deg), and contrast (mag), for example
            obtained with the :class:`~pynpoint.processing.fluxposition.SimplexMinimizationModule`.
            The angle is measured in counterclockwise direction with respect to the upward
            direction (i.e., East of North). The specified separation and angle are also used as
            fixed position for the aperture if *aperture* contains a float value.
        bounds : tuple(tuple(float, float), tuple(float, float), tuple(float, float))
            The boundaries of the separation (arcsec), angle (deg), and contrast (mag). Each set
            of boundaries is specified as a tuple.
        nwalkers : int
            Number of ensemble members.
        nsteps : int
            Number of steps to run per walker.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be negative in order to inject negative fake planets.
        pca_number : int
            Number of principal components used for the PSF subtraction.
        aperture : float, tuple(int, int, float)
            Either the aperture radius (arcsec) at the position of `param` or tuple with the
            position and aperture radius (arcsec) as (pos_x, pos_y, radius).
        mask : tuple(float, float), None
            Inner and outer mask radius (arcsec) for the PSF subtraction. Both elements of the
            tuple can be set to None. Masked pixels are excluded from the PCA computation,
            resulting in a smaller runtime. Masking is done after the artificial planet is
            injected.
        extra_rot : float
            Additional rotation angle of the images (deg).
        merit : str
            Figure of merit that is used for the likelihood function ('gaussian' or 'poisson').
            Pixels are assumed to be independent measurements which are expected to be equal to
            zero in case the best-fit negative PSF template is injected. With 'gaussian', the
            variance is estimated from the pixel values within an annulus at the separation of
            the aperture (but excluding the pixels within the aperture). With 'poisson', a Poisson
            distribution is assumed for the variance of each pixel value (see Wertz et al. 2017).
        residuals : str
            Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').

        Keyword arguments
        -----------------
        sigma : tuple(float, float, float)
            The standard deviations that randomly initializes the start positions of the walkers in
            a small ball around the a priori preferred position. The tuple should contain a value
            for the separation (arcsec), position angle (deg), and contrast (mag). The default is
            set to (1e-5, 1e-3, 1e-3).

        Returns
        -------
        NoneType
            None
        """

        if 'prior' in kwargs:
            warnings.warn('The \'prior\' parameter has been deprecated.', DeprecationWarning)

        if 'variance' in kwargs:
            warnings.warn('The \'variance\' parameter has been deprecated. Please use the '
                          '\'merit\' parameter instead.', DeprecationWarning)

        if 'sigma' in kwargs:
            self.m_sigma = kwargs['sigma']
        else:
            self.m_sigma = (1e-5, 1e-3, 1e-3)

        super(MCMCsamplingModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_chain_out_port = self.add_output_port(chain_out_tag)

        self.m_param = param
        self.m_bounds = bounds
        self.m_nwalkers = nwalkers
        self.m_nsteps = nsteps
        self.m_psf_scaling = psf_scaling
        self.m_pca_number = pca_number
        self.m_aperture = aperture
        self.m_extra_rot = extra_rot
        self.m_merit = merit
        self.m_residuals = residuals

        if mask is None:
            self.m_mask = (None, None)
        else:
            self.m_mask = mask

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. The posterior distributions of the separation, position angle,
        and flux contrast are sampled with the affine invariant Markov chain Monte Carlo (MCMC)
        ensemble sampler emcee. At each step, a negative copy of the PSF template is injected
        and the likelihood function is evaluated at the approximate position of the planet.

        Returns
        -------
        NoneType
            None
        """

        ndim = 3

        cpu = self._m_config_port.get_attribute('CPU')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        parang = self.m_image_in_port.get_attribute('PARANG')

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        im_shape = self.m_image_in_port.get_shape()[-2:]

        self.m_image_in_port.close_port()
        self.m_psf_in_port.close_port()

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number of '
                             'frames in image_in_tag. The DerotateAndStackModule can be used to '
                             'average the PSF frames (without derotating) before applying the '
                             'MCMCsamplingModule.')

        if self.m_mask[0] is not None:
            self.m_mask = (self.m_mask[0]/pixscale, self.m_mask[1])

        if self.m_mask[1] is not None:
            self.m_mask = (self.m_mask[0], self.m_mask[1]/pixscale)

        # create the mask and get the unmasked image indices
        mask = create_mask(im_shape[-2:], self.m_mask)
        indices = np.where(mask.reshape(-1) != 0.)[0]

        if isinstance(self.m_aperture, float):
            yx_pos = polar_to_cartesian(images, self.m_param[0]/pixscale, self.m_param[1])
            aperture = (int(round(yx_pos[0])), int(round(yx_pos[1])), self.m_aperture/pixscale)

        elif isinstance(self.m_aperture, tuple):
            aperture = (self.m_aperture[1], self.m_aperture[0], self.m_aperture[2]/pixscale)

        if self.m_merit == 'poisson':
            noise = None

        elif self.m_merit == 'gaussian':
            noise = gaussian_noise(images=images,
                                   parang=parang,
                                   cent_size=self.m_mask[0],
                                   edge_size=self.m_mask[1],
                                   pca_number=self.m_pca_number,
                                   residuals=self.m_residuals,
                                   aperture=aperture)

        initial = np.zeros((self.m_nwalkers, ndim))

        initial[:, 0] = self.m_param[0] + np.random.normal(0, self.m_sigma[0], self.m_nwalkers)
        initial[:, 1] = self.m_param[1] + np.random.normal(0, self.m_sigma[1], self.m_nwalkers)
        initial[:, 2] = self.m_param[2] + np.random.normal(0, self.m_sigma[2], self.m_nwalkers)

        print('Sampling the posterior distributions with MCMC...')

        with Pool(processes=cpu) as pool:
            sampler = emcee.EnsembleSampler(self.m_nwalkers,
                                            ndim,
                                            lnprob,
                                            pool=pool,
                                            args=([self.m_bounds,
                                                   images,
                                                   psf,
                                                   mask,
                                                   parang,
                                                   self.m_psf_scaling,
                                                   pixscale,
                                                   self.m_pca_number,
                                                   self.m_extra_rot,
                                                   aperture,
                                                   indices,
                                                   self.m_merit,
                                                   self.m_residuals,
                                                   noise]))

            sampler.run_mcmc(initial, self.m_nsteps, progress=True)

        samples = sampler.get_chain()

        self.m_image_in_port._check_status_and_activate()
        self.m_chain_out_port._check_status_and_activate()

        self.m_chain_out_port.set_all(samples)
        print(f'Number of samples stored: {samples.shape[0]*samples.shape[1]}')

        burnin = int(0.2*samples.shape[0])
        samples = samples[burnin:, :, :].reshape((-1, ndim))

        sep_percen = np.percentile(samples[:, 0], [16., 50., 84.])
        ang_percen = np.percentile(samples[:, 1], [16., 50., 84.])
        mag_percen = np.percentile(samples[:, 2], [16., 50., 84.])

        print('Median and uncertainties (20% removed as burnin):')

        print(f'Separation [mas] = {1e3*sep_percen[1]:.2f} '
              f'(-{1e3*sep_percen[1]-1e3*sep_percen[0]:.2f} '
              f'+{1e3*sep_percen[2]-1e3*sep_percen[1]:.2f})')

        print(f'Position angle [deg] = {ang_percen[1]:.2f} '
              f'(-{ang_percen[1]-ang_percen[0]:.2f} '
              f'+{ang_percen[2]-ang_percen[1]:.2f})')

        print(f'Contrast [mag] = {mag_percen[1]:.2f} '
              f'(-{mag_percen[1]-mag_percen[0]:.2f} '
              f'+{mag_percen[2]-mag_percen[1]:.2f})')

        history = f'walkers = {self.m_nwalkers}, steps = {self.m_nsteps}'
        self.m_chain_out_port.copy_attributes(self.m_image_in_port)
        self.m_chain_out_port.add_history('MCMCsamplingModule', history)

        mean_accept = np.mean(sampler.acceptance_fraction)
        print(f'Mean acceptance fraction: {mean_accept:.3f}')
        self.m_chain_out_port.add_attribute('ACCEPTANCE', mean_accept, static=True)

        try:
            autocorr = emcee.autocorr.integrated_time(sampler.get_chain())
            print(f'Integrated autocorrelation time ={autocorr}')

        except emcee.autocorr.AutocorrError:
            autocorr = [np.nan, np.nan, np.nan]
            print('The chain is too short to reliably estimate the autocorrelation time. [WARNING]')

        self.m_chain_out_port.add_attribute('AUTOCORR_0', autocorr[0], static=True)
        self.m_chain_out_port.add_attribute('AUTOCORR_1', autocorr[1], static=True)
        self.m_chain_out_port.add_attribute('AUTOCORR_2', autocorr[2], static=True)

        self.m_chain_out_port.close_port()


class AperturePhotometryModule(ProcessingModule):
    """
    Pipeline module for calculating the counts within a circular area.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 phot_out_tag: str,
                 radius: float = 0.1,
                 position: Tuple[float, float] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        phot_out_tag : str
            Tag of the database entry with the photometry values that are written as output.
        radius : float
            Radius (arcsec) of the circular aperture.
        position : tuple(float, float), None
            Center position (pix) of the aperture, (x, y), with subpixel precision. The center of
            the image will be used if set to None. Python indexing starts at zero so the bottom
            left corner of the image has coordinates (-0.5, -0.5).

        Returns
        -------
        NoneType
            None
        """

        super(AperturePhotometryModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_phot_out_port = self.add_output_port(phot_out_tag)

        self.m_radius = radius
        self.m_position = position

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Computes the flux within a circular aperture for each
        frame and saves the values in the database.

        Returns
        -------
        NoneType
            None
        """

        def _photometry(image, aperture):
            # https://photutils.readthedocs.io/en/stable/overview.html
            # In Photutils, pixel coordinates are zero-indexed, meaning that (x, y) = (0, 0)
            # corresponds to the center of the lowest, leftmost array element. This means that
            # the value of data[0, 0] is taken as the value over the range -0.5 < x <= 0.5,
            # -0.5 < y <= 0.5. Note that this is the same coordinate system as used by PynPoint.

            return aperture_photometry(image, aperture, method='exact')['aperture_sum']

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        self.m_radius /= pixscale

        if self.m_position is None:
            self.m_position = center_subpixel(self.m_image_in_port[0, ])

        # Position in CircularAperture is defined as (x, y)
        aperture = CircularAperture((self.m_position[1], self.m_position[0]), self.m_radius)

        self.apply_function_to_images(_photometry,
                                      self.m_image_in_port,
                                      self.m_phot_out_port,
                                      'Aperture photometry',
                                      func_args=(aperture, ))

        history = f'radius [arcsec] = {self.m_radius*pixscale:.3f}'
        self.m_phot_out_port.copy_attributes(self.m_image_in_port)
        self.m_phot_out_port.add_history('AperturePhotometryModule', history)
        self.m_phot_out_port.close_port()


class SystematicErrorModule(ProcessingModule):
    """
    Pipeline module for estimating the systematic error of the flux and position measurement.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 offset_out_tag: str,
                 position: Tuple[float, float],
                 magnitude: float,
                 angles: Tuple[float, float, int] = (0., 359., 360),
                 psf_scaling: float = 1.,
                 merit: str = 'gaussian',
                 aperture: float = 0.1,
                 tolerance: float = 0.01,
                 pca_number: int = 10,
                 mask: Union[Tuple[float, float], Tuple[None, float],
                             Tuple[float, None], Tuple[None, None]] = None,
                 extra_rot: float = 0.,
                 residuals: str = 'median',
                 offset: float = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images for which the systematic error is
            estimated.
        psf_in_tag : str
            Tag of the database entry with the PSF template that is used as fake planet. Can be
            either a single image or a stack of images equal in size to ``image_in_tag``.
        offset_out_tag : str
            Tag of the database entry at which the differences are stored between the injected and
            and retrieved values of the separation (arcsec), position angle (deg), and contrast
            (mag).
        position : tuple(float, float)
            Separation (arcsec) and position angle (deg) that are used to remove the planet signal.
            The separation is also used to estimate the systematic error.
        magnitude : float
            Magnitude that is used to remove the planet signal and estimate the systematic error.
        angles : tuple(float, float, int)
            The start, end, and number of the position angles (linearly sampled) that are used to
            estimate the systematic errors (default: 0., 359., 360). The endpoint is also included.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be a positive value.
        merit : str
            Figure of merit that is used for the likelihood function ('gaussian' or 'poisson').
            Pixels are assumed to be independent measurements which are expected to be equal to
            zero in case the best-fit negative PSF template is injected. With 'gaussian', the
            variance is estimated from the pixel values within an annulus at the separation of
            the aperture (but excluding the pixels within the aperture). With 'poisson', a Poisson
            distribution is assumed for the variance of each pixel value (see Wertz et al. 2017).
        aperture : float
            Aperture radius (arcsec) that is used for measuring the figure of merit.
        tolerance : float
            Absolute error on the input parameters, position (pix) and contrast (mag), that is used
            as acceptance level for convergence. Note that only a single value can be specified
            which is used for both the position and flux so tolerance=0.1 will give a precision of
            0.1 mag and 0.1 pix. The tolerance on the output (i.e., the chi-square value) is set to
            np.inf so the condition is always met.
        pca_number : int
            Number of principal components (PCs) used for the PSF subtraction.
        mask : tuple(float, float), None
            Inner and outer mask radius (arcsec) which is applied before the PSF subtraction. Both
            elements of the tuple can be set to None.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        offset : float, None
            Offset (pix) by which the negative PSF may deviate from the positive injected PSF. No
            constraint on the position is applied if set to None.

        Returns
        -------
        NoneType
            None
        """

        super(SystematicErrorModule, self).__init__(name_in)

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_offset_out_port = self.add_output_port(offset_out_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_angles = angles
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_tolerance = tolerance
        self.m_mask = mask
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals
        self.m_pca_number = pca_number
        self.m_offset = offset

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Removes the planet signal, then artificial planets are injected
        (one at a time) at equally separated position angles and their position and contrast is
        determined with the :class:`~pynpoint.processing.fluxposition.SimplexMinimizationModule`.
        The differences between the injected and retrieved separation, position angle, and contrast
        are then stored as output.

        Returns
        -------
        NoneType
            None
        """

        self.m_offset_out_port.del_all_data()
        self.m_offset_out_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        image = self.m_image_in_port[0, ]

        module = FakePlanetModule(name_in=f'{self._m_name}_fake',
                                  image_in_tag=self.m_image_in_tag,
                                  psf_in_tag=self.m_psf_in_tag,
                                  image_out_tag=f'{self._m_name}_empty',
                                  position=self.m_position,
                                  magnitude=self.m_magnitude,
                                  psf_scaling=-self.m_psf_scaling)

        module.connect_database(self._m_data_base)
        module.run()

        sep = float(self.m_position[0])
        angles = np.linspace(self.m_angles[0], self.m_angles[1], self.m_angles[2], endpoint=True)

        for i, ang in enumerate(angles):
            print(f'Processing position angle: {ang} deg...')

            module = FakePlanetModule(position=(sep, ang),
                                      magnitude=self.m_magnitude,
                                      psf_scaling=self.m_psf_scaling,
                                      name_in=f'{self._m_name}_fake_{i}',
                                      image_in_tag=f'{self._m_name}_empty',
                                      psf_in_tag=self.m_psf_in_tag,
                                      image_out_tag=f'{self._m_name}_fake')

            module.connect_database(self._m_data_base)
            module.run()

            position = polar_to_cartesian(image, sep/pixscale, ang)
            position = (int(round(position[1])), int(round(position[0])))

            module = SimplexMinimizationModule(position=position,
                                               magnitude=self.m_magnitude,
                                               psf_scaling=-self.m_psf_scaling,
                                               name_in=f'{self._m_name}_fake_{i}',
                                               image_in_tag=f'{self._m_name}_fake',
                                               psf_in_tag=self.m_psf_in_tag,
                                               res_out_tag=f'{self._m_name}_simplex',
                                               flux_position_tag=f'{self._m_name}_fluxpos',
                                               merit=self.m_merit,
                                               aperture=self.m_aperture,
                                               sigma=0.,
                                               tolerance=self.m_tolerance,
                                               pca_number=self.m_pca_number,
                                               cent_size=self.m_mask[0],
                                               edge_size=self.m_mask[1],
                                               extra_rot=self.m_extra_rot,
                                               residuals='median',
                                               offset=self.m_offset)

            module.connect_database(self._m_data_base)
            module.run()

            fluxpos_out_port = self.add_input_port(f'{self._m_name}_fluxpos')

            data = [self.m_position[0] - fluxpos_out_port[-1, 2],
                    ang - fluxpos_out_port[-1, 3],
                    self.m_magnitude - fluxpos_out_port[-1, 4]]

            if data[1] > 180.:
                data[1] -= 360.
            elif data[1] < -180.:
                data[1] += 360.

            print(f'Offset: {data[0]*1e3:.2f} mas, {data[1]:.2f} deg, {data[2]:.2f} mag')

            self.m_offset_out_port.append(data, data_dim=2)

        offset_in_port = self.add_input_port(self.m_offset_out_port.tag)
        offsets = offset_in_port.get_all()

        sep_percen = np.percentile(offsets[:, 0], [16., 50., 84.])
        ang_percen = np.percentile(offsets[:, 1], [16., 50., 84.])
        mag_percen = np.percentile(offsets[:, 2], [16., 50., 84.])

        print('Median and uncertainties:')

        print(f'Separation [mas] = {1e3*sep_percen[1]:.2f} '
              f'(-{1e3*sep_percen[1]-1e3*sep_percen[0]:.2f} '
              f'+{1e3*sep_percen[2]-1e3*sep_percen[1]:.2f})')

        print(f'Position angle [deg] = {ang_percen[1]:.2f} '
              f'(-{ang_percen[1]-ang_percen[0]:.2f} '
              f'+{ang_percen[2]-ang_percen[1]:.2f})')

        print(f'Contrast [mag] = {mag_percen[1]:.2f} '
              f'(-{mag_percen[1]-mag_percen[0]:.2f} '
              f'+{mag_percen[2]-mag_percen[1]:.2f})')

        history = f'sep = {self.m_position[0]:.3f}, ' \
                  f'pa = {self.m_position[1]:.1f}, ' \
                  f'mag = {self.m_magnitude:.1f}'

        self.m_offset_out_port.copy_attributes(self.m_image_in_port)
        self.m_offset_out_port.add_history('SystematicErrorModule', history)
        self.m_offset_out_port.close_port()
