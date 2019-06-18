"""
Pipeline modules for photometric and astrometric measurements.
"""

import sys
import time
import warnings

from typing import Union, Tuple

import numpy as np
import emcee

from typeguard import typechecked
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from photutils import aperture_photometry, CircularAperture

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.analysis import fake_planet, merit_function, false_alarm
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
            raise ValueError('The images in \'{self.m_image_in_port.tag}\' should have the same '
                             'dimensions as the images images in \'{self.m_psf_in_port.tag}\'.')

        frames = memory_frames(memory, im_shape[0])

        start_time = time.time()
        for j, _ in enumerate(frames[:-1]):
            progress(j, len(frames[:-1]), 'Running FakePlanetModule...', start_time)

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

        sys.stdout.write('Running FakePlanetModule... [DONE]\n')
        sys.stdout.flush()

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
                 pca_number: int = 20,
                 cent_size: float = None,
                 edge_size: float = None,
                 extra_rot: float = 0.,
                 residuals: str = 'median',
                 reference_in_tag: str = None) -> None:
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
        pca_number : int
            Number of principal components used for the PSF subtraction.
        cent_size : float
            Radius of the central mask (arcsec). No mask is used when set to None.
        edge_size : float
            Outer radius (arcsec) beyond which pixels are masked. No outer mask is used when set to
            None. The radius will be set to half the image size if the argument is larger than half
            the image size.
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

        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_flux_position_port = self.add_output_port(flux_position_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_pca_number = pca_number
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals

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

        self.m_res_out_port.del_all_data()
        self.m_res_out_port.del_all_attributes()

        self.m_flux_position_port.del_all_data()
        self.m_flux_position_port.del_all_attributes()

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

        if self.m_reference_in_port is not None:
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
            sklearn_pca = PCA(n_components=self.m_pca_number, svd_solver='arpack')
            sklearn_pca.fit(ref_reshape)

            # add mean of reference array as 1st PC and orthogonalize it to the PCA basis
            mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

            q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
                                                 sklearn_pca.components_[:-1, ])).T)

            sklearn_pca.components_ = q_ortho.T

        def _objective(arg):
            sys.stdout.write('.')
            sys.stdout.flush()

            pos_y = arg[0]
            pos_x = arg[1]
            mag = arg[2]

            sep_ang = cartesian_to_polar(center, pos_y, pos_x)

            fake = fake_planet(images=images,
                               psf=psf,
                               parang=parang,
                               position=(sep_ang[0], sep_ang[1]),
                               magnitude=mag,
                               psf_scaling=self.m_psf_scaling)

            mask = create_mask(fake.shape[-2:], (self.m_cent_size, self.m_edge_size))

            if self.m_reference_in_port is None:
                _, im_res = pca_psf_subtraction(images=fake*mask,
                                                angles=-1.*parang+self.m_extra_rot,
                                                pca_number=self.m_pca_number,
                                                pca_sklearn=None,
                                                im_shape=None,
                                                indices=None)

            else:
                im_reshape = np.reshape(fake*mask, (im_shape[0], im_shape[1]*im_shape[2]))

                _, im_res = pca_psf_subtraction(images=im_reshape,
                                                angles=-1.*parang+self.m_extra_rot,
                                                pca_number=self.m_pca_number,
                                                pca_sklearn=sklearn_pca,
                                                im_shape=im_shape,
                                                indices=None)

            res_stack = combine_residuals(method=self.m_residuals, res_rot=im_res)

            self.m_res_out_port.append(res_stack, data_dim=3)

            chi_square = merit_function(residuals=res_stack[0, ],
                                        merit=self.m_merit,
                                        aperture=aperture,
                                        sigma=self.m_sigma)

            position = rotate_coordinates(center, (pos_y, pos_x), -self.m_extra_rot)

            res = np.asarray([position[1],
                              position[0],
                              sep_ang[0]*pixscale,
                              (sep_ang[1]-self.m_extra_rot) % 360.,
                              mag,
                              chi_square])

            self.m_flux_position_port.append(res, data_dim=2)

            return chi_square

        sys.stdout.write('Running SimplexMinimizationModule')
        sys.stdout.flush()

        pos_init = rotate_coordinates(center,
                                      (self.m_position[1], self.m_position[0]),  # (y, x)
                                      self.m_extra_rot)

        minimize(fun=_objective,
                 x0=[pos_init[0], pos_init[1], self.m_magnitude],
                 method='Nelder-Mead',
                 tol=None,
                 options={'xatol': self.m_tolerance, 'fatol': float('inf')})

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        history = f'merit = {self.m_merit}'
        self.m_flux_position_port.copy_attributes(self.m_image_in_port)
        self.m_flux_position_port.add_history('SimplexMinimizationModule', history)

        self.m_res_out_port.copy_attributes(self.m_image_in_port)
        self.m_res_out_port.add_history('SimplexMinimizationModule', history)
        self.m_res_out_port.close_port()


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
                 **kwargs: Union[float, Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
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
            Optimize the SNR. The aperture position is written in the history. The size of the
            aperture is kept fixed.

        Keyword arguments
        -----------------
        tolerance : float
            The fractional tolerance on the position for the optimization to end. Default is set
            to 1e-3.
        bounds : tuple(tuple(float, float), tuple(float, float))
            Boundaries (pix) for the horizontal and vertical offset with respect to the `position`.
            The default is set to (-3, 3) for both directions.

        Returns
        -------
        NoneType
            None
        """

        if 'tolerance' in kwargs:
            self.m_tolerance = kwargs['tolerance']
        else:
            self.m_tolerance = 1e-3

        if 'bounds' in kwargs:
            self.m_bounds = kwargs['bounds']
        else:
            self.m_bounds = ((-3., 3.), (-3., 3.))

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

        bounds = ((self.m_position[0]+self.m_bounds[0][0], self.m_position[0]+self.m_bounds[0][1]),
                  (self.m_position[1]+self.m_bounds[1][0], self.m_position[1]+self.m_bounds[1][1]))

        start_time = time.time()
        for j in range(nimages):
            progress(j, nimages, 'Running FalsePositiveModule...', start_time)

            image = self.m_image_in_port[j, ]
            center = center_subpixel(image)

            if self.m_optimize:
                result = minimize(fun=_snr_optimize,
                                  x0=[self.m_position[0], self.m_position[1]],
                                  method='SLSQP',
                                  bounds=bounds,
                                  tol=None,
                                  options={'ftol': self.m_tolerance})

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

        sys.stdout.write('Running FalsePositiveModule... [DONE]\n')
        sys.stdout.flush()

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
                 mask: Tuple[float, float] = None,
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
            of the array is (nwalkers*nsteps, 3).
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
        mask : tuple(float, float)
            Inner and outer mask radius (arcsec) for the PSF subtraction. Both elements of the
            tuple can be set to None. Masked pixels are excluded from the PCA computation,
            resulting in a smaller runtime.
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
        scale : float
            The proposal scale parameter (Goodman & Weare 2010). The default is set to 2.
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

        if 'scale' in kwargs:
            self.m_scale = kwargs['scale']
        else:
            self.m_scale = 2.

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

    # @typechecked
    # def gaussian_variance(self,
    #                       images: np.ndarray,
    #                       psf: np.ndarray,
    #                       parang: np.ndarray,
    #                       aperture: Tuple[int, int, float]) -> float:
    #     """
    #     Function to compute the (constant) variance for the likelihood function when the
    #     merit parameter is set to 'gaussian'. The planet is first removed from the dataset
    #     with the `param` values.
    #
    #     Parameters
    #     ----------
    #     images : numpy.ndarray
    #         Masked input images.
    #     psf : numpy.ndarray
    #         PSF template.
    #     parang : numpy.ndarray
    #         Parallactic angles (deg).
    #     aperture : tuple(int, int, float)
    #         Properties of the circular aperture. The radius is recommended to be larger than or
    #         equal to 0.5*lambda/D.
    #
    #     Returns
    #     -------
    #     float
    #         Variance (counts).
    #     """
    #
    #     pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
    #
    #     fake = fake_planet(images=images,
    #                        psf=psf,
    #                        parang=parang,
    #                        position=(self.m_param[0]/pixscale, self.m_param[1]),
    #                        magnitude=self.m_param[2],
    #                        psf_scaling=self.m_psf_scaling)
    #
    #     _, res_arr = pca_psf_subtraction(images=fake,
    #                                      angles=-1.*parang+self.m_extra_rot,
    #                                      pca_number=self.m_pca_number)
    #
    #     res_stack = combine_residuals(method=self.m_residuals, res_rot=res_arr)
    #
    #     # separation (pix) and position angle (deg)
    #     sep_ang = cartesian_to_polar(center=center_subpixel(res_stack),
    #                                  y_pos=aperture[0],
    #                                  x_pos=aperture[1])
    #
    #     selected = select_annulus(image_in=res_stack[0, ],
    #                               radius_in=sep_ang[0]-aperture[2],
    #                               radius_out=sep_ang[0]+aperture[2],
    #                               mask_position=aperture[0:2],
    #                               mask_radius=aperture[2])
    #
    #     return np.var(selected)

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

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number of '
                             'frames in image_in_tag. The DerotateAndStackModule can be used to '
                             'average the PSF frames (without derotating) before applying the '
                             'MCMCsamplingModule.')

        im_shape = self.m_image_in_port.get_shape()[-2:]

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

        initial = np.zeros((self.m_nwalkers, ndim))

        initial[:, 0] = self.m_param[0] + np.random.normal(0, self.m_sigma[0], self.m_nwalkers)
        initial[:, 1] = self.m_param[1] + np.random.normal(0, self.m_sigma[1], self.m_nwalkers)
        initial[:, 2] = self.m_param[2] + np.random.normal(0, self.m_sigma[2], self.m_nwalkers)

        # if self.m_merit == 'gaussian':
        #     variance = self.gaussian_variance(images*mask, psf, parang, aperture)
        # else:
        #     variance = None

        sampler = emcee.EnsembleSampler(nwalkers=self.m_nwalkers,
                                        dim=ndim,
                                        lnpostfn=lnprob,
                                        a=self.m_scale,
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
                                               self.m_residuals]),
                                        threads=cpu)

        start_time = time.time()
        for i, _ in enumerate(sampler.sample(p0=initial, iterations=self.m_nsteps)):
            progress(i, self.m_nsteps, 'Running MCMCsamplingModule...', start_time)

        sys.stdout.write('Running MCMCsamplingModule... [DONE]\n')
        sys.stdout.flush()

        self.m_chain_out_port.set_all(sampler.chain)

        history = f'walkers = {self.m_nwalkers}, steps = {self.m_nsteps}'
        self.m_chain_out_port.copy_attributes(self.m_image_in_port)
        self.m_chain_out_port.add_history('MCMCsamplingModule', history)
        self.m_chain_out_port.close_port()

        print(f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}')

        try:
            autocorr = emcee.autocorr.integrated_time(sampler.flatchain,
                                                      low=10,
                                                      high=None,
                                                      step=1,
                                                      c=10,
                                                      full_output=False,
                                                      axis=0,
                                                      fast=False)

            print('Integrated autocorrelation time =', autocorr)

        except emcee.autocorr.AutocorrError:
            print('The chain is too short to reliably estimate the autocorrelation time. [WARNING]')


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
                                      'Running AperturePhotometryModule',
                                      func_args=(aperture, ))

        history = f'radius [arcsec] = {self.m_radius*pixscale:.3f}'
        self.m_phot_out_port.copy_attributes(self.m_image_in_port)
        self.m_phot_out_port.add_history('AperturePhotometryModule', history)
        self.m_phot_out_port.close_port()
