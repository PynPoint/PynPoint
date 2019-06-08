"""
Pipeline modules for photometric and astrometric measurements.
"""

import sys
import time

from typing import Union, Tuple

import numpy as np
import emcee

from typeguard import typechecked
from scipy.optimize import minimize
from photutils import aperture_photometry, CircularAperture
from sklearn.decomposition import PCA

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.analysis import fake_planet, merit_function, false_alarm
from pynpoint.util.image import create_mask, polar_to_cartesian, cartesian_to_polar, \
                                center_subpixel
from pynpoint.util.mcmc import lnprob
from pynpoint.util.module import progress, memory_frames, rotate_coordinates
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals


class FakePlanetModule(ProcessingModule):
    """
    Pipeline module to inject a positive or negative artificial planet into a stack of images.
    """

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
    and minimizing a function of merit.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 res_out_tag: str,
                 flux_position_tag: str,
                 position: Tuple[float, float],
                 magnitude: float,
                 psf_scaling: float = -1.,
                 merit: str = 'hessian',
                 aperture: float = 0.1,
                 sigma: float = 0.027,
                 tolerance: float = 0.1,
                 pca_number: int = 20,
                 cent_size: float = None,
                 edge_size: float = None,
                 extra_rot: float = 0.,
                 residuals: str = 'mean',
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
            either a single image (2D) or a cube (3D) with the dimensions equal to *image_in_tag*.
        res_out_tag : str
            Tag of the database entry with the image residuals that are written as output. Contains
            the results from the PSF subtraction during the minimization of the function of merit.
            The last image is the image with the best-fit residuals.
        flux_position_tag : str
            Tag of the database entry with flux and position results that are written as output.
            Each step of the minimization saves the x position (pix), y position (pix), separation
            (arcsec), angle (deg), contrast (mag), and the function of merit. The last row of
            values contain the best-fit results.
        position : tuple(float, float)
            Approximate position (x, y) of the planet (pix). This is also the location where the
            function of merit is calculated with an aperture of radius *aperture*.
        magnitude : float
            Approximate magnitude of the planet relative to the star.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be negative in order to inject negative fake planets.
        merit : str
            Function of merit for the minimization. Can be either *hessian*, to minimize the sum of
            the absolute values of the determinant of the Hessian matrix, or *sum*, to minimize the
            sum of the absolute pixel values (Wertz et al. 2017).
        aperture : float
            Either the aperture radius (arcsec) at the position specified at *position* or a
            dictionary with the aperture properties. See
            :class:`~pynpoint.util.analysis.create_aperture` for details.
        sigma : float
            Standard deviation (arcsec) of the Gaussian kernel which is used to smooth the images
            before the function of merit is calculated (in order to reduce small pixel-to-pixel
            variations).
        tolerance : float
            Absolute error on the input parameters, position (pix) and contrast (mag), that is used
            as acceptance level for convergence. Note that only a single value can be specified
            which is used for both the position and flux so tolerance=0.1 will give a precision of
            0.1 mag and 0.1 pix. The tolerance on the output (i.e., function of merit) is set to
            np.inf so the condition is always met.
        pca_number : int
            Number of principal components used for the PSF subtraction.
        cent_size : float
            Radius of the central mask (arcsec). No mask is used when set to None.
        edge_size : float
            Outer radius (arcsec) beyond which pixels are masked. No outer mask is used when set to
            None. The radius will be set to half the image size if the *edge_size* value is larger
            than half the image size.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
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
        Run method of the module. The position and flux of a planet are measured by injecting
        negative fake companions and applying a simplex method (Nelder-Mead) for minimization
        of a function of merit at the planet location. The default function of merit is the
        image curvature which is calculated as the sum of the absolute values of the
        determinant of the Hessian matrix.

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

        if isinstance(self.m_aperture, float):
            self.m_aperture = {'type':'circular',
                               'pos_x':self.m_position[0],
                               'pos_y':self.m_position[1],
                               'radius':self.m_aperture/pixscale}

        elif isinstance(self.m_aperture, dict):
            if self.m_aperture['type'] == 'circular':
                self.m_aperture['radius'] /= pixscale

            elif self.m_aperture['type'] == 'elliptical':
                self.m_aperture['semimajor'] /= pixscale
                self.m_aperture['semiminor'] /= pixscale

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

            sep_ang = cartesian_to_polar(center, pos_x, pos_y)

            fake = fake_planet(images=images,
                               psf=psf,
                               parang=parang,
                               position=(sep_ang[0], sep_ang[1]),
                               magnitude=mag,
                               psf_scaling=self.m_psf_scaling)

            mask_shape = (fake.shape[-2], fake.shape[-1])
            mask = create_mask(mask_shape, [self.m_cent_size, self.m_edge_size])

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

            stack = combine_residuals(method=self.m_residuals, res_rot=im_res)

            self.m_res_out_port.append(stack, data_dim=3)

            merit = merit_function(residuals=stack[0, ],
                                   function=self.m_merit,
                                   variance='poisson',
                                   aperture=self.m_aperture,
                                   sigma=self.m_sigma)

            position = rotate_coordinates(center, (pos_y, pos_x), -self.m_extra_rot)

            res = np.asarray((position[1],
                              position[0],
                              sep_ang[0]*pixscale,
                              (sep_ang[1]-self.m_extra_rot)%360.,
                              mag,
                              merit))

            self.m_flux_position_port.append(res, data_dim=2)

            return merit

        sys.stdout.write('Running SimplexMinimizationModule')
        sys.stdout.flush()

        pos_init = rotate_coordinates(center,
                                      (self.m_position[1], self.m_position[0]),
                                      self.m_extra_rot)

        # Change integer to float?
        pos_init = (int(pos_init[0]), int(pos_init[1])) # (y, x)

        minimize(fun=_objective,
                 x0=[pos_init[0], pos_init[1], self.m_magnitude],
                 method='Nelder-Mead',
                 tol=None,
                 options={'xatol':self.m_tolerance, 'fatol':float('inf')})

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

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 snr_out_tag: str,
                 position: Tuple[float, float],
                 aperture: float = 0.1,
                 ignore: bool = False,
                 optimize: bool = False,
                 **kwargs: float) -> None:
        """
        Parameters
        ----------
        name_in: str
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
            The absolute tolerance (pix) on the position for the optimization to end. Default is
            set to 0.01 pix.

        Returns
        -------
        NoneType
            None
        """

        if 'tolerance' in kwargs:
            self.m_tolerance = kwargs['tolerance']
        else:
            self.m_tolerance = 1e-2

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

        def _fpf_minimize(arg):
            pos_x, pos_y = arg

            try:
                _, _, _, fpf = false_alarm(image=image,
                                           x_pos=pos_x,
                                           y_pos=pos_y,
                                           size=self.m_aperture,
                                           ignore=self.m_ignore)

            except ValueError:
                fpf = float('inf')

            return fpf

        self.m_snr_out_port.del_all_data()
        self.m_snr_out_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        self.m_aperture /= pixscale

        nimages = self.m_image_in_port.get_shape()[0]

        start_time = time.time()
        for j in range(nimages):
            progress(j, nimages, 'Running FalsePositiveModule...', start_time)

            image = self.m_image_in_port[j, ]
            center = center_subpixel(image)

            if self.m_optimize:
                result = minimize(fun=_fpf_minimize,
                                  x0=[self.m_position[0], self.m_position[1]],
                                  method='Nelder-Mead',
                                  tol=None,
                                  options={'xatol':self.m_tolerance, 'fatol':float('inf')})

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

            sep_ang = cartesian_to_polar(center, x_pos, y_pos)
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
                 aperture: Union[float, dict] = 0.1,
                 mask: Tuple[float, float] = None,
                 extra_rot: float = 0.,
                 prior: str = 'flat',
                 variance: str = 'poisson',
                 residuals: str = 'mean',
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
            fixed position for the aperture if *aperture* contains a single value. Furthermore,
            the values are used to remove the planet signal before the noise is estimated when
            *variance* is set to 'gaussian' to prevent that self-subtraction lobes bias the noise
            measurement.
        bounds : tuple(tuple(float, float), tuple(float, float), tuple(float, float))
            The boundaries of the separation (arcsec), angle (deg), and contrast (mag). Each set
            of boundaries is specified as a tuple.
        nwalkers : int
            Number of ensemble members (i.e. chains).
        nsteps : int
            Number of steps to run per walker.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be negative in order to inject negative fake planets.
        pca_number : int
            Number of principal components used for the PSF subtraction.
        aperture : float or dict
            Either the aperture radius (arcsec) at the position specified in *param* or a
            dictionary with the aperture properties. See for more information
            :class:`~pynpoint.util.analysis.create_aperture`.
        mask : tuple(float, float)
            Inner and outer mask radius (arcsec) for the PSF subtraction. Both elements of the
            tuple can be set to None. Masked pixels are excluded from the PCA computation,
            resulting in a smaller runtime.
        extra_rot : float
            Additional rotation angle of the images (deg).
        prior : str
            Prior can be set to 'flat' or 'aperture'. With 'flat', the values of *bounds* are used
            as uniform priors. With 'aperture', the prior probability is set to zero beyond the
            aperture and unity within the aperture.
        variance : str
            Variance used in the likelihood function ('poisson' or 'gaussian').
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
        self.m_prior = prior
        self.m_variance = variance
        self.m_residuals = residuals

        if mask is None:
            self.m_mask = np.array((None, None))
        else:
            self.m_mask = np.array(mask)

    @typechecked
    def aperture_dict(self,
                      images: np.ndarray) -> None:
        """
        Function to create or update the dictionary with aperture properties.

        Parameters
        ----------
        images : numpy.ndarray
            Input images.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        if isinstance(self.m_aperture, float):
            xy_pos = polar_to_cartesian(images, self.m_param[0]/pixscale, self.m_param[1])

            self.m_aperture = {'type':'circular',
                               'pos_x':xy_pos[0],
                               'pos_y':xy_pos[1],
                               'radius':self.m_aperture/pixscale}

        elif isinstance(self.m_aperture, dict):
            if self.m_aperture['type'] == 'circular':
                self.m_aperture['radius'] /= pixscale

            elif self.m_aperture['type'] == 'elliptical':
                self.m_aperture['semimajor'] /= pixscale
                self.m_aperture['semiminor'] /= pixscale

        if self.m_variance == 'gaussian' and self.m_aperture['type'] != 'circular':
            raise ValueError('Gaussian variance can only be used in combination with a'
                             'circular aperture.')

    @typechecked
    def gaussian_noise(self,
                       images: np.ndarray,
                       psf: np.ndarray,
                       parang: np.ndarray,
                       aperture: dict) -> float:
        """
        Function to compute the (constant) variance for the likelihood function when the
        variance parameter is set to gaussian (see Mawet et al. 2014). The planet is first removed
        from the dataset with the values specified as *param* in the constructor of the instance.

        Parameters
        ----------
        images : numpy.ndarray
            Input images.
        psf : numpy.ndarray
            PSF template.
        parang : numpy.ndarray
            Parallactic angles (deg).
        aperture : dict
            Properties of the circular aperture. The radius is recommended to be larger than or
            equal to 0.5*lambda/D.

        Returns
        -------
        float
            Variance.
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        fake = fake_planet(images=images,
                           psf=psf,
                           parang=parang,
                           position=(self.m_param[0]/pixscale, self.m_param[1]),
                           magnitude=self.m_param[2],
                           psf_scaling=self.m_psf_scaling)

        _, res_arr = pca_psf_subtraction(images=fake,
                                         angles=-1.*parang+self.m_extra_rot,
                                         pca_number=self.m_pca_number)

        stack = combine_residuals(method=self.m_residuals, res_rot=res_arr)

        _, noise, _, _ = false_alarm(image=stack[0, ],
                                     x_pos=aperture['pos_x'],
                                     y_pos=aperture['pos_y'],
                                     size=aperture['radius'],
                                     ignore=False)

        return noise**2

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

        if not isinstance(self.m_param, tuple) or len(self.m_param) != 3:
            raise TypeError('The param argument should contain a tuple with the approximate '
                            'separation (arcsec), position angle (deg), and contrast (mag).')

        if not isinstance(self.m_bounds, tuple) or len(self.m_bounds) != 3:
            raise TypeError('The bounds argument should contain a tuple with three tuples for '
                            'the boundaries of the separation (arcsec), position angle (deg), and '
                            'contrast (mag).')

        if not isinstance(self.m_sigma, tuple) or len(self.m_sigma) != 3:
            raise TypeError('The sigma argument should contain a tuple with the standard '
                            'deviation of the separation (arcsec), position angle (deg), '
                            'and contrast (mag) that is used to sample the starting position '
                            'of the walkers.')

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
            self.m_mask[0] /= pixscale

        if self.m_mask[1] is not None:
            self.m_mask[1] /= pixscale

        # create the mask and get the unmasked image indices
        mask = create_mask(im_shape[-2:], self.m_mask)
        indices = np.where(mask.reshape(-1) != 0.)[0]

        self.aperture_dict(images)

        initial = np.zeros((self.m_nwalkers, ndim))

        initial[:, 0] = self.m_param[0] + np.random.normal(0, self.m_sigma[0], self.m_nwalkers)
        initial[:, 1] = self.m_param[1] + np.random.normal(0, self.m_sigma[1], self.m_nwalkers)
        initial[:, 2] = self.m_param[2] + np.random.normal(0, self.m_sigma[2], self.m_nwalkers)

        if self.m_variance == 'gaussian':
            student_t = self.gaussian_noise(images*mask, psf, parang, self.m_aperture)
            variance = (self.m_variance, student_t)

        else:
            variance = (self.m_variance, None)

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
                                               self.m_aperture,
                                               indices,
                                               self.m_prior,
                                               variance,
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

    @typechecked
    def __init__(self,
                 radius: float = 0.1,
                 position: Tuple[float, float] = None,
                 name_in: str = 'aperture_photometry',
                 image_in_tag: str = 'im_arr',
                 phot_out_tag: str = 'photometry') -> None:
        """
        Parameters
        ----------
        radius : float
            Radius (arcsec) of the circular aperture.
        position : tuple(float, float), None
            Center position (pix) of the aperture, (x, y), with subpixel precision. The center of
            the image will be used if set to None. Python indexing starts at zero so the bottom
            left corner of the image has coordinates (-0.5, -0.5).
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        phot_out_tag : str
            Tag of the database entry with the photometry values that are written as output.

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
            return aperture_photometry(image, aperture, method='exact')['aperture_sum']

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        self.m_radius /= pixscale

        if self.m_position is None:
            self.m_position = center_subpixel(self.m_image_in_port[0, ])

        # Position in CircularAperture is defined as (x, y)
        aperture = CircularAperture(self.m_position, self.m_radius)

        self.apply_function_to_images(_photometry,
                                      self.m_image_in_port,
                                      self.m_phot_out_port,
                                      'Running AperturePhotometryModule',
                                      func_args=(aperture, ))

        history = f'radius [arcsec] = {self.m_radius*pixscale:.3f}'
        self.m_phot_out_port.copy_attributes(self.m_image_in_port)
        self.m_phot_out_port.add_history('AperturePhotometryModule', history)
        self.m_phot_out_port.close_port()
