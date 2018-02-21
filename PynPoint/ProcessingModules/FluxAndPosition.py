"""
Modules for photometric and astrometric measurements of the planet.
"""

import math
import sys
import warnings

import numpy as np

from scipy.ndimage import shift
from scipy.optimize import minimize
from scipy.stats import t
from scipy.ndimage import fourier_shift
from skimage.feature import hessian_matrix
from astropy.nddata import Cutout2D
from photutils import aperture_photometry, CircularAperture

from PynPoint.Util.Progress import progress
from PynPoint.Core.Processing import ProcessingModule
from PynPoint.ProcessingModules.PSFpreparation import PSFpreparationModule
from PynPoint.ProcessingModules.PSFSubtractionPCA import PSFSubtractionModule, FastPCAModule


class FakePlanetModule(ProcessingModule):
    """
    Module to inject a positive or negative fake companion into a stack of images.
    """

    def __init__(self,
                 position,
                 magnitude,
                 psf_scaling=1.,
                 interpolation="fft",
                 name_in="fake_planet",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 image_out_tag="im_fake",
                 **kwargs):
        """
        Constructor of FakePlanetModule.

        :param position: Angular separation (arcsec) and position angle (deg) of the fake planet.
                         Angle is measured in counterclockwise direction with respect to the
                         upward direction (i.e., East of North).
        :type position: tuple
        :param magnitude: Magnitude of the fake planet with respect to the star.
        :type magnitude: float
        :param psf_scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                            neutral density filter). A negative value will inject a negative
                            planet signal.
        :type psf_scaling: float
        :param interpolation: Type of interpolation that is used for shifting the images (fft,
                              spline, or bilinear).
        :type interpolation: str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that are read as input.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry that contains the reference PSF that is used
                           as fake planet. Can be either a single image (2D) or a cube (3D) with
                           the dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param image_out_tag: Tag of the database entry with images that are written as output.
        :type image_out_tag: str
        :param \**kwargs:
            See below.

        :Keyword arguments:
             * **verbose** (*bool*) -- Print progress.

        :return: None
        """

        if "verbose" in kwargs:
            self.m_verbose = kwargs["verbose"]
        else:
            self.m_verbose = True

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

    def run(self):
        """
        Run method of the module. Shifts the reference PSF to the location of the fake planet
        with an additional correction for the parallactic angle and writes the stack with images
        with the injected planet signal.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        parang = self.m_image_in_port.get_attribute("NEW_PARA")
        parang *= math.pi/180.

        radial = self.m_position[0]/pixscale
        theta = self.m_position[1]*math.pi/180. + math.pi/2.

        flux_ratio = 10.**(-self.m_magnitude/2.5)

        ndim_image = np.size(self.m_image_in_port.get_shape())
        ndim_psf = np.size(self.m_psf_in_port.get_shape())

        if ndim_image == 3:
            n_image = self.m_image_in_port.get_shape()[0]

            if memory == -1 or memory >= n_image:
                n_stack_im = 1
            else:
                n_stack_im = int(float(n_image)/float(memory))

            im_size = (self.m_image_in_port.get_shape()[1],
                       self.m_image_in_port.get_shape()[2])

        else:
            raise ValueError("The image_in_tag should contain a cube of images.")

        if ndim_psf == 2:
            n_psf = 1
            psf_size = (self.m_psf_in_port.get_shape()[0],
                        self.m_psf_in_port.get_shape()[1])

        elif ndim_psf == 3:
            n_psf = self.m_psf_in_port.get_shape()[0]
            psf_size = (self.m_psf_in_port.get_shape()[1],
                        self.m_psf_in_port.get_shape()[2])

        if psf_size != im_size:
            raise ValueError("The images in %s should have the same dimensions as the images "
                             "images in %s.") % (self.m_image_in_port.tag, self.m_psf_in_port.tag)

        if memory == -1 or memory >= n_psf:
            n_stack_psf = 1
        else:
            n_stack_psf = int(float(n_psf)/float(memory))

        im_stacks = np.zeros(1, dtype=np.int)
        if memory == -1 or memory >= n_image:
            im_stacks = np.append(im_stacks, im_stacks[0]+n_image)

        else:
            for i in range(n_stack_im):
                im_stacks = np.append(im_stacks, im_stacks[i]+memory)
            if n_stack_im*memory != n_image:
                im_stacks = np.append(im_stacks, n_image)

        if ndim_psf == 3 and n_psf == 1:
            psf = np.squeeze(self.m_psf_in_port.get_all(), axis=0)
            ndim_psf = psf.ndim

        elif ndim_psf == 2:
            psf = self.m_psf_in_port.get_all()

        elif ndim_psf == 3 and n_image != n_psf:
            warnings.warn("The number of images in %s do not match with %s. Using the mean "
                          "instead." % (self.m_psf_in_port.tag, self.m_image_in_port.tag))

            psf = np.zeros((self.m_image_in_port.get_shape()[1],
                            self.m_image_in_port.get_shape()[2]))

            if memory == -1 or memory >= n_psf:
                psf = np.mean(self.m_psf_in_port.get_all(), axis=0)

            else:
                for i in range(n_stack_psf):
                    psf += np.sum(self.m_psf_in_port[i*memory:(i+1)*memory], axis=0)

                if n_stack_psf*memory != n_psf:
                    psf += np.sum(self.m_psf_in_port[n_stack_psf*memory:n_psf], axis=0)

                psf /= float(n_psf)

            ndim_psf == psf.ndim

        for j, _ in enumerate(im_stacks[:-1]):
            if self.m_verbose:
                progress(j, len(im_stacks[:-1]), "Running FakePlanetModule...")

            image = self.m_image_in_port[im_stacks[j]:im_stacks[j+1]]

            for i in range(image.shape[0]):
                if memory == -1 or memory >= n_image:
                    x_shift = radial*math.cos(theta-parang[i])
                    y_shift = radial*math.sin(theta-parang[i])

                else:
                    x_shift = radial*math.cos(theta-parang[j*memory+i])
                    y_shift = radial*math.sin(theta-parang[j*memory+i])

                if ndim_psf == 2:
                    psf_tmp = np.copy(psf)

                elif ndim_psf == 3:
                    if memory == -1 or memory >= n_psf:
                        psf_tmp = self.m_psf_in_port[i]

                    else:
                        psf_tmp = self.m_psf_in_port[j*memory+i]

                if self.m_interpolation == "fft":
                    psf_fft = fourier_shift(np.fft.fftn(psf_tmp), (y_shift, x_shift))
                    psf_tmp = np.fft.ifftn(psf_fft).real

                elif self.m_interpolation == "spline":
                    psf_tmp = shift(psf_tmp, (y_shift, x_shift), order=5, mode='reflect')

                elif self.m_interpolation == "bilinear":
                    psf_tmp = shift(psf_tmp, (y_shift, x_shift), order=1, mode='reflect')

                else:
                    raise ValueError("Interpolation should be fft, spline, or bilinear.")

                image[i,] += self.m_psf_scaling*flux_ratio*psf_tmp

            self.m_image_out_port.append(image)

        if self.m_verbose:
            sys.stdout.write("Running FakePlanetModule... [DONE]\n")
            sys.stdout.flush()

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Fake planet",
                                                      "(sep, angle, mag) = " + "(" + \
                                                      "{0:.2f}".format(self.m_position[0])+", "+ \
                                                      "{0:.2f}".format(self.m_position[1])+", "+ \
                                                      "{0:.2f}".format(self.m_magnitude)+")")

        self.m_image_out_port.close_database()


class SimplexMinimizationModule(ProcessingModule):
    """
    Module to measure the flux and position of a planet by injecting negative fake planets and
    minimizing a function of merit.
    """

    def __init__(self,
                 position,
                 magnitude,
                 psf_scaling=-1.,
                 name_in="simplex",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 res_out_tag="simplex_res",
                 flux_position_tag="flux_position",
                 merit="hessian",
                 aperture=0.1,
                 sigma=0.027,
                 tolerance=0.1,
                 pca_module='FastPCAModule',
                 pca_number=20,
                 mask=0.,
                 extra_rot=0.):
        """
        Constructor of SimplexMinimizationModule.

        :param position: Approximate position (x, y) of the planet (pix). This is also the location
                         where the function of merit is calculated with an aperture of radius
                         *aperture*.
        :type position: tuple
        :param magnitude: Approximate magnitude of the planet relative to the star.
        :type magnitude: float
        :param psf_scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                            neutral density filter). Should be negative in order to inject
                            negative fake planets.
        :type psf_scaling: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that are read as input.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry with the reference PSF that is used as fake
                           planet. Can be either a single image (2D) or a cube (3D) with the
                           dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param res_out_tag: Tag of the database entry with the image residuals that are written
                            as output. Contains the results from the PSF subtraction during the
                            minimization of the function of merit. The last image is the image
                            with the best-fit residuals.
        :type res_out_tag: str
        :param flux_position_tag: Tag of the database entry with flux and position results that are
                                  written as output. Each step of the minimization saves the
                                  x position (pix), y position (pix), separation (arcsec),
                                  angle (deg), contrast (mag), and the function of merit. The last
                                  row of values contain the best-fit results.
        :type flux_position_tag: str
        :param merit: Function of merit for the minimization. Can be either *hessian*, to minimize
                      the sum of the absolute values of the determinant of the Hessian matrix, or
                      *sum*, to minimize the sum of the absolute pixel values (Wertz et al. 2017).
        :type merit: str
        :param aperture: Aperture radius (arcsec) of the selected region, centered on the negative
                         fake companion, used for the minimization.
        :type aperture: float
        :param sigma: Standard deviation (arcsec) of the Gaussian kernel which is used to smooth
                      the images before the function of merit is calculated (in order to reduce
                      small pixel-to-pixel variations). Highest astrometric and photometric
                      precision is achieved when sigma is optimized.
        :type sigma: float
        :param tolerance: Absolute error on the input parameters, position (pix) and
                          contrast (mag), that is used as acceptance level for convergence. Note
                          that only a single value can be specified which is used for both the
                          position and flux so tolerance=0.1 will give a precision of 0.1 mag
                          and 0.1 pix. The tolerance on the output (i.e., function of merit)
                          is set to np.inf so the condition is always met.
        :type tolerance: float
        :param pca_module: Name of the processing module for the PSF subtraction
                           (PSFSubtractionModule or FastPCAModule).
        :type pca_module: str
        :param pca_number: Number of principle components used for the PSF subtraction.
        :type pca_number: int
        :param mask: Mask radius (arcsec) for the PSF subtraction.
        :type mask: float
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(SimplexMinimizationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)
        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_flux_position_port = self.add_output_port(flux_position_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_pca_module = pca_module
        self.m_pca_number = pca_number
        self.m_mask = mask
        self.m_extra_rot = extra_rot

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag

    def run(self):
        """
        Run method of the module. The position and flux of a planet are measured by injecting
        negative fake companions and applying a simplex method (Nelder-Mead) for minimization
        of a function of merit at the planet location. The default function of merit is the
        image curvature which is calculated as the sum of the absolute values of the
        determinant of the Hessian matrix.

        :return: None
        """

        def _rotate(center, position, angle):
            pos_x = (position[0]-center[0])*math.cos(np.radians(angle)) - \
                    (position[1]-center[1])*math.sin(np.radians(angle))

            pos_y = (position[0]-center[0])*math.sin(np.radians(angle)) + \
                    (position[1]-center[1])*math.cos(np.radians(angle))

            return (center[0]+pos_x, center[1]+pos_y)

        def _objective(arg):
            sys.stdout.write('.')
            sys.stdout.flush()

            pos_x = arg[0]
            pos_y = arg[1]
            mag = arg[2]

            sep = math.sqrt((pos_y-center[0])**2+(pos_x-center[1])**2)*pixscale
            ang = math.atan2(pos_y-center[0], pos_x-center[1])*180./math.pi - 90.

            fake_planet = FakePlanetModule(position=(sep, ang),
                                           magnitude=mag,
                                           psf_scaling=self.m_psf_scaling,
                                           interpolation="fft",
                                           name_in="fake_planet",
                                           image_in_tag=self.m_image_in_tag,
                                           psf_in_tag=self.m_psf_in_tag,
                                           image_out_tag="simplex_fake",
                                           verbose=False)

            fake_planet.connect_database(self._m_data_base)
            fake_planet.run()

            if self.m_pca_module == "PSFSubtractionModule":

                if self.m_mask > 0.:

                    psf_sub = PSFSubtractionModule(name_in="pca_simplex",
                                                   pca_number=self.m_pca_number,
                                                   svd="arpack",
                                                   images_in_tag="simplex_fake",
                                                   reference_in_tag="simplex_fake",
                                                   res_arr_out_tag="simplex_res_arr_out",
                                                   res_arr_rot_out_tag="simplex_res_arr_rot_out",
                                                   res_mean_tag="simplex_res_mean",
                                                   res_median_tag="simplex_res_median",
                                                   res_var_tag="simplex_res_var",
                                                   res_rot_mean_clip_tag="simplex_res_rot_mean_clip",
                                                   basis_out_tag="simplex_basis_out",
                                                   image_ave_tag="simplex_image_ave",
                                                   psf_model_tag="simplex_psf_model",
                                                   ref_prep_tag="simplex_ref_prep",
                                                   prep_tag="simplex_prep",
                                                   cent_mask_tag="simplex_cent_mask",
                                                   extra_rot=self.m_extra_rot,
                                                   cent_remove=True,
                                                   cent_size=self.m_mask,
                                                   verbose=False)

                else:

                    psf_sub = PSFSubtractionModule(name_in="pca_simplex",
                                                   pca_number=self.m_pca_number,
                                                   svd="arpack",
                                                   images_in_tag="simplex_fake",
                                                   reference_in_tag="simplex_fake",
                                                   res_arr_out_tag="simplex_res_arr_out",
                                                   res_arr_rot_out_tag="simplex_res_arr_rot_out",
                                                   res_mean_tag="simplex_res_mean",
                                                   res_median_tag="simplex_res_median",
                                                   res_var_tag="simplex_res_var",
                                                   res_rot_mean_clip_tag="simplex_res_rot_mean_clip",
                                                   basis_out_tag="simplex_basis_out",
                                                   image_ave_tag="simplex_image_ave",
                                                   psf_model_tag="simplex_psf_model",
                                                   ref_prep_tag="simplex_ref_prep",
                                                   prep_tag="simplex_prep",
                                                   cent_mask_tag="simplex_cent_mask",
                                                   extra_rot=self.m_extra_rot,
                                                   cent_remove=False,
                                                   verbose=False)

            elif self.m_pca_module == "FastPCAModule":

                cent_remove = bool(self.m_mask > 0.)

                prep = PSFpreparationModule(name_in="prep",
                                            image_in_tag="simplex_fake",
                                            image_out_tag="simplex_prep",
                                            image_mask_out_tag=None,
                                            mask_out_tag=None,
                                            norm=True,
                                            cent_remove=cent_remove,
                                            cent_size=self.m_mask,
                                            edge_size=1.,
                                            verbose=False)

                prep.connect_database(self._m_data_base)
                prep.run()

                psf_sub = FastPCAModule(name_in="pca_simplex",
                                        pca_numbers=self.m_pca_number,
                                        images_in_tag="simplex_prep",
                                        reference_in_tag="simplex_prep",
                                        res_mean_tag="simplex_res_mean",
                                        res_median_tag=None,
                                        res_arr_out_tag=None,
                                        res_rot_mean_clip_tag=None,
                                        extra_rot=self.m_extra_rot,
                                        verbose=False)

            else:

                raise ValueError("The pca_module should be either PSFSubtractionModule or "
                                 "FastPCAModule.")

            psf_sub.connect_database(self._m_data_base)
            psf_sub.run()

            res_input_port = self.add_input_port("simplex_res_mean")
            im_res = res_input_port.get_all()

            if len(im_res.shape) == 3:
                if im_res.shape[0] == 1:
                    im_res = np.squeeze(im_res, axis=0)
                else:
                    raise ValueError("Multiple residual images found, expecting only one.")

            self.m_res_out_port.append(im_res, data_dim=3)

            im_crop = Cutout2D(data=im_res,
                               position=self.m_position,
                               size=2*self.m_aperture).data

            npix = im_crop.shape[0]

            if npix%2 == 0:
                x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
            elif npix%2 == 1:
                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
            rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

            if self.m_merit == "hessian":

                hessian_rr, hessian_rc, hessian_cc = hessian_matrix(im_crop,
                                                                    sigma=self.m_sigma,
                                                                    mode='constant',
                                                                    cval=0.,
                                                                    order='rc')

                hes_det = (hessian_rr*hessian_cc) - (hessian_rc*hessian_rc)

                hes_det[rr_grid > self.m_aperture] = 0.

                merit = np.sum(np.abs(hes_det))

            elif self.m_merit == "sum":

                im_crop[rr_grid > self.m_aperture] = 0.

                merit = np.sum(np.abs(im_crop))
                # merit = np.sum( im_crop**2 / np.std(im_crop[rr_grid < self.m_radius])**2 )

            else:
                raise ValueError("Function of merit should be set to hessian or sum.")

            position = _rotate(center, (pos_x, pos_y), -self.m_extra_rot)

            res = np.asarray((position[0],
                              position[1],
                              sep,
                              (ang-self.m_extra_rot)%360.,
                              mag,
                              merit))

            self.m_flux_position_port.append(res, data_dim=2)

            return merit

        self.m_res_out_port.del_all_data()
        self.m_res_out_port.del_all_attributes()

        self.m_flux_position_port.del_all_data()
        self.m_flux_position_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_aperture /= pixscale
        self.m_aperture = int(math.ceil(self.m_aperture))

        self.m_sigma /= pixscale

        ndim_image = np.size(self.m_image_in_port.get_shape())
        ndim_psf = np.size(self.m_psf_in_port.get_shape())

        if ndim_image == 3:
            im_size = (self.m_image_in_port.get_shape()[1],
                       self.m_image_in_port.get_shape()[2])

        else:
            raise ValueError("The image_in_tag should contain a cube of images.")

        if ndim_psf == 2:
            psf_size = (self.m_image_in_port.get_shape()[0],
                        self.m_image_in_port.get_shape()[1])

        elif ndim_psf == 3:
            psf_size = (self.m_image_in_port.get_shape()[1],
                        self.m_image_in_port.get_shape()[2])

        center = (psf_size[0]/2., psf_size[1]/2.)

        self.m_mask /= (pixscale*im_size[1])

        sys.stdout.write("Running SimplexMinimizationModule")
        sys.stdout.flush()

        pos_init = _rotate(center, self.m_position, self.m_extra_rot)

        minimize(fun=_objective,
                 x0=[pos_init[0], pos_init[1], self.m_magnitude],
                 method="Nelder-Mead",
                 tol=None,
                 options={'xatol': self.m_tolerance, 'fatol': float("inf")})

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_res_out_port.add_history_information("Flux and position",
                                                    "Simplex minimization")

        self.m_flux_position_port.add_history_information("Flux and position",
                                                          "Simplex minimization")

        self.m_res_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_flux_position_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_res_out_port.close_database()


class FalsePositiveModule(ProcessingModule):
    """
    Module to calculate the signal-to-noise ratio (SNR) and false positive fraction (FPF) at a
    specified location in an image by using the Student's t-test (Mawet et al. 2014).
    """

    def __init__(self,
                 position,
                 aperture=0.1,
                 ignore=False,
                 name_in="snr",
                 image_in_tag="im_arr",
                 snr_out_tag="snr_fpf"):
        """
        Constructor of FalsePositiveModule.

        :param position: The x and y position (pix) where the SNR and FPF is calculated. Note that
                         the bottom left of the image is defined as (0, 0) so there is a -0.5
                         offset with respect to the DS9 coordinate system. Aperture photometry
                         corrects for the partial inclusion of pixels at the boundary.
        :type position: tuple
        :param aperture: Aperture radius (arcsec).
        :type aperture: float
        :param ignore: Ignore the two neighboring apertures that may contain self-subtraction from
                       the planet.
        :type ignore: bool
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that are read as input.
        :type image_in_tag: str
        :param snr_out_tag: Tag of the database entry that is written as output. The output format
                            is: (x position (pix), y position (pix), separation (arcsec), position
                            angle (deg), SNR, FPF). The position angle is measured in
                            counterclockwise direction with respect to the upward direction (i.e.,
                            East of North).
        :type snr_out_tag: str

        :return: None
        """

        super(FalsePositiveModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_snr_out_port = self.add_output_port(snr_out_tag)

        self.m_position = position
        self.m_aperture = aperture
        self.m_ignore = ignore

    def run(self):
        """
        Run method of the module. Calculates the SNR and FPF for a specified position in a post-
        processed image with the Student's t-test (Mawet et al. 2014). This approach accounts
        for small sample statistics.

        :return: None
        """

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        self.m_aperture /= pixscale

        image = self.m_image_in_port.get_all()

        if len(image.shape) == 3:
            warnings.warn("Using the first image of "+str(self.m_image_in_port.tag)+".")
            image = image[0, ]

        npix = image.shape[0]
        center = npix/2.

        if image.ndim > 2:
            raise ValueError("The image_in_tag should contain a 2D array.")

        sep = math.sqrt((center-self.m_position[0])**2.+(center-self.m_position[1])**2.)
        ang = (math.atan2(self.m_position[1]-center,
                          self.m_position[0]-center)*180./math.pi - 90.)%360.

        num_ap = int(math.pi*sep/self.m_aperture)
        ap_theta = np.linspace(0, 2.*math.pi, num_ap, endpoint=False)

        if self.m_ignore:
            num_ap -= 2
            ap_theta = np.delete(ap_theta, [1, np.size(ap_theta)-1])

        ap_phot = np.zeros(num_ap)
        for i, theta in enumerate(ap_theta):
            x_tmp = center + (self.m_position[0]-center)*math.cos(theta) - \
                             (self.m_position[1]-center)*math.sin(theta)
            y_tmp = center + (self.m_position[0]-center)*math.sin(theta) + \
                             (self.m_position[1]-center)*math.cos(theta)

            aperture = CircularAperture((x_tmp, y_tmp), self.m_aperture)
            phot_table = aperture_photometry(image, aperture, method='exact')
            ap_phot[i] = phot_table['aperture_sum']

        snr = (ap_phot[0] - np.mean(ap_phot[1:])) / \
              (np.std(ap_phot[1:]) * math.sqrt(1.+1./float(num_ap-1)))

        fpf = 1. - t.cdf(snr, num_ap-2)

        result = np.column_stack((self.m_position[0],
                                  self.m_position[1],
                                  sep*pixscale,
                                  ang,
                                  snr,
                                  fpf))

        self.m_snr_out_port.set_all(result)

        self.m_snr_out_port.add_history_information("Signal-to-noise ratio",
                                                    "Student's t-test")

        self.m_snr_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_snr_out_port.close_database()
