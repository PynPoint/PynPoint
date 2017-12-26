"""
Modules for photometry and astrometry.
"""

import math
import warnings
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import shift
from scipy.optimize import minimize
from astropy.nddata import Cutout2D
from numdifftools import Hessian

from PynPoint.core.Processing import ProcessingModule
from PynPoint.processing_modules import PSFSubtractionModule


class FakePlanetModule(ProcessingModule):
    """
    Module to inject a positive or negative fake companion into a stack of images.
    """

    def __init__(self,
                 position,
                 magnitude,
                 scaling=1.,
                 name_in="fake_planet",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 image_out_tag="im_fake"):
        """
        Constructor of FakePlanetModule.

        :param position: Angular separation (arcsec) and position angle (deg) of the fake planet.
                         Angle is measured in counterclockwise direction with respect to the
                         upward direction (i.e., East of North).
        :type position: tuple
        :param magnitude: Magnitude of the fake planet with respect to the star.
        :type magnitude: float
        :param scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                        neutral density filter). A negative value will inject a negative planet
                        signal.
        :type scaling: float
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

        :return: None
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
        self.m_scaling = scaling

    def run(self):
        """
        Run method of the module. Shifts the reference PSF to the location of the fake planet
        with an additional correction for the parallactic angle and writes the stack with images
        with the injected planet signal.

        :return: None
        """

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
            n_stack_im = int(float(n_image)/float(memory))
            im_size = (self.m_image_in_port.get_shape()[1],
                       self.m_image_in_port.get_shape()[2])

        else:
            raise ValueError("The image_in_tag should contain a cube of images.")

        if ndim_psf == 2:
            n_psf = 1
            psf_size = (self.m_image_in_port.get_shape()[0],
                        self.m_image_in_port.get_shape()[1])

        elif ndim_psf == 3:
            n_psf = self.m_psf_in_port.get_shape()[0]
            psf_size = (self.m_image_in_port.get_shape()[1],
                        self.m_image_in_port.get_shape()[2])

        if psf_size != im_size:
            raise ValueError("The images in image_in_tag "+str(im_size)+" should have the same "
                             "dimensions as the image(s) in psf_in_tag "+str(psf_size)+".")

        n_stack_psf = int(float(n_psf)/float(memory))

        im_stacks = np.zeros(1, dtype=np.int)
        for i in range(n_stack_im):
            im_stacks = np.append(im_stacks, im_stacks[i]+memory)
        if n_stack_im*memory != n_image:
            im_stacks = np.append(im_stacks, n_image)

        if ndim_psf == 2:
            psf = self.m_psf_in_port.get_all()

        if ndim_psf == 3 and n_image != n_psf:
            warnings.warn("The number of images in psf_in_tag does not match with image_in_tag. "
                          "Calculating the mean of psf_in_tag...")

            psf = np.zeros((self.m_image_in_port.get_shape()[1],
                            self.m_image_in_port.get_shape()[2]))

            for i in range(n_stack_psf):
                psf += np.sum(self.m_psf_in_port[i*memory:(i+1)*memory], axis=0)

            if n_stack_psf*memory != n_psf:
                psf += np.sum(self.m_psf_in_port[n_stack_psf*memory:n_psf], axis=0)

            psf /= float(n_psf)

        for j, _ in enumerate(im_stacks[:-1]):
            image = self.m_psf_in_port[im_stacks[j]:im_stacks[j+1]]

            for i in range(image.shape[0]):
                x_shift = radial*math.cos(theta-parang[j*memory+i])
                y_shift = radial*math.sin(theta-parang[j*memory+i])

                if ndim_psf == 2 or (ndim_psf == 3 and n_psf != n_image):

                    psf_tmp = np.copy(psf)

                elif ndim_psf == 3 and n_psf == n_image:
                    psf_tmp = self.m_psf_in_port[j*memory+i]

                psf_tmp = shift(psf_tmp,
                                (y_shift, x_shift),
                                order=5,
                                mode='reflect')

                image[i,] += self.m_scaling*flux_ratio*psf_tmp

            if j == 0:
                self.m_image_out_port.set_all(image)
            else:
                self.m_image_out_port.append(image)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Fake planet",
                                                      "(sep, angle, mag) = " + "(" + \
                                                      "{0:.2f}".format(self.m_position[0])+", "+ \
                                                      "{0:.2f}".format(self.m_position[1])+", "+ \
                                                      "{0:.2f}".format(self.m_magnitude)+")")

        self.m_image_out_port.close_port()


class HessianMatrixModule(ProcessingModule):
    """
    Module to measure the flux and position of a planet signal by injecting negative fake planets
    and minimizing the curvature at the planet location.
    """

    def __init__(self,
                 position,
                 magnitude,
                 scaling=-1.,
                 name_in="hessian_matrix",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 image_out_tag="im_hessian",
                 subpix=5.,
                 tolerance=0.1,
                 sigma=1.,
                 crop_size=10,
                 num_pos=10,
                 pca_number=20,
                 mask=0.1):
        """
        Constructor of HessianMatrixModule.

        :param position: Approximate position (x, y) of the planet (pix).
        :type position: tuple
        :param magnitude: Approximate magnitude of the planet relative to the star.
        :type magnitude: float
        :param scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                        neutral density filter). Should be negative in order to inject negative
                        fake planets.
        :type scaling: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that are read as input.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry with the reference PSF that is used as fake
                           planet. Can be either a single image (2D) or a cube (3D) with the
                           dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Contains the
                              results from the PSF subtraction during the minimization of the
                              image curvature. The last image contains the image with the
                              best-fit curvature.
        :type image_out_tag: str
        :param subpix: Subpixel precision (1/subpix) for the optimization of the astrometry
                       (and photometry?).
        :type subpix: int
        :param tolerance: Tolerance for termination of the minimization.
        :type tolerance: float
        :param sigma: Standard deviation for the Gaussian kernel (pix).
        :type sigma: float
        :param crop_size: Size of the cropped image on which the Hessian matrix is calculated.
        :type crop_size: int
        :param num_pos: Sampling of the cropped image. The curvature is sampled at
                        *num_pos x num_pos* equally spaced positions.
        :type num_pos: int
        :param pca_number: Number of principle components used for the PSF subtraction
        :type pca_number: int
        :param mask: Mask radius (arcsec) for the PSF subtraction.
        :type mask: float

        :return: None
        """

        super(HessianMatrixModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_scaling = scaling
        self.m_subpix = subpix
        self.m_tolerance = tolerance
        self.m_sigma = sigma
        self.m_crop_size = crop_size
        self.m_num_pos = num_pos
        self.m_pca_number = pca_number
        self.m_mask = mask
        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag
        self.m_image_out_tag = image_out_tag

    def run(self):
        """
        Run method of the module. The position and flux of a planet are measured by injecting
        negative fake companions and applying a Nelder-Mead method minimization of the curvature
        of the image at the planet location. The curvature is calculated as the sum of the
        absolute values of the determinant of the Hessian matrix.

        :return: None
        """

        def _hessian(arg):
            pos_x = arg[0]
            pos_y = arg[1]
            mag = arg[2]

            sep = math.sqrt((pos_y-center[0])**2+(pos_x-center[1])**2)*pixscale
            ang = math.atan2(pos_y-center[0], pos_x-center[1])*180./math.pi - 90.

            if ang < 360.:
                ang += 360.

            print "Trying separation [arcsec] = " + "{0:.3f}".format(sep) + \
                  ", PA [deg] = " + "{0:.2f}".format(ang) + \
                  ", contrast [mag] = " + "{0:.2f}".format(mag) + " ..."

            fake_planet = FakePlanetModule(position=(sep, ang),
                                           magnitude=mag,
                                           scaling=self.m_scaling,
                                           name_in="fake_planet",
                                           image_in_tag=self.m_image_in_tag,
                                           psf_in_tag=self.m_psf_in_tag,
                                           image_out_tag="hessian_fake")

            fake_planet.connect_database(self._m_data_base)
            fake_planet.run()

            psf_sub = PSFSubtractionModule(name_in="pca_hessian",
                                           pca_number=self.m_pca_number,
                                           images_in_tag="hessian_fake",
                                           reference_in_tag="hessian_fake",
                                           res_arr_out_tag="hessian_res_arr_out",
                                           res_arr_rot_out_tag="hessian_res_arr_rot_out",
                                           res_mean_tag="hessian_res_mean",
                                           res_median_tag="hessian_res_median",
                                           res_var_tag="hessian_res_var",
                                           res_rot_mean_clip_tag="hessian_res_rot_mean_clip",
                                           basis_out_tag="hessian_basis_out",
                                           image_ave_tag="hessian_image_ave",
                                           psf_model_tag="hessian_psf_model",
                                           ref_prep_tag="hessian_ref_prep",
                                           prep_tag="hessian_prep",
                                           extra_rot=0.,
                                           cent_size=self.m_mask,
                                           cent_mask_tag="hessian_cent_mask",
                                           verbose=False)

            psf_sub.connect_database(self._m_data_base)
            psf_sub.run()

            res_input_port = self.add_input_port("hessian_res_mean")
            im_res = res_input_port.get_all()
            im_smooth = gaussian_filter(im_res, sigma=self.m_sigma)

            im_crop = Cutout2D(data=im_smooth,
                               position=(pos_x, pos_y),
                               size=(self.m_crop_size, self.m_crop_size)).data

            self.m_image_out_port.append(im_res, data_dim=3)

            xx = np.arange(self.m_crop_size)
            yy = np.arange(self.m_crop_size)
            im_interp = interp2d(xx, yy, im_crop, kind='cubic')

            hessian = Hessian(lambda z: im_interp(z[0], z[1]))

            det = 0.
            for i in np.linspace(0, self.m_crop_size, self.m_num_pos):
                for j in np.linspace(0, self.m_crop_size, self.m_num_pos):
                    det += np.abs(np.linalg.det(hessian((i, j))))

            return det

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

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

        center = (psf_size[0]/2., psf_size[0]/2.)

        self.m_mask /= (pixscale*im_size[1]/2.)

        result = minimize(fun=_hessian,
                          x0=[self.m_position[0], self.m_position[1], self.m_magnitude],
                          method="Nelder-Mead",
                          tol=self.m_tolerance)

        position = (np.round(result.x[0]*self.m_subpix)/self.m_subpix, \
                    np.round(result.x[1]*self.m_subpix)/self.m_subpix)

        mag = result.x[2]
        sep = math.sqrt((position[0]-center[0])**2+(position[1]-center[1])**2)*pixscale
        ang = math.atan2(position[1]-center[1], position[0]-center[0])*180./math.pi - 90.

        if ang < 0.:
            ang += 360.

        fake_planet = FakePlanetModule(position=(sep, ang),
                                       magnitude=mag,
                                       scaling=self.m_scaling,
                                       name_in="fake_planet",
                                       image_in_tag=self.m_image_in_tag,
                                       psf_in_tag=self.m_psf_in_tag,
                                       image_out_tag="hessian_fake")

        fake_planet.connect_database(self._m_data_base)
        fake_planet.run()

        psf_sub = PSFSubtractionModule(name_in="pca_hessian",
                                       pca_number=self.m_pca_number,
                                       images_in_tag="hessian_fake",
                                       reference_in_tag="hessian_fake",
                                       res_arr_out_tag="hessian_res_arr_out",
                                       res_arr_rot_out_tag="hessian_res_arr_rot_out",
                                       res_mean_tag="hessian_res_mean",
                                       res_median_tag="hessian_res_median",
                                       res_var_tag="hessian_res_var",
                                       res_rot_mean_clip_tag="hessian_res_rot_mean_clip",
                                       basis_out_tag="hessian_basis_out",
                                       image_ave_tag="hessian_image_ave",
                                       psf_model_tag="hessian_psf_model",
                                       ref_prep_tag="hessian_ref_prep",
                                       prep_tag="hessian_prep",
                                       extra_rot=0.,
                                       cent_size=self.m_mask,
                                       cent_mask_tag="hessian_cent_mask",
                                       verbose=False)

        psf_sub.connect_database(self._m_data_base)
        psf_sub.run()

        res_input_port = self.add_input_port("hessian_res_mean")
        im_res = res_input_port.get_all()
        self.m_image_out_port.append(im_res, data_dim=3)

        pos_err = np.sqrt((1./self.m_subpix)**2 + self.m_tolerance**2)

        sep_err = (np.sqrt((pixscale**2*1./sep*(position[0]-center[0])*pos_err)**2+ \
                  (pixscale**2*1./sep*(position[1]-center[1])*pos_err)**2))

        ang_err = np.sqrt((pos_err*((position[0]-center[0]) / \
                  ((position[0]-center[0])**2+(position[1]-center[1])**2)))**2 + \
                  (pos_err*((position[1]-center[1])/((position[0]-center[0])**2 + \
                  (position[1]-center[1])**2)))**2) * 180./math.pi

        self.m_image_out_port.add_history_information("Flux and position",
                                                      "Hessian matrix")

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()

        # TODO output should go to database

        print "Position = "+str(position)
        print "Separation [arcsec] = "+"{0:.3f}".format(sep)+" +/- "+"{0:.3f}".format(sep_err)
        print "Position angle [deg] = "+"{0:.2f}".format(ang)+" +/- "+"{0:.2f}".format(ang_err)
        print "Contrast [mag] = "+"{0:.2f}".format(mag)

        im_res_cut_f_before = Cutout2D(data=im_res,
                                       position=position,
                                       size=(self.m_crop_size, self.m_crop_size)).data

        vmax = np.max(im_res_cut_f_before)
        vmin = np.min(im_res_cut_f_before)

        im_res_smooth_f = gaussian_filter(im_res, sigma=self.m_sigma)

        im_res_cut_f_after = Cutout2D(data=im_res_smooth_f,
                                      position=position,
                                      size=(self.m_crop_size, self.m_crop_size)).data

        surface, (before, after) = plt.subplots(1, 2, figsize=(12, 6))
        before.imshow(im_res_cut_f_before, vmax=vmax, vmin=vmin, cmap='autumn')
        for i in np.arange(self.m_crop_size):
            for j in np.arange(self.m_crop_size):
                before.text(i-0.4, j, '%s'%float('%.2g'%(im_res_cut_f_before[j, i]*1e7)))
        before.set_title('Before Convolution')

        after.imshow(im_res_cut_f_after, vmax=vmax, vmin=vmin, cmap='autumn')
        for i in np.arange(self.m_crop_size):
            for j in np.arange(self.m_crop_size):
                after.text(i-0.4, j, '%s'%float('%.2g'%(im_res_cut_f_after[j, i]*1e7)))
        after.set_title('After Convolution')

        surface.savefig('pixelmap.png')
