"""
Modules for photometry and astrometry.
"""

import math
import warnings

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import shift
from scipy.optimize import minimize
from astropy.nddata import Cutout2D
from astropy.table import Table

from PynPoint.core.Processing import ProcessingModule
from PynPoint.processing_modules import PSFSubtractionModule


class FakePlanetModule(ProcessingModule):
    """
    Module to
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
                         rightward horizontal direction.
        :type position: tuple
        :param magnitude: Magnitude of the fake planet with respect to the star.
        :type magnitude: float
        :param scaling: Additional scaling factor for the flux of the fake planet.
        :type scaling: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that is read as input.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry that contains the reference PSF that is used
                           as fake planet. Can be either a single image (2D) or a cube (3D) with
                           the dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param image_out_tag: Tag of the database entry with images that is written as output.
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
        Run method of the module.

        :return: None
        """        

        pixscale = self._m_config_port.get_attribute("PIXSCALE")

        parang = self.m_image_in_port.get_attribute("NEW_PARA")
        parang *= math.pi/180.
        
        # TODO use MEMORY
        im = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        center = (np.size(im, 0)/2., np.size(im, 1)/2.)

        radial = self.m_position[0]/pixscale
        theta = self.m_position[1]*math.pi/180.
        
        flux_ratio = 10.**(-self.m_magnitude/2.5)

        for i in range(np.size(im, 0)):
            x_shift = radial*math.cos(theta-parang[i])
            y_shift = radial*math.sin(theta-parang[i])
            
            psf_tmp = np.copy(psf)

            if psf_tmp.ndim == 2:
                psf_tmp = shift(psf_tmp, (y_shift, x_shift), order=5, mode='reflect')
                im[i,] += self.m_scaling*flux_ratio*psf_tmp

            elif psf_tmp.ndim == 3:
                if psf_tmp.shape[0] == im.shape[0]:
                        psf_tmp = shift(psf_tmp[i,],
                                        (y_shift, x_shift),
                                        order=5,
                                        mode='reflect')
                        im[i,] += self.m_scaling*flux_ratio*psf_tmp

                else:
                    psf_tmp = np.mean(psf_tmp, axis=0)
                    psf_tmp = shift(psf_tmp,
                                    (y_shift, x_shift),
                                    order=5,
                                    mode='reflect')
                    im[i,] += self.m_scaling*flux_ratio*psf_tmp

        self.m_image_out_port.set_all(im)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.add_history_information("Fake planet",
                                                      "(r [arcsec], theta [deg]) = "+str(self.m_position))

        self.m_image_out_port.close_port()


class HessianMatrixModule(ProcessingModule):
    """
    Module to
    """

    def __init__(self,
                 position,
                 magnitude,
                 scaling=-1.,
                 name_in="hessian_matrix",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 pca_number=20,
                 mask=0.1,
                 subpix=2.,
                 ROI_size=10,
                 tolerance=0.1,
                 num_pos=100):
        """
        Constructor of HessianMatrixModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag* unless *number_of_images_in_memory*
                              is set to *None*.
        :type image_out_tag: str
        :param num_lines: Number of top rows to delete from each frame.
        :type num_lines: int
        :param num_image_in_memory: Number of frames that are simultaneously loaded into the memory.
        :type num_image_in_memory: int

        :return: None
        """

        super(HessianMatrixModule, self).__init__(name_in)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_scaling = scaling
        self.m_pca_number = pca_number
        self.m_mask = mask
        self.m_subpix = subpix
        self.m_ROI_size = ROI_size
        self.m_image_in_tag = image_in_tag
        self.m_tolerance = tolerance
        self.m_num_pos = num_pos
        self.m_psf_in_tag = psf_in_tag
        self.m_image_in_tag = image_in_tag

    def run(self):
        """
        Run method of the module.

        :return: None
        """
    
        def hessian(arg):
            pos_x = arg[0]
            pos_y = arg[1]
            mag = arg[2]

            self.m_image_in_port = self.add_input_port(self.m_image_in_tag)
            if self.m_psf_in_tag == self.m_image_in_tag:
                self.m_psf_in_port = self.m_image_in_port
            else:
                self.m_psf_in_port = self.add_input_port(self.m_psf_in_tag)

            pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

            self.im = self.m_image_in_port.get_all()
            self.psf = self.m_psf_in_port.get_all()

            center = (np.size(self.im, 1)/2., np.size(self.im, 2)/2.)

            self.m_sep = math.sqrt((pos_y-center[0])**2+(pos_x-center[1])**2)*pixscale
            self.m_ang = math.atan2(pos_y-center[0], pos_x-center[1])*180./math.pi
            self.m_mag = mag

            fake_planet = FakePlanetModule(position=(self.m_sep, self.m_ang),
                                           magnitude=self.m_mag,
                                           scaling=self.m_scaling,
                                           name_in="fake_planet",
                                           image_in_tag=self.m_psf_in_tag,
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

            self.m_res_input_port = self.add_input_port("hessian_res_mean")

            im_res = self.m_res_input_port.get_all()
            #TODO make sigma argument
            im_res_smooth = gaussian_filter(im_res, 1)

            im_res_cut = Cutout2D(data=im_res_smooth,
                                  position=(pos_x, pos_y),
                                  size=(self.m_ROI_size, self.m_ROI_size)).data

            xx = np.arange(self.m_ROI_size)
            yy = np.arange(self.m_ROI_size)
            f = interp2d(xx, yy, im_res_cut, kind='cubic')

            sum_det = 0.
            for i in np.linspace(0, self.m_ROI_size, self.m_num_pos):
                for j in np.linspace(0, self.m_ROI_size, self.m_num_pos):
                    H = nd.Hessian(lambda z:f(z[0],z[1]))
                    det = np.linalg.det(H(np.array([i,j])))
                    sum_det += np.abs(det)

            self.m_image_in_port.close_port()

            return np.abs(sum_det)
        
        # if self.psf.ndim == 3 and self.psf.shape[0] != self.im.shape[0]:
        #     warnings.warn('The number of frames in psf_in_tag does not match with the number of '
        #                   'frames in image_in_tag. Using the mean of psf_in_tag as fake planet.')

        res = minimize(fun=hessian,
                       x0=[self.m_position[0], self.m_position[1], self.m_magnitude],
                       method="Nelder-Mead",
                       tol=self.m_tolerance)

        self.m_image_in_port = self.add_input_port(self.m_image_in_tag)
        if self.m_psf_in_tag == self.m_image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(self.m_psf_in_tag)

        pos_x = np.round(res.x[0]*self.m_subpix)/self.m_subpix
        pos_y = np.round(res.x[1]*self.m_subpix)/self.m_subpix
        mag = res.x[2]

        center = (np.size(self.im, 1)/2., np.size(self.im, 2)/2.)

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_sep = math.sqrt((pos_y-center[0])**2+(pos_x-center[1])**2)*pixscale
        self.m_ang = math.atan2(pos_y-center[0], pos_x-center[1])*180./math.pi
        self.m_mag = mag

        fake_planet = FakePlanetModule(position=(self.m_sep, self.m_ang),
                                       magnitude=self.m_mag,
                                       scaling=self.m_scaling,
                                       name_in="fake_planet",
                                       image_in_tag=self.m_psf_in_tag,
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

        self.m_image_in_port.close_port()

        self.m_pix2mas = self.m_image_in_port.get_attribute("PIXSCALE")*1000.

        im_init = self.m_image_in_port.get_all()

        image_size = np.size(im_init, 1)
        pos = np.array([res.x[0],res.x[1]])
        
        pos_err_x = np.sqrt((1. / self.m_subpix) ** 2 + (self.m_tolerance) ** 2)
        pos_err_y = np.sqrt((1. / self.m_subpix) ** 2 + (self.m_tolerance) ** 2)

        rad_dist = (self.m_pix2mas * np.sqrt((pos[0] - image_size / 2.) ** 2 + (pos[1] - image_size / 2.) ** 2))  ###This is in mas
        rad_dist_err = (np.sqrt((self.m_pix2mas ** 2 * 1. / rad_dist * (pos[0] - image_size / 2.) * pos_err_x) ** 2 +
                            (self.m_pix2mas ** 2 * 1. / rad_dist * (pos[1] - image_size / 2.) * pos_err_y) ** 2))

        # Calculate Position Angle and error:
        # See which quadrant the planet is in, and give accordingly the correct additional factor for the arctan calculation:
        if pos[0] > image_size / 2. and pos[1] > image_size / 2.:
            arctan_fac = 270.
            pos_angle = (90-np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.)))))+arctan_fac
        if pos[0] < image_size / 2. and pos[1] < image_size / 2.:
            arctan_fac = 90.
            pos_angle = (90-np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.)))))+arctan_fac
        if pos[0] > image_size / 2. and pos[1] < image_size / 2.:
            arctan_fac = 180.
            pos_angle =np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.))))+arctan_fac
        if pos[0] < image_size / 2. and pos[1] > image_size / 2.:
            arctan_fac = 0.
            pos_angle =np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.))))+arctan_fac

        pos_angle_err = (180./np.pi)*np.sqrt((pos_err_y*((pos[0]-image_size/2.)/((pos[0]-image_size/2.)**2+(pos[1]-image_size/2.)**2)))**2+(pos_err_x*((pos[1]-image_size/2.)/((pos[0]-image_size/2.)**2+(pos[1]-image_size/2.)**2)))**2)

        print '\n########################################'
        print 'POSITION = (' + str(np.round(res.x[0]*self.m_subpix)/self.m_subpix)+', '+ str(np.round(res.x[1]*self.m_subpix)/self.m_subpix) + ' )\n'
        print 'ASTROMETRY:\nRadial distance = ' + str(rad_dist*10**(-3)) + ' +- ' + str(rad_dist_err*10**(-3)) + ' [arcsec]\n' + 'P.A. = ' + str(pos_angle) + ' +- ' + str(pos_angle_err) + ' [deg]\n'
        print 'mag contrast = %: mag = ' + str(res.x[2])
        print '\n########################################'

        names = ['Pos x', 'Pos y', 'Rad dist', 'dist err', 'PA', 'PA err', 'mag contrast']

        results = Table([[pos_x], [pos_y], [rad_dist * (10 ** (-3))], [rad_dist_err * (10 ** (-3))], [pos_angle], [pos_angle_err], [mag]], names=names)

        results.write('hessian.txt', format='ascii.basic', delimiter='\t', overwrite=True)

        res_in_port_f = self.add_input_port("hessian_res_median")
        im_res_f = res_in_port_f.get_all()
        res_in_port_f.close_port()

        im_res_cut_f_before = Cutout2D(data=im_res_f,
                                       position=np.array([pos_x, pos_y]),
                                       size=(self.m_ROI_size, self.m_ROI_size)).data

        vmax = np.max(im_res_cut_f_before)
        vmin = np.min(im_res_cut_f_before)

        im_res_smooth_f = gaussian_filter(im_res_f,1)

        im_res_cut_f_after = Cutout2D(data=im_res_smooth_f,
                                      position=np.array([pos_x, pos_y]),
                                      size=(self.m_ROI_size, self.m_ROI_size)).data

        surface, (before, after) = plt.subplots(1,2, figsize=(12,6))
        before.imshow(im_res_cut_f_before, vmax=vmax, vmin=vmin,cmap='autumn')
        for i in np.arange(self.m_ROI_size):
            for j in np.arange(self.m_ROI_size):
                before.text(i-0.4,j, '%s'%float('%.2g'%(im_res_cut_f_before[j,i]*1e7)))
        before.set_title('Before Convolution')

        after.imshow(im_res_cut_f_after, vmax=vmax, vmin=vmin,cmap='autumn')
        for i in np.arange(self.m_ROI_size):
            for j in np.arange(self.m_ROI_size):
                after.text(i-0.4,j, '%s'%float('%.2g'%(im_res_cut_f_after[j,i]*1e7)))
        after.set_title('After Convolution')

        surface.savefig('pixelmap.png')
