"""
Pipeline modules for estimating detection limits.
"""

from __future__ import absolute_import

import sys
import os
import warnings
import multiprocessing as mp

import numpy as np

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.limits import contrast_limit
from pynpoint.util.module import progress
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals

class MassLimitsModule(ProcessingModule):
    """
    Module to calculate mass limits from an grid model from https://phoenix.ens-lyon.fr/Grids/ 
    and calculated ContrastLimits

    Parameters
    ----------
        name_in : str
            Unique name of the module instance.
        data_in_tag : str
            Tag of the database entry that contains the contrast curve data
        data_out_tag : str
            Tag of the database entry that contains the separation, azimuthally averaged mass
            limits, the one sigma boundaries of the mass limits.
        host_star_propertiers: dict
            Dictionary containing host star properties. Must have the following keys:
             - 'mag': apparent Magnitude, in the same band as the observation filter
             - 'dist': Distance in parsec
             - 'age': age of the system in the Myr
        observation_filter: str
            Name of the filter in which the observations were made. Must be the same as in the COND
            model data file
        model_file: str
            Relative path to the file containing the model data. Must be in the same format as the grids found
            on: https://phoenix.ens-lyon.fr/Grids/
    """
    def __init__(self,
                name_in="mass",
                data_in_tag="contrast_limits",
                data_out_tag="mass_limits",
                host_star_propertiers={'mag': 0, 'mag_app':True, 'dist': 10, 'age': 500, 'age_unit':'Myr'},
                observation_filter="L\'",
                model_file=""):
        
        """
        Constructor of MassLimitsModule
        """
        super(MassLimitsModule, self).__init__(name_in)

        # calculate the absolute magnitude of the star, given its apparent magnitude and its distance
        self.m_host_magnitude = host_star_propertiers['mag'] - 5 * np.log10(host_star_propertiers['dist'] / 10)

        
        self.m_distance = host_star_propertiers['dist']

        self.m_model_file = os.path.join(os.getcwd(), model_file)

        # add in and out ports
        self.m_data_in_port = self.add_input_port(data_in_tag)
        self.m_data_out_port = self.add_output_port(data_out_tag)

        self.m_ages, self.m_model_data, self.m_header = self._read_model()

        assert observation_filter in self.m_header, "The selected filter was not found in the list of available filters from the model"
        self.m_filter = observation_filter

        self.m_age = host_star_propertiers['age'] / 1000
        
    def _read_model(self):
        """
        Internal function. 
        Reads the data from the model file and structures it.
        Returns an array of available model ages and a list of model data for each age
        """

        # read in all the data, selecting out empty lines or '---' lines
        data = []
        with open(self.m_model_file) as file:
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
        model_data = []
        k=-1
        for _line in data:
            if '(Gyr)' in _line: # get time line
                ages += [float(_line[-1])]
                k += 1
                model_data += [[]]
            elif 'lg(g)' in _line: # get header line
                temp = ['M/Ms', 'Teff(K)'] + _line[1:]
                header = temp
            else: # save the data
                model_data[k] += [_line]
        for index, _ in enumerate(model_data):
            model_data[index] = np.array(model_data[index], dtype=float) 

        ages = np.array(ages, dtype = float)
        return ages, model_data, header

    def _interpolate_model(self, evaluate_age, evaluate_contrast):
        """
        Internal function to interpolate the grid based model data
        """
        # find the correct filter
        filter_index = np.argwhere([self.m_filter == j for j in self.m_header])[0] # simple argwhere gives empty list?!
        
        def _model_mass(age, contrast):
            """
            Internal function which returns the mass line for a given age and contrast
            """ 
            # find the correct grid
            age_grid_index = np.argwhere(self.m_ages == age)[0][0]
            grid = self.m_model_data[age_grid_index]

            # find the correct column (filter) 
            filter_column = grid[:, filter_index]
            # find the correct row (contrast)
            contrast_row = np.argwhere(contrast == filter_column)[0][0]
            
            # mass is found in the 0th column
            mass_column = 0

            # return the mass entry which corresponds to a given time, contrast and filter
            return grid[contrast_row, mass_column]
        
        points = np.array([])
        # create array of available points
        for age_index, age in enumerate(self.m_ages):
            for contrast in self.m_model_data[age_index][:,filter_index]:
                points = np.append(points, [age, contrast])
        
        points = points.reshape(-1, 2)
        values = np.empty((points.shape[0]))
        for i, line in enumerate(points):
            values[i] = _model_mass(*line)

        from scipy.interpolate import griddata, LinearNDInterpolator
        age_array = np.ones_like(evaluate_contrast) * self.m_age
        xi = np.column_stack((age_array, evaluate_contrast - self.m_host_magnitude ))
    
        return griddata(points, values, xi) 

    # def _interpolate_data_by_age(self, age):
    #     """
    #     Internal function to interpolate the ages when the input age is not in the list of model ages
    #     """
    #     # find the closest to models to the inout age
    #     closest_age_index = (np.abs(self.m_ages-age)).argmin()
    #     # find out whether the age is above or below the closest value and write to indices
    #     if age < self.m_ages[closest_age_index]:
    #         closest_age_indices = [closest_age_index -1, closest_age_index]
    #     elif age > self.m_ages[closest_age_index]:
    #         closest_age_indices = [closest_age_index, closest_age_index-1]
        
    #     lower_data = self.m_model_data[closest_age_indices[0]]
    #     upper_data = self.m_model_data[closest_age_indices[1]]

    #     # linear interpolation of the data
    #     interpolated_data = lower_data + (age - self.m_ages[closest_age_indices[0]]) * \
    #         (upper_data - lower_data) / (self.m_ages[closest_age_indices[1]] - self.m_ages[closest_age_indices[0]])

    #     return interpolated_data

    # def _interpolate_model(self, interpolate_age=False):
    #     """
    #     Internal function.
    #     Takes Model data and interpolates it
    #     """

    #     from scipy.interpolate import interp1d

    #     # find the filter corresponding to the m_filter
    #     filter_index = np.argwhere([self.m_filter == j for j in self.m_header])[0] # simple argwhere gives empty list?!
            
    #     # find the model corresponding to the m_age
    #     if not interpolate_age:
    #         age_index = np.argwhere(self.m_age == self.m_ages)[0][0]

    #         # grab the data to be interpolated
    #         mass = self.m_model_data[age_index] [:, 0]
    #         absoulteMagnitude = np.squeeze(self.m_model_data[age_index] [:, filter_index])
        
    #     else:
    #         # interpolate the data by age
    #         interpolated_age_data = self._interpolate_data_by_age(self.m_age)
    #         mass = interpolated_age_data[:,0]
    #         absoulteMagnitude = np.squeeze(interpolated_age_data[:, filter_index])
        
    #     # interpolate the data for one age
    #     interpols_mass_contrast = interp1d(absoulteMagnitude, mass, kind='linear', bounds_error = False)
    #     return interpols_mass_contrast

    
    def run(self):
        """
        Run method of the Module. Calculates the mass limits given precalculated contrast limits
        and a grid based model. Interpolates between the ages and contrast limits.
        """
        contrast_data = self.m_data_in_port.get_all()

        print(contrast_data)
        print(contrast_data.shape)
        r = contrast_data[:,0]
        contrast = contrast_data[:,1]
        contrast_std = np.sqrt(contrast_data[:,2])

        print(contrast)

        # interpolate_age_flag = not (self.m_age in self.m_ages)
        mass = self._interpolate_model(self.m_age, contrast) 
        mass_upper = self._interpolate_model(self.m_age, contrast + contrast_std) - mass
        mass_lower = self._interpolate_model(self.m_age, contrast - contrast_std) - mass

        print(mass)

        mass_summary = np.column_stack((r, mass, mass_upper, mass_lower))
        self.m_data_out_port.set_all(mass_summary, data_dim=2)

        sys.stdout.write("\rRunning MassCurveModule... [DONE]\n")
        sys.stdout.flush()

        history = " absolute Magnitude: " + str(self.m_host_magnitude) + " at distance: "\
            + str(self.m_distance) + "at age: " + str(self.m_age)
        self.m_data_out_port.add_history("MassCurveModule", history)
        self.m_data_out_port.copy_attributes(self.m_data_in_port)
        self.m_data_out_port.close_port()

class ContrastCurveModule(ProcessingModule):
    """
    Pipeline module to calculate contrast limits for a given sigma level or false positive
    fraction, with a correction for small sample statistics. Positions are processed in
    parallel if ``CPU`` is set to a value larger than 1 in the configuration file.
    """

    def __init__(self,
                 name_in="contrast",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 contrast_out_tag="contrast_limits",
                 separation=(0.1, 1., 0.01),
                 angle=(0., 360., 60.),
                 threshold=("sigma", 5.),
                 psf_scaling=1.,
                 aperture=0.05,
                 pca_number=20,
                 cent_size=None,
                 edge_size=None,
                 extra_rot=0.,
                 residuals="median",
                 snr_inject=100.,
                 **kwargs):
        """
        Constructor of ContrastCurveModule.

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
            Detection threshold for the contrast curve, either in terms of "sigma" or the false
            positive fraction (FPF). The value is a tuple, for example provided as ("sigma", 5.)
            or ("fpf", 1e-6). Note that when sigma is fixed, the false positive fraction will
            change with separation. Also, sigma only corresponds to the standard deviation of a
            normal distribution at large separations (i.e., large number of samples).
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should have a positive value.
        aperture : float
            Aperture radius (arcsec).
        pca_number : int
            Number of principal components used for the PSF subtraction.
        cent_size : float
            Central mask radius (arcsec). No mask is used when set to None.
        edge_size : float
            Outer edge radius (arcsec) beyond which pixels are masked. No outer mask is used when
            set to None. If the value is larger than half the image size then it will be set to
            half the image size.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method used for combining the residuals ("mean", "median", "weighted", or "clipped").
        snr_inject : float
            Signal-to-noise ratio of the injected planet signal that is used to measure the amount
            of self-subtraction.

        Returns
        -------
        NoneType
            None
        """

        super(ContrastCurveModule, self).__init__(name_in)

        if "sigma" in kwargs:
            warnings.warn("The 'sigma' parameter has been deprecated. Please use the 'threshold' "
                          "parameter instead.", DeprecationWarning)

        if "norm" in kwargs:
            warnings.warn("The 'norm' parameter has been deprecated. It is not recommended to "
                          "normalize the images before PSF subtraction.", DeprecationWarning)

        if "accuracy" in kwargs:
            warnings.warn("The 'accuracy' parameter has been deprecated. The parameter is no "
                          "longer required.", DeprecationWarning)

        if "magnitude" in kwargs:
            warnings.warn("The 'magnitude' parameter has been deprecated. The parameter is no "
                          "longer required.", DeprecationWarning)

        if "ignore" in kwargs:
            warnings.warn("The 'ignore' parameter has been deprecated. The parameter is no "
                          "longer required.", DeprecationWarning)

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

            raise ValueError("The angular positions of the fake planets should lie between "
                             "0 deg and 360 deg.")

    def run(self):
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
            raise ValueError('The number of frames in psf_in_tag {0} does not match with the '
                             'number of frames in image_in_tag {1}. The DerotateAndStackModule can '
                             'be used to average the PSF frames (without derotating) before '
                             'applying the ContrastCurveModule.'.format(psf.shape, images.shape))

        cpu = self._m_config_port.get_attribute("CPU")
        parang = self.m_image_in_port.get_attribute("PARANG")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

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

        sys.stdout.write("Running ContrastCurveModule...\r")
        sys.stdout.flush()

        positions = []
        for sep in pos_r:
            for ang in pos_t:
                positions.append((sep, ang))

        # Create a queue object which will contain the results
        queue = mp.Queue()

        result = []
        jobs = []

        working_place = self._m_config_port.get_attribute("WORKING_PLACE")

        # Create temporary files
        tmp_im_str = os.path.join(working_place, "tmp_images.npy")
        tmp_psf_str = os.path.join(working_place, "tmp_psf.npy")

        np.save(tmp_im_str, images)
        np.save(tmp_psf_str, psf)

        mask = create_mask(images.shape[-2:], [self.m_cent_size, self.m_edge_size])

        _, im_res = pca_psf_subtraction(images=images*mask,
                                        angles=-1.*parang+self.m_extra_rot,
                                        pca_number=self.m_pca_number)

        noise = combine_residuals(method=self.m_residuals, res_rot=im_res)

        for i, pos in enumerate(positions):
            process = mp.Process(target=contrast_limit,
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
                                       pos,
                                       queue),
                                 name=(str(os.path.basename(__file__)) + '_radius=' +
                                       str(np.round(pos[0]*pixscale, 1)) + '_angle=' +
                                       str(np.round(pos[1], 1))))

            jobs.append(process)

        for i, job in enumerate(jobs):
            job.start()

            if (i+1)%cpu == 0:
                # Start *cpu* number of processes. Wait for them to finish and start again *cpu*
                # number of processes.

                for k in jobs[i+1-cpu:(i+1)]:
                    k.join()

            elif (i+1) == len(jobs) and (i+1)%cpu != 0:
                # Wait for the last processes to finish if number of processes is not a multiple
                # of *cpu*

                for k in jobs[(i + 1 - (i+1)%cpu):]:
                    k.join()

            progress(i, len(jobs), "Running ContrastCurveModule...")

        # Send termination sentinel to queue
        queue.put(None)

        while True:
            item = queue.get()

            if item is None:
                break
            else:
                result.append(item)

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

        self.m_contrast_out_port.set_all(limits, data_dim=2)

        sys.stdout.write("\rRunning ContrastCurveModule... [DONE]\n")
        sys.stdout.flush()

        history = str(self.m_threshold[0])+" = "+str(self.m_threshold[1])

        self.m_contrast_out_port.add_history("ContrastCurveModule", history)
        self.m_contrast_out_port.copy_attributes(self.m_image_in_port)
        self.m_contrast_out_port.close_port()
