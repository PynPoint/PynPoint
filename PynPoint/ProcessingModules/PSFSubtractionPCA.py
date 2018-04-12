"""
Modules for PSF subtraction with principle component analysis.
"""

import warnings
from copy import deepcopy
from sys import platform, stdout

import numpy as np

from sklearn.decomposition import PCA
from scipy import linalg, ndimage, sparse

from PynPoint.Util.ModuleTools import progress
from PynPoint.Util.MultiprocessingPCA import PcaMultiprocessingCapsule
from PynPoint.Core.Processing import ProcessingModule
from PynPoint.ProcessingModules.PSFpreparation import PSFpreparationModule


class PSFSubtractionModule(ProcessingModule):
    """
    Module to perform PCA-based PSF subtraction. This module is a wrapper that prepares the data,
    creates the PCA basis, models the PSF, and creates the image residuals. The implementation of
    the PSF subtraction is the same as in PynPoint 0.2.0 (Amara & Quanz 2012; Amara et al. 2015).
    """

    def __init__(self,
                 pca_number,
                 svd="lapack",
                 name_in="PSF_subtraction",
                 images_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 res_arr_out_tag="res_arr",
                 res_arr_rot_out_tag="res_rot",
                 res_mean_tag="res_mean",
                 res_median_tag="res_median",
                 res_var_tag="res_var",
                 res_rot_mean_clip_tag="res_rot_mean_clip",
                 extra_rot=0.,
                 **kwargs):
        """
        Constructor of PSFSubtractionModule.

        :param pca_number: Number of principle components used for the PSF subtraction.
        :type pca_number: int
        :param svd: Method used for the singular value composition (*lapack* or *arpack*).
        :type svd: str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param images_in_tag: Tag of the database entry with the science images that are read
                              as input.
        :type images_in_tag: str
        :param reference_in_tag: Tag of the database entry with the reference images that are
                                 read as input.
        :type reference_in_tag: str
        :param res_arr_out_tag: Tag of the database entry with the image residuals from the PSF
                                subtraction.
        :type res_arr_out_tag: str
        :param res_arr_rot_out_tag: Tag of the database entry with the image residuals from the
                                    PSF subtraction that are rotated by PARANG to a common
                                    orientation.
        :type res_arr_rot_out_tag: str
        :param res_mean_tag: Tag of the database entry with the mean collapsed residuals.
        :type res_mean_tag: str
        :param res_median_tag: Tag of the database entry with the median collapsed residuals.
        :type res_median_tag: str
        :param res_var_tag: Tag of the database entry with the variance of the pixel values across
                            the stack of residuals.
        :type res_var_tag: str
        :param res_rot_mean_clip_tag: Tag of the database entry of the clipped mean residuals.
        :type res_rot_mean_clip_tag: str
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float
        :param \**kwargs:
            See below.

        :Keyword arguments:
             * **basis_out_tag** (*str*) -- Tag of the database entry with the basis set.
             * **image_ave_tag** (*str*) -- Tag of the database entry with the mean of the image
                                            stack subtracted from all images.
             * **psf_model_tag** (*str*) -- Tag of the database entry with the constructed model
                                            PSF of each image.
             * **cent_mask_tag** (*str*) -- Tag of the database entry with the circular mask of
                                            the inner and outer regions.
             * **cent_size** (*float*) -- Radius of the central mask (arcsec). No mask is used
                                          when set to None.
             * **edge_size** (*float*) -- Outer radius (arcsec) beyond which pixels are masked.
                                          No outer mask is used when set to None. If the value
                                          is larger than half the image size then it will be
                                          set to half the image size.
             * **prep_tag** (*str*) -- Tag of the database entry with the prepared science data.
             * **ref_prep_tag** (*str*) -- Tag of the database entry the prepared reference data.
             * **verbose** (*bool*) -- Print progress to the standard output.

        :return: None
        """

        super(PSFSubtractionModule, self).__init__(name_in)

        if "basis_out_tag" in kwargs:
            basis_tag = kwargs["basis_out_tag"]
        else:
            basis_tag = "pca_basis_set"

        if "image_ave_tag" in kwargs:
            im_average_tag = kwargs["image_ave_tag"]
        else:
            im_average_tag = "image_average"

        if "psf_model_tag" in kwargs:
            psf_model_tag = kwargs["psf_model_tag"]
        else:
            psf_model_tag = "psf_model"

        if "cent_mask_tag" in kwargs:
            cent_mask_tag = kwargs["cent_mask_tag"]
        else:
            cent_mask_tag = "cent_mask_tag"

        if "cent_size" in kwargs:
            cent_size = kwargs["cent_size"]
        else:
            cent_size = None

        if "edge_size" in kwargs:
            edge_size = kwargs["edge_size"]
        else:
            edge_size = None

        if "norm" in kwargs:
            norm = kwargs["norm"]
        else:
            norm = True

        if "prep_tag" in kwargs:
            prep_tag = kwargs["prep_tag"]
        else:
            prep_tag = "data_prep"

        if "ref_prep_tag" in kwargs:
            ref_prep_tag = kwargs["ref_prep_tag"]
        else:
            ref_prep_tag = "ref_prep"

        if "verbose" in kwargs:
            self.m_verbose = kwargs["verbose"]
        else:
            self.m_verbose = True

        self.m_pca_number = pca_number
        self.m_svd = svd

        self._m_preparation_images = PSFpreparationModule(name_in="prep_im",
                                                          image_in_tag=images_in_tag,
                                                          image_out_tag=prep_tag,
                                                          mask_out_tag=cent_mask_tag,
                                                          norm=norm,
                                                          cent_size=cent_size,
                                                          edge_size=edge_size,
                                                          verbose=False)

        self._m_preparation_reference = PSFpreparationModule(name_in="prep_ref",
                                                             image_in_tag=reference_in_tag,
                                                             image_out_tag=ref_prep_tag,
                                                             mask_out_tag=cent_mask_tag,
                                                             norm=norm,
                                                             cent_size=cent_size,
                                                             edge_size=edge_size,
                                                             verbose=False)

        self._m_make_pca_basis = MakePCABasisModule(pca_number=self.m_pca_number,
                                                    svd=self.m_svd,
                                                    im_arr_in_tag=ref_prep_tag,
                                                    im_arr_out_tag="not_needed",
                                                    im_average_out_tag=im_average_tag,
                                                    basis_out_tag=basis_tag)

        self._m_make_pca_basis.m_im_arr_out_port.deactivate()

        self._m_make_psf_model = MakePSFModelModule(num=pca_number,
                                                    im_arr_in_tag=prep_tag,
                                                    im_ave_in_tag=im_average_tag,
                                                    basis_in_tag=basis_tag,
                                                    psf_basis_out_tag=psf_model_tag)

        self._m_residuals_module = \
            CreateResidualsModule(im_arr_in_tag=prep_tag,
                                  psf_im_in_tag=psf_model_tag,
                                  mask_in_tag=cent_mask_tag,
                                  res_arr_out_tag=res_arr_out_tag,
                                  res_arr_rot_out_tag=res_arr_rot_out_tag,
                                  res_mean_tag=res_mean_tag,
                                  res_median_tag=res_median_tag,
                                  res_var_tag=res_var_tag,
                                  res_rot_mean_clip_tag=res_rot_mean_clip_tag,
                                  extra_rot=extra_rot)

    def get_all_input_tags(self):
        return self._m_preparation_images.get_all_input_tags() +\
            self._m_preparation_reference.get_all_input_tags()

    def get_all_output_tags(self):
        return self._m_residuals_module.get_all_output_tags() + \
               self._m_preparation_images.get_all_output_tags() + \
               self._m_make_pca_basis.get_all_output_tags() + \
               self._m_preparation_reference.get_all_output_tags() + \
               self._m_make_psf_model.get_all_output_tags()

    def connect_database(self,
                         data_base_in):

        self._m_preparation_images.connect_database(data_base_in)
        self._m_preparation_reference.connect_database(data_base_in)
        self._m_make_pca_basis.connect_database(data_base_in)
        self._m_make_psf_model.connect_database(data_base_in)
        self._m_residuals_module.connect_database(data_base_in)

        super(PSFSubtractionModule, self).connect_database(data_base_in)

    def run(self):
        """
        Run method of the module. Wrapper that prepares the data, creates the PCA basis, constructs
        the PSF model, and creates the residuals.

        :return: None
        """

        warnings.warn("PSFSubtractionModule is deprecated and will be removed in the next "
                      "release. Please use PcaPsfSubtractionModule instead.",
                      DeprecationWarning)

        if self.m_verbose:
            stdout.write("Preparing PSF subtraction...")
            stdout.flush()
        self._m_preparation_images.run()
        self._m_preparation_reference.run()
        if self.m_verbose:
            stdout.write(" [DONE]\n")
            stdout.write("Creating PCA basis set...")
            stdout.flush()
        self._m_make_pca_basis.run()
        if self.m_verbose:
            stdout.write(" [DONE]\n")
            stdout.write("Constructing PSF model...")
            stdout.flush()
        self._m_make_psf_model.run()
        if self.m_verbose:
            stdout.write(" [DONE]\n")
            stdout.write("Creating residuals...")
            stdout.flush()
        self._m_residuals_module.run()
        if self.m_verbose:
            stdout.write(" [DONE]\n")

        input_port = self._m_residuals_module.m_im_arr_in_port

        out_ports = [self._m_residuals_module.m_res_arr_out_port,
                     self._m_residuals_module.m_res_arr_rot_out_port,
                     self._m_residuals_module.m_res_mean_port,
                     self._m_residuals_module.m_res_median_port,
                     self._m_residuals_module.m_res_var_port,
                     self._m_residuals_module.m_res_rot_mean_clip_port]

        for port in out_ports:
            port.copy_attributes_from_input_port(input_port)
            port.add_history_information("PSF subtraction",
                                         "PCA with "+str(self.m_pca_number)+"components")

        out_ports[0].flush()


class CreateResidualsModule(ProcessingModule):
    """
    Module to create the residuals of the PSF subtraction.
    """

    def __init__(self,
                 name_in="residuals_module",
                 im_arr_in_tag="im_arr",
                 psf_im_in_tag="psf_im",
                 mask_in_tag="cent_mask",
                 res_arr_out_tag="res_arr",
                 res_arr_rot_out_tag="res_rot",
                 res_mean_tag="res_mean",
                 res_median_tag="res_median",
                 res_var_tag="res_var",
                 res_rot_mean_clip_tag="res_rot_mean_clip",
                 extra_rot=0.):
        """
        Constructor of CreateResidualsModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param im_arr_in_tag: Tag of the database entry with the science images that are read
                              as input.
        :type im_arr_in_tag: str
        :param psf_im_in_tag: Tag of the database entry with the model PSF images that are read
                              as input.
        :type psf_im_in_tag: str
        :param mask_in_tag: Tag of the database entry with the mask.
        :type mask_in_tag: str
        :param res_arr_out_tag: Tag of the database entry with the image residuals from the PSF
                                subtraction.
        :type res_arr_out_tag: str
        :param res_arr_rot_out_tag: Tag of the database entry with the image residuals from the
                                    PSF subtraction that are rotated by PARANG to a common
                                    orientation.
        :type res_arr_rot_out_tag: str
        :param res_mean_tag: Tag of the database entry with the mean collapsed residuals.
        :type res_mean_tag: str
        :param res_median_tag: Tag of the database entry with the median collapsed residuals.
        :type res_median_tag: str
        :param res_var_tag: Tag of the database entry with the variance of the pixel values across
                            the stack of residuals.
        :type res_var_tag: str
        :param res_rot_mean_clip_tag: Tag of the database entry of the clipped mean residuals.
        :type res_rot_mean_clip_tag: str
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(CreateResidualsModule, self).__init__(name_in)

        if mask_in_tag is None:
            self._m_mask_in_port = None
        else:
            self._m_mask_in_port = self.add_input_port(mask_in_tag)

        self.m_im_arr_in_port = self.add_input_port(im_arr_in_tag)
        self._m_psf_im_port = self.add_input_port(psf_im_in_tag)

        self.m_res_arr_out_port = self.add_output_port(res_arr_out_tag)
        self.m_res_arr_rot_out_port = self.add_output_port(res_arr_rot_out_tag)
        self.m_res_mean_port = self.add_output_port(res_mean_tag)
        self.m_res_median_port = self.add_output_port(res_median_tag)
        self.m_res_var_port = self.add_output_port(res_var_tag)
        self.m_res_rot_mean_clip_port = self.add_output_port(res_rot_mean_clip_tag)

        self.m_extra_rot = extra_rot

    def run(self):
        """
        Run method of the module. Creates the residuals of the PSF subtraction, rotates the images
        to a common orientation, and calculates the mean, median, variance, and clipped mean.

        :return: None
        """

        im_data = self.m_im_arr_in_port.get_all()
        psf_im = self._m_psf_im_port.get_all()

        if self._m_mask_in_port is None:
            cent_mask = np.zeros((im_data.shape[1], im_data.shape[2]))
        else:
            cent_mask = self._m_mask_in_port.get_all()

        # create result array
        res_arr = im_data.copy()
        for i in range(res_arr.shape[0]):
            res_arr[i, ] -= (psf_im[i, ] * cent_mask)

        # rotate result array
        delta_para = -1.*self.m_im_arr_in_port.get_attribute("PARANG")
        res_rot = np.zeros(shape=res_arr.shape)

        for i in range(0, len(delta_para)):
            res_temp = res_arr[i, ]
            # ndimage.rotate rotates in clockwise direction for positive angles
            res_rot[i, ] = ndimage.rotate(res_temp,
                                          delta_para[i]+self.m_extra_rot,
                                          reshape=False)

        # create mean
        tmp_res_rot_mean = np.mean(res_rot, axis=0)

        # create median
        tmp_res_rot_median = np.median(res_rot, axis=0)

        # create variance
        res_rot_temp = res_rot.copy()
        for i in range(res_rot_temp.shape[0]):
            res_rot_temp[i, ] -= - tmp_res_rot_mean

        res_rot_var = (res_rot_temp**2.).sum(axis=0)
        tmp_res_rot_var = res_rot_var

        # create mean clip
        res_rot_mean_clip = np.zeros(im_data[0,].shape)

        for i in range(0, res_rot_mean_clip.shape[0]):
            for j in range(0, res_rot_mean_clip.shape[1]):
                temp = res_rot[:, i, j]
                if temp.var() > 0.0:
                    mean_sub = temp - temp.mean()
                    clip1 = mean_sub.compress((mean_sub < 3.0*np.sqrt(mean_sub.var())).flat)
                    clip2 = clip1.compress((clip1 > -3.0*np.sqrt(mean_sub.var())).flat)
                    res_rot_mean_clip[i, j] = temp.mean() + clip2.mean()

        self.m_res_arr_out_port.set_all(res_arr)
        self.m_res_arr_rot_out_port.set_all(res_rot)
        self.m_res_mean_port.set_all(tmp_res_rot_mean)
        self.m_res_median_port.set_all(tmp_res_rot_median)
        self.m_res_var_port.set_all(tmp_res_rot_var)
        self.m_res_rot_mean_clip_port.set_all(res_rot_mean_clip)

        self.m_res_arr_out_port.flush()


class MakePSFModelModule(ProcessingModule):
    """
    Module to create the PSF model.
    """

    def __init__(self,
                 num,
                 name_in="psf_model_module",
                 im_arr_in_tag="im_arr",
                 im_ave_in_tag="im_ave",
                 basis_in_tag="basis_im",
                 psf_basis_out_tag="psf_basis"):
        """
        Constructor of MakePSFModelModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param im_arr_in_tag: Tag of the database entry with the science images that are read
                              as input.
        :type im_arr_in_tag: str
        :param im_ave_in_tag: Tag of the database entry with the mean of the image stack.
        :type im_ave_in_tag: str
        :param basis_in_tag: Tag of the database entry with the basis set.
        :type basis_in_tag: str
        :param psf_basis_out_tag: Tag of the database entry with the image residuals from the PSF
                                  subtraction.
        :type psf_basis_out_tag: str

        :return: None
        """

        super(MakePSFModelModule, self).__init__(name_in)

        self._m_im_arr_in_port = self.add_input_port(im_arr_in_tag)
        self._m_im_ave_in_port = self.add_input_port(im_ave_in_tag)
        self._m_basis_in_port = self.add_input_port(basis_in_tag)
        self._m_psf_basis_out_port = self.add_output_port(psf_basis_out_tag)

        self.m_num = num

    def run(self):
        """
        Run method of the module. Reshapes the images to 1D, subtracts the mean of the image stack
        from all images, calculates the coefficients of the basis, calculates the PSF model of all
        images, adds the mean of the original stack back to the images.

        :return: None
        """

        im_data = self._m_im_arr_in_port.get_all()
        im_ave = self._m_im_ave_in_port.get_all()
        basis_data = self._m_basis_in_port.get_all()

        temp_im_arr = np.zeros([im_data.shape[0], im_data.shape[1]*im_data.shape[2]])

        # Remove the mean used to build the basis
        for i in range(im_data.shape[0]):
            temp_im_arr[i, ] = im_data[i, ].reshape(-1) - im_ave.reshape(-1)

        # Use matrix multiplication to calculate the coefficients
        psf_coeff = np.array((np.mat(temp_im_arr) * \
                             np.mat(basis_data.reshape(basis_data.shape[0], -1)).T))

        # Model PSF
        psf_im = (np.mat(psf_coeff[:, 0: self.m_num]) *
                  np.mat(basis_data.reshape(basis_data.shape[0], -1)[0:self.m_num, ]))

        # Add the mean back to the image
        for i in range(im_data.shape[0]):
            psf_im[i, ] += im_ave.reshape(-1)

        result = np.array(psf_im).reshape(im_data.shape[0],
                                          im_data.shape[1],
                                          im_data.shape[2])

        self._m_psf_basis_out_port.set_all(result, keep_attributes=True)
        self._m_psf_basis_out_port.add_attribute(name="psf_coeff", value=psf_coeff, static=False)
        self._m_psf_basis_out_port.flush()


class MakePCABasisModule(ProcessingModule):
    """
    Module to create a PCA basis set of a stack of images through singular value decomposition.
    """

    def __init__(self,
                 pca_number=None,
                 svd="lapack",
                 name_in="make_pca_basis",
                 im_arr_in_tag="im_arr",
                 im_arr_out_tag="im_arr",
                 im_average_out_tag="im_av",
                 basis_out_tag="basis"):
        """
        Constructor of MakePCABasisModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param im_arr_in_tag: Tag of the database entry with the science images that are read
                              as input.
        :type im_arr_in_tag: str
        :param im_arr_out_tag: Tag of the database entry with the science images from which the
                               mean of the image stack is subtracted.
        :type im_arr_out_tag: str
        :param im_average_out_tag: Tag of the database entry with the mean of the image stack.
        :type im_average_out_tag: str
        :param basis_out_tag: Tag of the database entry with the create PCA basis set.
        :type basis_out_tag: str

        :return: None
        """

        super(MakePCABasisModule, self).__init__(name_in)

        self._m_im_arr_in_port = self.add_input_port(im_arr_in_tag)
        self.m_im_arr_out_port = self.add_output_port(im_arr_out_tag)
        self._m_im_average_out_port = self.add_output_port(im_average_out_tag)
        self._m_basis_out_port = self.add_output_port(basis_out_tag)

        self.m_pca_number = pca_number
        self.m_svd = svd

    @staticmethod
    def _make_average_sub(im_arr_in):
        """
        Internal method to subtract the mean of the image stack from all images.
        """

        im_ave = np.mean(im_arr_in, axis=0)

        for i in range(im_arr_in.shape[0]):
            im_arr_in[i,] -= im_ave

        return im_arr_in, im_ave

    def run(self):
        """
        Run method of the module. Subtracts the mean of the image stack from all images, reshapes
        the stack of images into a 2D array, uses singular value decomposition to construct the
        orthogonal basis set, and reshapes the basis set into a 3D array. Note that linalg.svd
        calculates the maximum number of principle components which is equal to the number of
        images.

        :return: None
        """

        im_data = self._m_im_arr_in_port.get_all()

        num_entries = im_data.shape[0]
        im_size = [im_data.shape[1], im_data.shape[2]]

        if self.m_pca_number is None:
            self.m_pca_number = num_entries - 1

        tmp_im_data, tmp_im_ave = self._make_average_sub(im_data)

        if self.m_svd == "lapack":
            _, _, v_svd = linalg.svd(tmp_im_data.reshape(num_entries,
                                                         im_size[0]*im_size[1]),
                                     full_matrices=False,
                                     compute_uv=True)

        elif self.m_svd == "arpack":
            _, _, v_svd = sparse.linalg.svds(tmp_im_data.reshape(num_entries,
                                                                 im_size[0]*im_size[1]),
                                             k=self.m_pca_number,
                                             return_singular_vectors=True)

            v_svd = v_svd[::-1, ]

        else:
            raise ValueError("The svd argument should be set to either 'lapack' or 'arpack'.")

        basis_pca_arr = v_svd.reshape(v_svd.shape[0], im_size[0], im_size[1])

        self.m_im_arr_out_port.set_all(tmp_im_data, keep_attributes=True)
        self._m_im_average_out_port.set_all(tmp_im_ave)
        self._m_basis_out_port.set_all(basis_pca_arr)

        self._m_basis_out_port.flush()


class PcaPsfSubtractionModule(ProcessingModule):
    """
    Module for fast (compared to PSFSubtractionModule) PCA subtraction. The multiprocessing
    implementation is only supported for Linux and Windows. Mac only runs in single processing
    due to a bug in the numpy package.
    """

    def __init__(self,
                 pca_numbers,
                 name_in="PSF_subtraction",
                 images_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 res_mean_tag="res_mean",
                 res_median_tag=None,
                 res_arr_out_tag=None,
                 res_rot_mean_clip_tag=None,
                 extra_rot=0.,
                 **kwargs):
        """
        Constructor of PcaPsfSubtractionModule.

        :param pca_numbers: Number of PCA components used for the PSF model. Can be a single value
                            or a list of integers. A list of PCAs will be processed (if supported)
                            using multiprocessing.
        :type pca_numbers: int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param images_in_tag: Tag of the database entry with the science images that are read
                              as input.
        :type images_in_tag: str
        :param reference_in_tag: Tag of the database entry with the reference images that are
                                 read as input.
        :type reference_in_tag: str
        :param res_mean_tag: Tag of the database entry with the mean collapsed residuals.
        :type res_mean_tag: str
        :param res_median_tag: Tag of the database entry with the median collapsed residuals. Not
                               calculated if set to *None*.
        :type res_median_tag: str
        :param res_arr_out_tag: Tag of the database entry with the image residuals from the PSF
                                subtraction. If a list of PCs is provided in *pca_numbers* then
                                multiple tags will be created in the central database. Not
                                calculated if set to *None*.
        :type res_arr_out_tag: str
        :param res_rot_mean_clip_tag: Tag of the database entry of the clipped mean residuals. Not
                                      calculated if set to *None*.
        :type res_rot_mean_clip_tag: str
        :param extra_rot: Additional rotation angle of the images (deg).
        :type extra_rot: float
        :param \**kwargs:
            See below.

        :Keyword arguments:
             * **basis_out_tag** (*str*) -- Tag of the database entry with the basis set.
             * **verbose** (*bool*) -- Print progress to the standard output.

        :return: None
        """

        super(PcaPsfSubtractionModule, self).__init__(name_in)

        if "basis_out_tag" in kwargs:
            self.m_basis_tag = kwargs["basis_out_tag"]
        else:
            self.m_basis_tag = None

        if "verbose" in kwargs:
            self.m_verbose = kwargs["verbose"]
        else:
            self.m_verbose = True

        # look for the maximum number of components
        self.m_max_pacs = np.max(pca_numbers)
        self.m_components = np.sort(np.atleast_1d(pca_numbers))
        self.m_extra_rot = extra_rot

        self.m_pca = PCA(n_components=self.m_max_pacs, svd_solver="arpack")

        # add input Ports
        self.m_reference_in_port = self.add_input_port(reference_in_tag)
        self.m_star_in_port = self.add_input_port(images_in_tag)

        # create output if none create special not used port
        self.m_res_mean_out_port = self.add_output_port(str(res_mean_tag))

        # None check
        if res_median_tag is None:
            self.m_res_median_out_port = self.add_output_port(str("no median"))
            self.m_res_median_out_port.deactivate()
        else:
            self.m_res_median_out_port = self.add_output_port(str(res_median_tag))

        if res_rot_mean_clip_tag is None:
            self.m_res_rot_mean_clip_out_port = self.add_output_port(str("no clip"))
            self.m_res_rot_mean_clip_out_port.deactivate()
        else:
            self.m_res_rot_mean_clip_out_port = self.add_output_port(str(res_rot_mean_clip_tag))

        if self.m_basis_tag is not None:
            self.m_basis_out_port = self.add_output_port(self.m_basis_tag)

        # use a dict to store output ports for the non-stacked residuals
        self.m_res_arr_out_ports = {}
        self.m_res_arr_required = True

        for pca_number in self.m_components:
            # (cast to string for None case)
            # if res_arr_out_tag is None we still get different Tag names like None02
            self.m_res_arr_out_ports[pca_number] = self.add_output_port(str(res_arr_out_tag)
                                                                        + str(pca_number))

        # deactivate not needed array out ports
        if res_arr_out_tag is None:
            self.m_res_arr_required = False
            for port in self.m_res_arr_out_ports.itervalues():
                port.deactivate()

    def _run_multi_processing(self, star_data):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using multiprocessing.

        :return: None
        """

        tmp_output = np.zeros((len(self.m_components), star_data.shape[1], star_data.shape[2]))

        self.m_res_mean_out_port.set_all(tmp_output, keep_attributes=False)
        self.m_res_median_out_port.set_all(tmp_output, keep_attributes=False)
        self.m_res_rot_mean_clip_out_port.set_all(tmp_output, keep_attributes=False)

        cpu_count = self._m_config_port.get_attribute("CPU")

        rotations = -1.*self.m_star_in_port.get_attribute("PARANG")
        rotations += np.ones(rotations.shape[0]) * self.m_extra_rot

        pca_capsule = PcaMultiprocessingCapsule(self.m_res_mean_out_port,
                                                self.m_res_median_out_port,
                                                self.m_res_rot_mean_clip_out_port,
                                                cpu_count,
                                                deepcopy(self.m_components),
                                                deepcopy(self.m_pca),
                                                deepcopy(star_data),
                                                deepcopy(rotations),
                                                result_requirements=(False, False, False))
        pca_capsule.run()

    def _run_single_processing(self, star_sklearn, star_data):
        """
        Internal function to create the residuals, derotate the images, and write the output
        using a single process.

        :return: None
        """

        self.m_res_mean_out_port.del_all_data()
        self.m_res_median_out_port.del_all_data()
        self.m_res_rot_mean_clip_out_port.del_all_data()

        for i, pca_number in enumerate(self.m_components):
            if self.m_verbose:
                progress(i, len(self.m_components), "Creating residuals...")

            tmp_pca_representation = np.matmul(self.m_pca.components_[:pca_number],
                                               star_sklearn.T)

            tmp_pca_representation = np.vstack((tmp_pca_representation,
                                                np.zeros((self.m_max_pacs - pca_number,
                                                          star_data.shape[0])))).T

            tmp_psf_images = self.m_pca.inverse_transform(tmp_pca_representation)
            tmp_psf_images = tmp_psf_images.reshape((star_data.shape[0],
                                                     star_data.shape[1],
                                                     star_data.shape[2]))

            # subtract the psf model of the star
            tmp_without_psf = star_data - tmp_psf_images

            # inverse rotation
            delta_para = -1.*self.m_star_in_port.get_attribute("PARANG")
            res_array = np.zeros(shape=tmp_without_psf.shape)
            for j, item in enumerate(delta_para):
                res_temp = tmp_without_psf[j, ]
                # ndimage.rotate rotates in clockwise direction for positive angles
                res_array[j, ] = ndimage.rotate(res_temp, item+self.m_extra_rot, reshape=False)

            # create residuals
            # 1.) The de-rotated result images
            if self.m_res_arr_required:
                self.m_res_arr_out_ports[pca_number].set_all(res_array)
                self.m_res_arr_out_ports[pca_number].copy_attributes_from_input_port(
                    self.m_star_in_port)
                self.m_res_arr_out_ports[pca_number].add_history_information("PSF subtraction",
                                                                             "PcaPsfSubtractionModule")

            # 2.) mean
            tmp_res_rot_mean = np.mean(res_array, axis=0)
            self.m_res_mean_out_port.append(tmp_res_rot_mean, data_dim=3)

            # 3.) median
            if self.m_res_median_out_port.m_activate:
                tmp_res_rot_median = np.median(res_array, axis=0)
                self.m_res_median_out_port.append(tmp_res_rot_median, data_dim=3)

            # 4.) clipped mean
            if self.m_res_rot_mean_clip_out_port.m_activate:
                res_rot_temp = res_array.copy()
                for j in range(res_rot_temp.shape[0]):
                    res_rot_temp[j, ] -= - tmp_res_rot_mean

                res_rot_var = (res_rot_temp ** 2.).sum(axis=0)
                tmp_res_rot_var = res_rot_var
                self.m_res_rot_mean_clip_out_port.append(tmp_res_rot_var, data_dim=3)

        if self.m_verbose:
            stdout.write("Creating residuals... [DONE]\n")
            stdout.flush()

    def run(self):
        """
        Run method of the module. Subtracts the mean of the image stack from all images, reshapes
        the stack of images into a 2D array, uses singular value decomposition to construct the
        orthogonal basis set, calculates the PCA coefficients for each image, subtracts the PSF
        model, and writes the residuals as output.

        :return: None
        """

        # get all data and subtract the mean
        star_data = self.m_star_in_port.get_all()
        mean_star = np.mean(star_data, axis=0)
        star_data -= mean_star

        if self.m_reference_in_port.tag == self.m_star_in_port.tag:
            ref_star_data = deepcopy(star_data)

        else:
            ref_star_data = self.m_reference_in_port.get_all()
            mean_ref_star = np.mean(ref_star_data, axis=0)
            ref_star_data -= mean_ref_star

        # Fit the PCA model
        if self.m_verbose:
            stdout.write("Constructing PSF model...")
            stdout.flush()

        ref_star_sklearn = star_data.reshape((ref_star_data.shape[0],
                                              ref_star_data.shape[1] * ref_star_data.shape[2]))
        self.m_pca.fit(ref_star_sklearn)

        if self.m_verbose:
            stdout.write(" [DONE]\n")
            stdout.flush()

        if self.m_basis_tag is not None:
            basis = self.m_pca.components_.reshape((self.m_pca.components_.shape[0],
                                                    star_data.shape[1], star_data.shape[2]))
            self.m_basis_out_port.set_all(basis)

        # prepare the data for sklearns PCA
        star_sklearn = star_data.reshape((star_data.shape[0],
                                          star_data.shape[1] * star_data.shape[2]))

        # multiprocessing crashed on Mac in combination with numpy
        if platform == "darwin" or self.m_res_arr_required:
            self._run_single_processing(star_sklearn, star_data)
        else:
            if self.m_verbose:
                stdout.write("Creating residuals...")
                stdout.flush()
            self._run_multi_processing(star_data)
            if self.m_verbose:
                stdout.write(" [DONE]\n")
                stdout.flush()

        # save history for all other ports
        self.m_res_mean_out_port.copy_attributes_from_input_port(self.m_star_in_port)
        self.m_res_median_out_port.copy_attributes_from_input_port(self.m_star_in_port)
        self.m_res_rot_mean_clip_out_port.copy_attributes_from_input_port(self.m_star_in_port)

        self.m_res_mean_out_port.add_history_information("PSF subtraction", "PcaPsfSubtractionModule")
        self.m_res_median_out_port.add_history_information("PSF subtraction", "PcaPsfSubtractionModule")
        self.m_res_rot_mean_clip_out_port.add_history_information("PSF subtraction", "PcaPsfSubtractionModule")

        self.m_res_mean_out_port.close_port()
