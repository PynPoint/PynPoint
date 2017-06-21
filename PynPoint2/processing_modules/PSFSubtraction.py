import numpy as np
from scipy import linalg, ndimage

from PynPoint.core.Processing import ProcessingModule
from PSFsubPreparation import PSFdataPreparation


class PSFSubtractionModule(ProcessingModule):

    def __init__(self,
                 pca_number,
                 name_in="PSF_subtraction",
                 images_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 res_arr_out_tag="res_arr",
                 res_arr_rot_out_tag="res_rot",
                 res_mean_tag="res_mean",
                 res_median_tag="res_median",
                 res_var_tag="res_var",
                 res_rot_mean_clip_tag="res_rot_mean_clip",
                 extra_rot=0.0,
                 **kwargs):

        super(PSFSubtractionModule, self).__init__(name_in)

        # additional keywords
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

        if "cent_remove" in kwargs:
            cent_remove = kwargs["cent_remove"]
        else:
            cent_remove = False

        if "cent_size" in kwargs:
            cent_size = kwargs["cent_size"]
            cent_remove = True
        else:
            cent_size = 0.05

        if "edge_size" in kwargs:
            edge_size = kwargs["edge_size"]
            cent_remove = True
        else:
            edge_size = 1.0

        if "ref_prep_tag" in kwargs:
            ref_prep_tag = kwargs["ref_prep_tag"]
        else:
            ref_prep_tag = "ref_prep"

        if "prep_tag" in kwargs:
            prep_tag = kwargs["prep_tag"]
        else:
            prep_tag = "data_prep"

        self.m_num_components = pca_number

        self._m_preparation_images = PSFdataPreparation(name_in="prep_im",
                                                        image_in_tag=images_in_tag,
                                                        image_out_tag=prep_tag,
                                                        image_mask_out_tag="not_needed",
                                                        mask_out_tag=cent_mask_tag,
                                                        cent_remove=cent_remove,
                                                        para_sort=False,
                                                        cent_size=cent_size,
                                                        edge_size=edge_size)

        self._m_preparation_images.m_image_mask_out_port.deactivate()

        self._m_preparation_reference = PSFdataPreparation(name_in="prep_ref",
                                                           image_in_tag=reference_in_tag,
                                                           image_out_tag=ref_prep_tag,
                                                           image_mask_out_tag="not_needed",
                                                           mask_out_tag=cent_mask_tag,
                                                           cent_remove=cent_remove,
                                                           para_sort=False,
                                                           cent_size=cent_size,
                                                           edge_size=edge_size)

        self._m_preparation_reference.m_image_mask_out_port.deactivate()

        self._m_make_pca_basis = MakePCABasisModule(im_arr_in_tag=ref_prep_tag,
                                                    im_arr_out_tag="not_needed",
                                                    im_average_out_tag=im_average_tag,
                                                    basis_out_tag=basis_tag)

        self._m_make_pca_basis.m_im_arr_out_port.deactivate()

        self._m_make_psf_model = MakePSFModelModule(num=pca_number,
                                                    im_arr_in_tag=prep_tag,
                                                    basis_in_tag=basis_tag,
                                                    basis_average_in_tag=im_average_tag,
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

        print "Preparing data..."
        self._m_preparation_images.run()
        self._m_preparation_reference.run()
        print "Finished preparing data..."
        print "Creating PCA-basis set..."
        self._m_make_pca_basis.run()
        print "Finished creating PCA-basis set..."
        print "Calculating PSF-Model..."
        self._m_make_psf_model.run()
        print "Finished calculating PSF-model..."
        print "Creating residuals..."
        self._m_residuals_module.run()
        print "Finished creating residuals..."

        # Take Header Information
        input_port = self._m_residuals_module.m_im_arr_in_port

        out_ports = [self._m_residuals_module.m_res_arr_out_port,
                     self._m_residuals_module.m_res_arr_rot_out_port,
                     self._m_residuals_module.m_res_mean_port,
                     self._m_residuals_module.m_res_median_port,
                     self._m_residuals_module.m_res_var_port,
                     self._m_residuals_module.m_res_rot_mean_clip_port]

        history = "PCA with " + str(self.m_num_components) + " PCA-components"
        for port in out_ports:
            port.copy_attributes_from_input_port(input_port)
            port.add_history_information("PSF_subtraction",
                                         history)

        out_ports[0].flush()


class CreateResidualsModule(ProcessingModule):

    def __init__(self,
                 name_in= "residuals_module",
                 im_arr_in_tag="im_arr",
                 psf_im_in_tag="psf_im",
                 mask_in_tag="cent_mask",
                 res_arr_out_tag="res_arr",
                 res_arr_rot_out_tag="res_rot",
                 res_mean_tag="res_mean",
                 res_median_tag="res_median",
                 res_var_tag="res_var",
                 res_rot_mean_clip_tag="res_rot_mean_clip",
                 extra_rot=0.0):

        super(CreateResidualsModule, self).__init__(name_in)

        # Inputs
        if mask_in_tag is None:
            self._m_mask_in_port = None
        else:
            self._m_mask_in_port = self.add_input_port(mask_in_tag)

        self.m_im_arr_in_port = self.add_input_port(im_arr_in_tag)
        self._m_psf_im_port = self.add_input_port(psf_im_in_tag)

        # Outputs
        self.m_res_arr_out_port = self.add_output_port(res_arr_out_tag)
        self.m_res_arr_rot_out_port = self.add_output_port(res_arr_rot_out_tag)
        self.m_res_mean_port = self.add_output_port(res_mean_tag)
        self.m_res_median_port = self.add_output_port(res_median_tag)
        self.m_res_var_port = self.add_output_port(res_var_tag)
        self.m_res_rot_mean_clip_port = self.add_output_port(res_rot_mean_clip_tag)

        self.m_extra_rot = extra_rot

    def run(self):
        im_data = self.m_im_arr_in_port.get_all()
        psf_im = self._m_psf_im_port.get_all()

        if self._m_mask_in_port is None:
            cent_mask = np.zeros((im_data.shape[1],
                                 im_data.shape[2]))
        else:
            cent_mask = self._m_mask_in_port.get_all()

        # create result array
        res_arr = im_data.copy()
        for i in range(0,
                       len(res_arr[:, 0, 0])):
            res_arr[i, ] -= (psf_im[i, ] * cent_mask)

        # rotate result array
        para_angles = self.m_im_arr_in_port.get_attribute("NEW_PARA")
        delta_para = - para_angles
        res_rot = np.zeros(shape=im_data.shape)
        for i in range(0, len(delta_para)):
            res_temp = res_arr[i, ]

            res_rot[i, ] = ndimage.rotate(res_temp,
                                          delta_para[i]+self.m_extra_rot,
                                          reshape=False)

        # create mean
        tmp_res_rot_mean = np.mean(res_rot,
                                   axis=0)

        # create median
        tmp_res_rot_median = np.median(res_rot,
                                       axis=0)

        # create variance
        res_rot_temp = res_rot.copy()
        for i in range(0,
                       res_rot_temp.shape[0]):

            res_rot_temp[i, ] -= - tmp_res_rot_mean
        res_rot_var = (res_rot_temp**2.).sum(axis=0)
        tmp_res_rot_var = res_rot_var

        # create mean clip
        res_rot_mean_clip = np.zeros(im_data[0,].shape)
        '''
        for i in range(0, res_rot_mean_clip.shape[0]):
            for j in range(0, res_rot_mean_clip.shape[1]):
                temp = res_rot[:, i, j]
                if temp.var() > 0.0:
                    a = temp - temp.mean()
                    b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
                    b2 = b1.compress((b1 > (-1.0)*3.0*np.sqrt(a.var())).flat)
                    res_rot_mean_clip[i, j] = temp.mean() + b2.mean()'''

        # save results
        self.m_res_arr_out_port.set_all(res_arr)
        self.m_res_arr_rot_out_port.set_all(res_rot)
        self.m_res_mean_port.set_all(tmp_res_rot_mean)
        self.m_res_median_port.set_all(tmp_res_rot_median)
        self.m_res_var_port.set_all(tmp_res_rot_var)
        self.m_res_rot_mean_clip_port.set_all(res_rot_mean_clip)

        self.m_res_arr_out_port.flush()


class MakePSFModelModule(ProcessingModule):
    """
    should be just a part of the whole processing
    """

    def __init__(self,
                 num,
                 name_in="psf_model_module",
                 im_arr_in_tag="im_arr",
                 basis_in_tag="basis_im",
                 basis_average_in_tag="basis_ave",
                 psf_basis_out_tag="psf_basis"):

        self.m_num = num

        super(MakePSFModelModule, self).__init__(name_in)

        # Inputs
        self._m_im_arr_in_port = self.add_input_port(im_arr_in_tag)
        self._m_basis_average_in_port = self.add_input_port(basis_average_in_tag)
        self._m_basis_in_port = self.add_input_port(basis_in_tag)

        self._m_psf_basis_out_tag = psf_basis_out_tag
        self._m_psf_basis_out_port = self.add_output_port(psf_basis_out_tag)

    def run(self):

        im_data = self._m_im_arr_in_port.get_all()
        basis_data = self._m_basis_in_port.get_all()
        basis_average = self._m_basis_average_in_port.get_all()

        temp_im_arr = np.zeros([im_data.shape[0],
                                im_data.shape[1]*im_data.shape[2]])

        for i in range(0, im_data.shape[0]):
            # Remove the mean used to build the basis. Might be able to speed this up
            temp_im_arr[i, ] = im_data[i, ].reshape(-1) - basis_average.reshape(-1)

        # use matrix multiplication
        coeff_temp = np.array((np.mat(temp_im_arr) *
                               np.mat(basis_data.reshape(basis_data.shape[0], -1)).T))
        psf_coeff = coeff_temp  # attach the full list of coefficients to input object

        psf_im = (np.mat(psf_coeff[:, 0: self.m_num]) *
                  np.mat(basis_data.reshape(basis_data.shape[0], -1)[0:self.m_num, ]))

        for i in range(0, im_data.shape[0]):  # Add the mean back to the image
            psf_im[i, ] += basis_average.reshape(-1)

        result = np.array(psf_im).reshape(im_data.shape[0],
                                          im_data.shape[1],
                                          im_data.shape[2])

        self._m_psf_basis_out_port.set_all(result,
                                           keep_attributes=True)

        self._m_psf_basis_out_port.add_attribute(name="psf_coeff",
                                                 value=psf_coeff,
                                                 static=False)
        self._m_psf_basis_out_port.flush()


class MakePCABasisModule(ProcessingModule):
    """
    should be just a part of the whole processing
    """

    def __init__(self,
                 name_in="make_pca_basis",
                 im_arr_in_tag="im_arr",
                 im_arr_out_tag="im_arr",
                 im_average_out_tag="im_ave",
                 basis_out_tag="basis"):

        super(MakePCABasisModule, self).__init__(name_in)

        # Inputs
        self._m_im_arr_in_port = self.add_input_port(im_arr_in_tag)

        # Outputs
        self.m_im_arr_out_port = self.add_output_port(im_arr_out_tag)
        self._m_im_average_out_port = self.add_output_port(im_average_out_tag)
        self._m_basis_out_port = self.add_output_port(basis_out_tag)

    @staticmethod
    def _make_average_sub(im_arr_in):
        im_ave = im_arr_in.mean(axis=0)

        for i in range(0, len(im_arr_in[:,0,0])):
            im_arr_in[i,] -= im_ave
        return im_arr_in, im_ave

    def run(self):

        im_data = self._m_im_arr_in_port.get_all()

        num_entries = im_data.shape[0]
        im_size = [im_data.shape[1],
                   im_data.shape[2]]

        tmp_im_data, tmp_im_ave = self._make_average_sub(im_data)

        _,_,V = linalg.svd(tmp_im_data.reshape(num_entries,
                                               im_size[0]*im_size[1]),
                           full_matrices=False)

        basis_pca_arr = V.reshape(V.shape[0], im_size[0], im_size[1])

        self.m_im_arr_out_port.set_all(tmp_im_data, keep_attributes=True)
        self._m_im_average_out_port.set_all(tmp_im_ave)
        self._m_basis_out_port.set_all(basis_pca_arr)
        self._m_basis_out_port.add_attribute(name="basis_type",
                                             value="pca")

        self._m_basis_out_port.flush()