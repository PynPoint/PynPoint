from scipy import linalg, ndimage
import numpy as np

from PynPoint.Processing import ProcessingModule


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

        self._m_im_arr_in_tag = im_arr_in_tag
        self.add_input_port(im_arr_in_tag)

        self._m_psf_im_tag = psf_im_in_tag
        self.add_input_port(psf_im_in_tag)

        self._m_mask_in_tag = mask_in_tag
        self.add_input_port(mask_in_tag)

        # Outputs
        self._m_res_arr_out_tag = res_arr_out_tag
        self.add_output_port(res_arr_out_tag)

        self._m_res_arr_rot_out_tag = res_arr_rot_out_tag
        self.add_output_port(res_arr_rot_out_tag)

        self._m_res_mean_tag = res_mean_tag
        self.add_output_port(res_mean_tag)

        self._m_res_median_tag = res_median_tag
        self.add_output_port(res_median_tag)

        self._m_res_var_tag = res_var_tag
        self.add_output_port(res_var_tag)

        self._m_res_rot_mean_clip_tag = res_rot_mean_clip_tag
        self.add_output_port(res_rot_mean_clip_tag)

        self.m_extra_rot = extra_rot

    def run(self):
        im_data = self._m_input_ports[self._m_im_arr_in_tag].get_all()
        psf_im = self._m_input_ports[self._m_psf_im_tag].get_all()
        cent_mask = self._m_input_ports[self._m_mask_in_tag].get_all()

        # create result array
        res_arr = im_data.copy()
        for i in range(0,
                       len(res_arr[:, 0, 0])):
            res_arr[i, ] -= (psf_im[i, ] * cent_mask)

        # rotate result array
        para_angles = self._m_input_ports[self._m_im_arr_in_tag].get_attribute("NEW_PARA")
        delta_para = para_angles[0] - para_angles
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

        for i in range(0, res_rot_mean_clip.shape[0]):
            for j in range(0, res_rot_mean_clip.shape[1]):
                temp = res_rot[:, i, j]
                if temp.var() > 0.0:
                    a = temp - temp.mean()
                    b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
                    b2 = b1.compress((b1 > (-1.0)*3.0*np.sqrt(a.var())).flat)
                    res_rot_mean_clip[i, j] = temp.mean() + b2.mean()

        # save results
        self._m_output_ports[self._m_res_arr_out_tag].set_all(res_arr)
        self._m_output_ports[self._m_res_arr_rot_out_tag].set_all(res_rot)
        self._m_output_ports[self._m_res_mean_tag].set_all(tmp_res_rot_mean)
        self._m_output_ports[self._m_res_median_tag].set_all(tmp_res_rot_median)
        self._m_output_ports[self._m_res_var_tag].set_all(tmp_res_rot_var)
        self._m_output_ports[self._m_res_rot_mean_clip_tag].set_all(res_rot_mean_clip)


class MakePSFModelModule(ProcessingModule):
    """
    should be just a part of the whole processing
    """

    def __init__(self,
                 num,
                 name_in = "psf_model_module",
                 im_arr_in_tag="im_arr",
                 basis_in_tag="basis_im",
                 basis_average_in_tag="basis_ave",
                 psf_basis_out_tag="psf_basis"):

        # TODO find out what num is
        self.m_num = num

        super(MakePSFModelModule, self).__init__(name_in)

        self._m_im_arr_in_tag = im_arr_in_tag
        self.add_input_port(im_arr_in_tag)

        self._m_basis_average_in_tag = basis_average_in_tag
        self.add_input_port(basis_average_in_tag)

        self._m_basis_in_tag = basis_in_tag
        self.add_input_port(basis_in_tag)

        self._m_psf_basis_out_tag = psf_basis_out_tag
        self.add_output_port(psf_basis_out_tag)

    def run(self):

        im_data = self._m_input_ports[self._m_im_arr_in_tag].get_all()
        basis_data = self._m_input_ports[self._m_basis_in_tag].get_all()
        basis_average = self._m_input_ports[self._m_basis_average_in_tag].get_all()

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

        self._m_output_ports[self._m_psf_basis_out_tag].set_all(result,
                                                                keep_attributes=True)

        self._m_output_ports[self._m_psf_basis_out_tag].add_attribute(name="psf_coeff",
                                                                      value=psf_coeff,
                                                                      static=False)


class MakePCABasisModule(ProcessingModule):
    """
    should be just a part of the whole processing
    """

    def __init__(self,
                 name_in,
                 im_arr_in_tag="im_arr",
                 im_arr_out_tag="im_arr",
                 im_average_out_tag="im_ave",
                 basis_out_tag="basis"):

        super(MakePCABasisModule, self).__init__(name_in)

        self._m_im_arr_in_tag = im_arr_in_tag
        self.add_input_port(im_arr_in_tag)

        self._m_im_arr_out_tag = im_arr_out_tag
        self.add_output_port(im_arr_out_tag)

        self._m_im_average_out_tag = im_average_out_tag
        self.add_output_port(im_average_out_tag)

        self._m_basis_out_tag = basis_out_tag
        self.add_output_port(basis_out_tag)

    @staticmethod
    def _make_average_sub(im_arr_in):
        im_ave = im_arr_in.mean(axis=0)

        for i in range(0, len(im_arr_in[:,0,0])):
            im_arr_in[i,] -= im_ave
        return im_arr_in, im_ave

    def run(self):

        im_data = self._m_input_ports[self._m_im_arr_in_tag].get_all()

        num_entries = im_data.shape[0]
        im_size = [im_data.shape[1],
                   im_data.shape[2]]

        tmp_im_data, tmp_im_ave = self._make_average_sub(im_data)

        _,_,V = linalg.svd(tmp_im_data.reshape(num_entries,
                                               im_size[0]*im_size[1]),
                           full_matrices=False)

        basis_pca_arr = V.reshape(V.shape[0], im_size[0], im_size[1])

        self._m_output_ports[self._m_im_arr_out_tag].set_all(tmp_im_data, keep_attributes=True)
        self._m_output_ports[self._m_im_average_out_tag].set_all(tmp_im_ave)
        self._m_output_ports[self._m_basis_out_tag].set_all(basis_pca_arr)
        self._m_output_ports[self._m_basis_out_tag].add_attribute(name="basis_type",
                                                                  value="pca")