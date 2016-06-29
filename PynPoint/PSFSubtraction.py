from scipy import linalg

from PynPoint.Processing import ProcessingModule


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