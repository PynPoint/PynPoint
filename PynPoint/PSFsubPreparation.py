from __future__ import division

import numpy as np
from scipy import ndimage


from PynPoint.Processing import ProcessingModule


class PSFdataPreparation(ProcessingModule):

    def __init__(self,
                 name_in=None,
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 image_mask_out_tag="im_mask_arr",
                 mask_out_tag="mask_arr",
                 resize=False,
                 cent_remove=True,
                 F_final=2.0,
                 ran_sub=None,
                 para_sort=True,
                 cent_size=0.05,
                 edge_size=1.0):

        super(PSFdataPreparation, self).__init__(name_in)

        # Note recentering is not longer supported
        self.m_resize = resize
        self.m_cent_remove = cent_remove
        self.m_F_final = F_final
        self.m_ran_sub = ran_sub
        self.m_para_sort = para_sort
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size

        self.m_image_in_tag = image_in_tag
        self.m_image_out_tag = image_out_tag
        self.m_image_mask_out_tag = image_mask_out_tag
        self.m_mask_out_tag = mask_out_tag
        self.m_norm_port = "header_" + image_in_tag + "/im_norm"

        # create Ports
        self.add_input_port(image_in_tag)
        self.add_output_port(image_mask_out_tag)
        self.add_output_port(mask_out_tag)
        self.add_output_port(image_out_tag)
        self.add_output_port("header_" + image_in_tag + "/im_norm")

    def run(self):

        # TODO para_sort

        # image normalization
        im_data = self._m_input_ports[self.m_image_in_tag].get_all()

        print im_data[0,0,0]

        im_norm = (im_data.sum(axis = 1)).sum(axis = 1)
        print(im_norm)

        for i in range(0,len(im_data[:, 0, 0])):
            im_data[i, ] /= im_norm[i]

        # TODO del this line ?
        self._m_output_ports[self.m_norm_port].set_all(im_norm)
        self._m_output_ports[self.m_image_out_tag][:] = im_data

        # image resizing
        if self.m_resize:
            xnum_final,  ynum_final = int(im_data.shape[1] * self.m_F_final),\
                                      int(im_data.shape[2] * self.m_F_final)
            im_arr_res = np.zeros([im_data.shape[0], xnum_final, ynum_final])

            for i in range(0, im_data.shape[0]):
                im_tmp = im_data[i]
                im_tmp = ndimage.interpolation.zoom(im_tmp,
                                                    [self.m_F_final,
                                                     self.m_F_final],
                                                    order=3)  # > 5 TODO
                im_arr_res[i,] = im_tmp

            self._m_output_ports[self.m_image_out_tag].set_all(im_arr_res,
                                                               keep_attributes = True)

        # image masking

        def mk_circle_func(center_x,center_y):
            """sets up a function for calculating the radius to x,y (after having been initialised
            with x_cent and y_cent) """
            return lambda x,y:np.sqrt((center_x-x)**2 +(center_y-y)**2)

        def mk_circle(xnum,ynum,xcent,ycent,rad_lim):
            """function for making a circular aperture"""
            Y,X = np.indices([xnum,ynum]) #seems strange and backwards, check!
            rad = mk_circle_func(xcent,ycent)(X,Y)
            i,j = np.where(rad <= rad_lim)
            mask_base = np.ones((xnum,ynum),float) #something strange about the order of x and y!
            mask_base[i,j] = 0.0
            return mask_base

        im_data = self._m_input_ports[self.m_image_in_tag].get_all()
        im_size = im_data[0, ].shape

        if self.m_cent_remove:


            mask_c = mk_circle(im_size[0],
                               im_size[1],
                               im_size[0]/2.,
                               im_size[1]/2.,
                               self.m_cent_size * im_size[0])

            mask_outside = mk_circle(im_size[0],
                                     im_size[1],
                                     im_size[0]/2.,
                                     im_size[1]/2.,
                                     0.5 * im_size[0])

            cent_mask = mask_c * (1.0 - mask_outside)
            res_cent_mask = (1.0 - cent_mask)
            im_arr_imask = im_data * res_cent_mask
            self._m_output_ports[self.m_image_mask_out_tag].set_all(im_arr_imask)

            im_arr_omask = im_data * cent_mask
            self._m_output_ports[self.m_image_out_tag].set_all(im_arr_omask,
                                                               keep_attributes=True)


        else:
            cent_mask = np.ones(im_size)

        self._m_output_ports[self.m_mask_out_tag].set_all(cent_mask)




