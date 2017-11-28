import numpy as np
import cv2
import math

from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from skimage.transform import rescale

from PynPoint.core.Processing import ProcessingModule


class StarExtractionModule(ProcessingModule):

    def __init__(self,
                 name_in="star_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut",
                 pos_out_tag="star_positions",
                 psf_size=3,
                 psf_size_as_pixel_resolution=False,
                 num_images_in_memory=100,
                 fwhm_star=7):

        super(StarExtractionModule, self).__init__(name_in)

        # Ports

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_pos_out_port = self.add_output_port(pos_out_tag)

        self.m_psf_size = psf_size
        self.m_psf_size_as_pixel_resolution = psf_size_as_pixel_resolution
        self.m_num_images_in_memory = num_images_in_memory
        self.m_fwhm_star = fwhm_star # needed for the best gaussian blur 7 is good for L-band data
        self.count = 0

    def run(self):

        if self.m_psf_size_as_pixel_resolution:
            psf_radius = np.floor(self.m_psf_size / 2.0)
        else:
            pixel_scale = self.m_image_in_port.get_attribute('ESO INS PIXSCALE')
            psf_radius = np.floor((self.m_psf_size / 2.0) / pixel_scale)

        star_positions = []

        def cut_psf(current_image):

            sigma = self.m_fwhm_star/math.sqrt(8.*math.log(2.))
            kernel_size = (self.m_fwhm_star*2 + 1, self.m_fwhm_star*2 + 1)

            search_image = cv2.GaussianBlur(current_image,
                                            kernel_size,
                                            sigma)

            # cut the image by maximum
            argmax = np.unravel_index(search_image.argmax(), search_image.shape)

            if argmax[0] <= psf_radius or argmax[1] <= psf_radius \
                    or argmax[0] + psf_radius > current_image.shape[0] \
                    or argmax[1] + psf_radius > current_image.shape[1]:

                raise ValueError('Highest value is near the border. PSF size is too '
                                 'large to be cut (frame index = '+str(self.count)+').')

            # cut the image
            cut_image = current_image[int(argmax[0] - psf_radius):int(argmax[0] + psf_radius),
                                      int(argmax[1] - psf_radius):int(argmax[1] + psf_radius)]

            star_positions.append(argmax)
            
            self.count += 1

            return cut_image

        self.apply_function_to_images(cut_psf,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_pos_out_port.set_all(np.array(star_positions))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("PSF extract",
                                                      "Maximum search in gaussian burred input")
        self.m_image_out_port.close_port()


class StarAlignmentModule(ProcessingModule):

    def __init__(self,
                 name_in="star_align",
                 image_in_tag="im_arr",
                 ref_image_in_tag=None,
                 image_out_tag="im_arr_aligned",
                 interpolation="spline",
                 accuracy=10,
                 resize=1,
                 num_references=10,
                 num_images_in_memory=100):

        super(StarAlignmentModule, self).__init__(name_in)

        # Ports

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if ref_image_in_tag is not None:
            self.m_ref_image_in_port = self.add_input_port(ref_image_in_tag)
        else:
            self.m_ref_image_in_port = None

        # Parameter
        self.m_interpolation = interpolation
        self.m_accuracy = accuracy
        self.m_num_images_in_memory = num_images_in_memory
        self.m_resize = resize
        self.m_num_references = num_references

    def run(self):

        # get ref image
        if self.m_ref_image_in_port is not None:
            if len(self.m_ref_image_in_port.get_shape()) == 3:
                ref_images = np.asarray(self.m_ref_image_in_port.get_all(),
                                        dtype=np.float64)
            elif len(self.m_ref_image_in_port.get_shape()) == 2:
                ref_images = np.array([self.m_ref_image_in_port.get_all(),])
            else:
                raise ValueError("reference Image needs to be 2 D or 3 D.")
        else:
            ref_images = self.m_image_in_port[np.sort(
                np.random.choice(self.m_image_in_port.get_shape()[0],
                                 self.m_num_references,
                                 replace=False)), :, :]

        def align_image(image_in):

            offset = np.array([0.0, 0.0])
            for i in range(self.m_num_references):
                tmp_offset, _, _ = register_translation(ref_images[i, :, :],
                                                        image_in,
                                                        self.m_accuracy)
                offset += tmp_offset

            offset /= float(self.m_num_references)
            offset *= self.m_resize

            if self.m_resize is not 1:
                # the rescale function normalizes all values to [0 ... 1]. We want to keep the total
                # flux of the images and rescale the images afterwards
                sum_before = np.sum(image_in)
                tmp_image = rescale(image=np.asarray(image_in,
                                                     dtype=np.float64),
                                    scale=(self.m_resize,
                                           self.m_resize),
                                    order=5,
                                    mode="reflect")
                sum_after = np.sum(tmp_image)
                tmp_image = tmp_image*(sum_before/sum_after)
            else:
                tmp_image = image_in

            if self.m_interpolation == "spline":
                tmp_image = shift(tmp_image, offset, order=5)

            elif self.m_interpolation == "fft":
                tmp_image_spec = fourier_shift(np.fft.fftn(tmp_image), offset)
                tmp_image = np.fft.ifftn(tmp_image_spec)

            elif self.m_interpolation == "bilinear":
                tmp_image = shift(tmp_image, offset, order=1)

            else:
                raise ValueError("Interpolation needs to be spline, bilinear or fft")

            return tmp_image

        self.apply_function_to_images(align_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        # Change pix to mas scale corresponding to the reshaping
        tmp_pixscale = self.m_image_in_port.get_attribute("ESO INS PIXSCALE")

        tmp_pixscale /= self.m_resize
        self.m_image_out_port.add_attribute("ESO INS PIXSCALE", tmp_pixscale)

        history = "cross-correlation with up-sampling factor " + str(self.m_accuracy)
        self.m_image_out_port.add_history_information("PSF align",
                                                      history)
        self.m_image_out_port.close_port()
