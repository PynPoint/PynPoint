"""
Modules for locating and aligning of the star.
"""

import numpy as np
import cv2

from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

from PynPoint.core.Processing import ProcessingModule


class StarExtractionModule(ProcessingModule):
    """
    Module to locate the position of the star in each image.
    """

    def __init__(self,
                 name_in="star_cutting",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut",
                 pos_out_tag="star_positions",
                 image_size=2.,
                 num_images_in_memory=100,
                 fwhm_star=0.2):
        """
        Constructor of StarExtractionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with the images that are written as
                              output. Should be different from *image_in_tag*.
        :type image_out_tag: str
        :param pos_out_tag: Tag of the database entry with the star positions that are written
                            as output.
        :type pos_out_tag: str
        :param image_size: Cropped size (arcsec) of the images.
        :type image_size: float
        :param fwhm_star: Full width at half maximum (arcsec) of the Gaussian kernel that is used
                          to convolve the images.
        :type fwhm_star: float

        :return: None
        """

        super(StarExtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_pos_out_port = self.add_output_port(pos_out_tag)
        self.m_image_size = image_size
        self.m_num_images_in_memory = num_images_in_memory
        self.m_fwhm_star = fwhm_star # 7 pix / 0.2 arcsec is good for L-band data

    def run(self):
        """
        Run method of the module. Locates the position of the star (only pixel precision) through
        the largest pixel value. A Gaussian kernel with a FWHM similar to the PSF is used to
        smooth away the contribution of bad pixels which may have higher values than the peak
        of the PSF. Images are cropped and written to an output port, as well as the position
        values of the star.

        :return: None
        """

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        pixel_scale = self.m_image_in_port.get_attribute("PIXSCALE")
        psf_radius = int((self.m_image_size / 2.0) / pixel_scale)

        self.m_fwhm_star /= pixel_scale
        self.m_fwhm_star = int(self.m_fwhm_star)

        star_positions = []

        def cut_psf(current_image):

            # see https://en.wikipedia.org/wiki/Full_width_at_half_maximum
            sigma = self.m_fwhm_star/2.335

            kernel_size = (self.m_fwhm_star*2 + 1, self.m_fwhm_star*2 + 1)

            search_image = cv2.GaussianBlur(current_image,
                                            kernel_size,
                                            sigma)

            argmax = np.unravel_index(search_image.argmax(), search_image.shape)

            if argmax[0] <= psf_radius or argmax[1] <= psf_radius \
                    or argmax[0] + psf_radius > current_image.shape[0] \
                    or argmax[1] + psf_radius > current_image.shape[1]:

                raise ValueError('Highest value is near the border. PSF size is too '
                                 'large to be cut')

            cut_image = current_image[int(argmax[0] - psf_radius):int(argmax[0] + psf_radius),
                                      int(argmax[1] - psf_radius):int(argmax[1] + psf_radius)]

            star_positions.append(argmax)

            return cut_image

        self.apply_function_to_images(cut_psf,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarExtractionModule...",
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_pos_out_port.set_all(np.array(star_positions))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("PSF extract",
                                                      "Maximum search in gaussian burred input")
        self.m_image_out_port.close_port()


class StarAlignmentModule(ProcessingModule):
    """
    Module to align the images with a cross-correlation in Fourier space.
    """

    def __init__(self,
                 name_in="star_align",
                 image_in_tag="im_arr",
                 ref_image_in_tag=None,
                 image_out_tag="im_arr_aligned",
                 interpolation="spline",
                 accuracy=10,
                 resize=1,
                 num_references=10):
        """
        Constructor of StarAlignmentModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with the stack of images that is read as
                             input.
        :type image_in_tag: str
        :param ref_image_in_tag: Tag of the database entry with the reference image(s) that are
                                 read as input.
        :type ref_image_in_tag: str
        :param image_out_tag: Tag of the database entry with the images that are written as
                              output.
        :type image_out_tag: str
        :param interpolation: Type of interpolation that is used for shifting the images (spline,
                              bilinear, or fft).
        :type interpolation: str
        :param accuracy: Upsampling factor for the cross-correlation. Images will be registered
                         to within 1/accuracy of a pixel.
        :type accuracy: float
        :param resize: Scaling factor for the up/down-sampling before the images are shifted.
        :type resize: float
        :param num_references: Number of reference images for the cross-correlation.
        :type num_references: int

        :return: None
        """

        super(StarAlignmentModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if ref_image_in_tag is not None:
            self.m_ref_image_in_port = self.add_input_port(ref_image_in_tag)
        else:
            self.m_ref_image_in_port = None

        self.m_interpolation = interpolation
        self.m_accuracy = accuracy
        self.m_resize = resize
        self.m_num_references = num_references

    def run(self):
        """
        Run method of the module. Applies a cross-correlation of the input images with respect to
        a stack of reference images, rescales the image dimensions, and shifts the images to a
        common center.

        :return: None
        """

        self.m_num_images_in_memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_ref_image_in_port is not None:
            if len(self.m_ref_image_in_port.get_shape()) == 3:
                ref_images = np.asarray(self.m_ref_image_in_port.get_all())
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
                                                        upsample_factor=self.m_accuracy)
                offset += tmp_offset

            offset /= float(self.m_num_references)
            offset *= self.m_resize

            if self.m_resize is not 1:
                sum_before = np.sum(image_in)
                tmp_image = rescale(image=np.asarray(image_in),
                                    scale=(self.m_resize,
                                           self.m_resize),
                                    order=5,
                                    mode="reflect")
                sum_after = np.sum(tmp_image)
                # Conserve flux because the rescale function normalizes all values to [0:1].
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
                                      "Running StarAlignmentModule...",
                                      num_images_in_memory=self.m_num_images_in_memory)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        tmp_pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        tmp_pixscale /= self.m_resize
        self.m_image_out_port.add_attribute("PIXSCALE", tmp_pixscale)

        history = "cross-correlation with up-sampling factor " + str(self.m_accuracy)
        self.m_image_out_port.add_history_information("PSF alignment",
                                                      history)
        self.m_image_out_port.close_port()
