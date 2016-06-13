# external modules
import numpy as np
import os
from astropy.io import fits

# own classes
from Processing import ReadingModule


class ReadFitsCubesDirectory(ReadingModule):
    """
    Reads .fits files from the given input_dir or the default directory of the Pypeline.
    The .fits files need to be in cube mode containing multiple images per .fits file.
    For the calculation of the paralactic angle the tags "" and "" need to be in the header of the files.
    (THIS CLASS IS MADE FOR NACO / VLT ADI DATA)
    """

    def __init__(self,
                 name_in=None,
                 input_dir=None,
                 image_tag="im_arr",
                 calc_para_angle=True,
                 para_angle_tag="new_para"):

        super(ReadFitsCubesDirectory, self).__init__(name_in,
                                                     input_dir)
        self.m_image_tag = image_tag
        self.add_output_port(image_tag,
                             True)

        self.m_para_angle_tag = para_angle_tag
        self.add_output_port(para_angle_tag,
                             calc_para_angle)

    def run(self):

        if not self.m_input_location.endswith('/'):
            tmp_location = self.m_input_location +'/'
        else:
            tmp_location = self.m_input_location

        # search for fits files
        files = []
        for tmp_file in os.listdir(tmp_location):
            if tmp_file.endswith('.fits'):
                files.append(tmp_file)
        files.sort()

        assert(len(files) > 0), 'Error no .fits files found in %s' % self.m_input_location

        # read file and append data to storage
        for fits_file in files:
            print fits_file
            print tmp_location
            hdulist = fits.open(tmp_location + fits_file)

            print hdulist[0].data.byteswap().newbyteorder().shape
            self._m_out_ports[self.m_image_tag].append(hdulist[0].data.byteswap().newbyteorder())

            '''# del not a valid fits args
            tmp_header = hdulist[0].header
            if 'ESO DET CHIP PXSPACE' in tmp_header:
                del tmp_header['ESO DET CHIP PXSPACE']
            datacube.m_header = [tmp_header for i in range(len(datacube.m_images))]
            datacube.m_flags = [False for i in range(len(datacube.m_images))]'''

            hdulist.close()
        self._m_out_ports[self.m_image_tag].close_port()
