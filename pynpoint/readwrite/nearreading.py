"""
Module for reading FITS files obtained with VLT/VISIR for the NEAR experiment.
"""

import os
import sys
import time
import shlex
import subprocess
import threading
import warnings

import numpy as np

from astropy.io import fits

from pynpoint.core.processing import ReadingModule
from pynpoint.util.attributes import set_static_attr, set_nonstatic_attr, set_extra_attr
from pynpoint.util.module import progress, memory_frames


class NearReadingModule(ReadingModule):
    """
    Pipeline module for reading VLT/VISIR data of the NEAR experiment. The FITS files and required
    header information are read from the input directory and stored in four datasets, corresponding
    to nod A/B and chop A/B. The primary HDU of the FITS files should contain the main header
    information, while the subsequent HDUs contain each a single image (alternated for chop A and
    chop B) and some additional header information for that image. The last HDU is ignored as it
    contains the average of all images.
    """

    __author__ = 'Jasper Jonker, Tomas Stolker'

    def __init__(self,
                 name_in='burst',
                 input_dir=None,
                 chopa_out_tag='chopa',
                 chopb_out_tag='chopb'):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the instance.
        input_dir : str, None
            Input directory where the FITS files are located. The default input folder of the
            Pypeline is used if set to None.
        chopa_out_tag : str
            Database entry where the chop A images will be stored. Should be different from
            `chop_b_out_tag`.
        chopb_out_tag : str
            Database entry where the chop B images will be stored. Should be different from
            `chop_a_out_tag`.

        Returns
        -------
        NoneType
            None
        """

        super(NearReadingModule, self).__init__(name_in, input_dir)

        self.m_chopa_out_port = self.add_output_port(chopa_out_tag)
        self.m_chopb_out_port = self.add_output_port(chopb_out_tag)

    def _uncompress_file(self,
                         filename):
        """
        Internal function to uncompress a .Z file.

        Parameters
        ----------
        filename : str
            Compressed .Z file.

        Returns
        -------
        NoneType
            None
        """

        try:
            # try running a subprocess with the 'uncompress' command
            command = 'uncompress ' + filename
            subprocess.check_call(shlex.split(command))

        except(FileNotFoundError, OSError):
            # or else run a subprocess with the 'gunzip' command
            command = 'gunzip -d ' + filename
            subprocess.check_call(shlex.split(command))

    def uncompress(self):
        """
        Method to check if the input directory contains compressed files ending with .fits.Z.
        If this is the case, the files will be uncompressed using multithreading. The number
        of threads can be set with the ``CPU`` parameter in the configuration file.

        Returns
        -------
        NoneType
            None
        """

        cpu = self._m_config_port.get_attribute('CPU')

        # list all files ending with .fits.Z in the input location
        files = []
        for item in os.listdir(self.m_input_location):
            if item.endswith('.fits.Z'):
                files.append(os.path.join(self.m_input_location, item))

        if files:
            # subdivide the file indices by number of CPU
            indices = memory_frames(cpu, len(files))

            start_time = time.time()
            for i, _ in enumerate(indices[:-1]):
                progress(i, len(indices[:-1]), 'Uncompressing NEAR data...', start_time)

                # select subset of compressed files
                subset = files[indices[i]:indices[i+1]]

                # create a list of threads to uncompress CPU number of files
                # each file is processed by a different thread
                threads = []
                for filename in subset:
                    thread = threading.Thread(target=self._uncompress_file, args=(filename, ))
                    threads.append(thread)

                # start the threads
                for item in threads:
                    item.start()

                # join the threads
                for item in threads:
                    item.join()

            sys.stdout.write('Uncompressing NEAR data... [DONE]\n')
            sys.stdout.flush()

    def check_header(self,
                     header):
        """
        Method to check the header information and prompt a warning if a value is not as expected.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            Header information from the FITS file that is read.

        Returns
        -------
        NoneType
            None
        """

        if str(header['ESO DET CHOP ST']) == 'F':
            warnings.warn('Chopping has been set to disabled.')

        skipped = int(header['ESO DET CHOP CYCSKIP'])
        if skipped != 0:
            warnings.warn('{} chop cycles have been skipped during operation.'.format(skipped))

        if str(header['ESO DET CHOP CYCSUM']) == 'T':
            warnings.warn('Frames have been averaged by default. This module will probably not '
                          'work properly.')

    def read_header(self,
                    filename):
        """
        Function that opens a FITS file and separates the chop A and chop B images. The primary HDU
        contains only a general header. The subsequent HDUs contain a single image with a small
        extra header. The last HDU is the average of all images, which will be ignored.

        Parameters
        ----------
        filename : str
            Absolute path and filename of the FITS file.

        Returns
        -------
        astropy.io.fits.header.Header
            Primary header, which is valid for all images.
        tuple(int, int, int)
            Shape of a stack of images for chop A or B.
        """

        # open the FITS file
        hdulist = fits.open(filename)

        # number of images = total number of HDUs - primary HDU - last HDU (average image)
        nimages = len(hdulist) - 2

        # check if the file contains an even number of images, as expected with two chop positions
        if nimages%2 != 0:
            warnings.warn('FITS file contains odd number of images: {}'.format(filename))

            # decreasing nimages to an even number such that nimages // 2 gives the correct size
            nimages -= 1

        # primary header
        header = hdulist[0].header

        # header of the first image
        header_image = hdulist[1].header

        # create a list of key = value from the primary header
        fits_header = []
        for key in header:
            if key:
                fits_header.append(str(key)+' = '+str(header[key]))

        # write the primary header information to the fits_header group
        header_out_port = self.add_output_port('fits_header/' + filename)
        header_out_port.set_all(fits_header)

        # shape of the image stacks for chop A/B (hence nimages/2)
        im_shape = (nimages // 2, header_image['NAXIS2'], header_image['NAXIS1'])

        # set the NAXIS image shape in the primary header
        # required by util.attributes.set_nonstatic_attr
        header.set('NAXIS', 3)
        header.set('NAXIS1', im_shape[2])
        header.set('NAXIS2', im_shape[1])
        header.set('NAXIS3', im_shape[0])

        # check primary header
        self.check_header(header)

        hdulist.close()

        return header, im_shape

    def read_images(self,
                    filename,
                    im_shape):
        """
        Function that opens a FITS file and separates the chop A and chop B images. The primary HDU
        contains only a general header. The subsequent HDUs contain a single image with a small
        extra header. The last HDU is the average of all images, which will be ignored.

        Parameters
        ----------
        filename : str
            Absolute path and filename of the FITS file.
        im_shape : tuple(int, int, int)
            Shape of a stack of images for chop A or B.

        Returns
        -------
        numpy.array
            Array containing the images of chop A.
        numpy.array
            Array containing the images of chop B.
        """

        # open the FITS file
        hdulist = fits.open(filename)

        # initialize the image arrays for chop A and B
        chopa = np.zeros(im_shape, dtype=np.float32)
        chopb = np.zeros(im_shape, dtype=np.float32)

        count_chopa, count_chopb = 0, 0
        prev_cycle = None

        for i in range(2*im_shape[0]):
            # get the chop position (HCYCLE1 = chop A, HCYCLE2 = chop B)
            # primary HDU is skipped with +1
            if 'ESO DET FRAM TYPE' in hdulist[i+1].header:
                cycle = hdulist[i+1].header['ESO DET FRAM TYPE']

            else:
                cycle = None

                warnings.warn('The chop position (=ESO DET FRAM TYPE) is not available '
                              'in the FITS header. Image number: {}'.format(i))

            # write the HDU image to the chop A or B array
            # count the number of chop A and B images
            if cycle == 'HCYCLE1' and cycle != prev_cycle:
                chopa[count_chopa, ] = hdulist[i+1].data.byteswap().newbyteorder()

                count_chopa += 1
                prev_cycle = cycle

            elif cycle == 'HCYCLE2' and cycle != prev_cycle:
                chopb[count_chopb, ] = hdulist[i+1].data.byteswap().newbyteorder()

                count_chopb += 1
                prev_cycle = cycle

            elif cycle == prev_cycle:
                warnings.warn('Previous and current chop position ({}) are the same. Skipping the '
                              'current image.'.format(cycle))

            else:
                raise ValueError('Frame type ({}) not a valid value. Expecting HCYCLE1 or HCYCLE2 '
                                 'as value for ESO DET FRAM TYPE.'.format(cycle))

        # check if the number of chop A and B images is equal, this error should never occur
        if count_chopa != count_chopb:
            raise ValueError('The number of images is not equal for chop A and chop B.')

        hdulist.close()

        return chopa, chopb

    def run(self):
        """
        Run the module. The FITS files are collected from the input directory and uncompressed if
        needed. The images are then sorted by the two nod positions (nod A and nod B) and two chop
        positions (chop A and chop B) per nod position. The required FITS header keywords (which
        should be set in the configuration file) are also imported and stored as attributes to the
        four output datasets in the HDF5 database. The nodding position of each FITS files is
        determined relative to the exposure number (ESO TPL EXPNO) of the first FITS file that is
        read. Therefore, the first FITS file should correspond to the first position in the chosen
        nodding scheme.

        Returns
        -------
        NoneType
            None
        """

        # clear the output ports
        self.m_chopa_out_port.del_all_data()
        self.m_chopa_out_port.del_all_attributes()
        self.m_chopb_out_port.del_all_data()
        self.m_chopb_out_port.del_all_attributes()

        # uncompress the FITS files if needed
        self.uncompress()

        # find and sort the FITS files
        files = []

        for filename in os.listdir(self.m_input_location):
            if filename.endswith('.fits'):
                files.append(os.path.join(self.m_input_location, filename))

        files.sort()

        # check if there are FITS files present in the input location
        assert(files), 'No FITS files found in {}.'.format(self.m_input_location)

        start_time = time.time()
        for i, filename in enumerate(files):
            progress(i, len(files), 'Running NearReadingModule...', start_time)

            # get the images of chop A and B, the primary header data, the nod position,
            # and the number of images per chop position
            header, im_shape = self.read_header(filename)
            chopa, chopb = self.read_images(filename, im_shape)

            # append the images of chop A and B
            self.m_chopa_out_port.append(chopa, data_dim=3)
            self.m_chopb_out_port.append(chopb, data_dim=3)

            # starting value for the INDEX attribute
            first_index = 0

            for port in (self.m_chopa_out_port, self.m_chopb_out_port):

                # set the static attributes
                set_static_attr(fits_file=filename,
                                header=header,
                                config_port=self._m_config_port,
                                image_out_port=port,
                                check=True)

                # set the non-static attributes
                set_nonstatic_attr(header=header,
                                   config_port=self._m_config_port,
                                   image_out_port=port,
                                   check=True)

                # set the remaining attributes
                set_extra_attr(fits_file=filename,
                               nimages=im_shape[0]//2,
                               config_port=self._m_config_port,
                               image_out_port=port,
                               first_index=first_index)

                # increase the first value of the INDEX attribute
                first_index += im_shape[0]//2

                # flush the output port
                port.flush()

        sys.stdout.write('Running NearReadingModule... [DONE]\n')
        sys.stdout.flush()

        # add history information
        self.m_chopa_out_port.add_history('NearReadingModule', 'Chop A')
        self.m_chopb_out_port.add_history('NearReadingModule', 'Chop B')

        # close all connections to the database
        self.m_chopa_out_port.close_port()
