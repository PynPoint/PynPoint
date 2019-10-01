"""
Module for reading FITS files obtained with VLT/VISIR for the NEAR experiment.
"""

import os
import math
import time
import shlex
import subprocess
import threading
import warnings

from typing import Union, Tuple

import numpy as np

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.attributes import set_static_attr, set_nonstatic_attr, set_extra_attr
from pynpoint.util.module import progress, memory_frames
from pynpoint.util.image import crop_image


class NearReadingModule(ReadingModule):
    """
    Pipeline module for reading VLT/VISIR data of the NEAR experiment. The FITS files and required
    header information are read from the input directory and stored in two datasets, corresponding
    to chop A and chop B. The primary HDU of the FITS files should contain the main header
    information, while the subsequent HDUs contain each a single image (alternated for chop A and
    chop B) and some additional header information for that image. The last HDU is ignored as it
    contains the average of all images.
    """

    __author__ = 'Jasper Jonker, Tomas Stolker, Anna Boehle'

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: str = None,
                 chopa_out_tag: str = 'chopa',
                 chopb_out_tag: str = 'chopb',
                 subtract: bool = False,
                 crop: Union[Tuple[int, int, float], Tuple[None, None, float]] = None,
                 combine: str = None):
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
            ``chop_b_out_tag``.
        chopb_out_tag : str
            Database entry where the chop B images will be stored. Should be different from
            ``chop_a_out_tag``.
        subtract : bool
            If True, the other chop position is subtracted before saving out the chop A and chop B
            images.
        crop: tuple(int, int, float), None
            The pixel position (x, y) around which the chop A and chop B images are cropped and
            the new image size (arcsec), together provided as (pos_x, pos_y, size). The same size
            will be used for both image dimensions. It is recommended to crop the images around
            the approximate coronagraph position. No cropping is applied if set to None.
        combine: str, None
            Method ('mean' or 'median') for combining (separately) the chop A and chop B frames
            from each cube into a single frame. All frames are stored if set to None.

        Returns
        -------
        NoneType
            None
        """

        super(NearReadingModule, self).__init__(name_in, input_dir)

        self.m_chopa_out_port = self.add_output_port(chopa_out_tag)
        self.m_chopb_out_port = self.add_output_port(chopb_out_tag)

        self.m_subtract = subtract
        self.m_crop = crop
        self.m_combine = combine

    @typechecked
    def _uncompress_file(self,
                         filename: str) -> None:
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

    @typechecked
    def uncompress(self) -> None:
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

    @typechecked
    def check_header(self,
                     header: fits.header.Header) -> None:
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
            warnings.warn('Dataset was obtained without chopping.')

        skipped = int(header['ESO DET CHOP CYCSKIP'])
        if skipped != 0:
            warnings.warn(f'Chop cycles ({skipped}) have been skipped.')

        if str(header['ESO DET CHOP CYCSUM']) == 'T':
            warnings.warn('FITS file contains averaged images.')

    @typechecked
    def read_header(self,
                    filename: str) -> Tuple[fits.header.Header, Tuple[int, int, int]]:
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
        if nimages % 2 != 0:
            warnings.warn(f'FITS file contains odd number of images: {filename}')

            # decreasing nimages to an even number such that nimages // 2 gives the correct size
            nimages -= 1

        # primary header
        header = hdulist[0].header

        # number of chop cycles
        ncycles = header['ESO DET CHOP NCYCLES']

        # number of chop cycles should be equal to half the number of available images
        if ncycles != nimages // 2:
            warnings.warn(f'The number of chop cycles ({ncycles}) is not equal to half the '
                          f'number of available HDU images ({nimages // 2}).')

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

    @typechecked
    def read_images(self,
                    filename: str,
                    im_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
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
                hdulist.close()
                raise ValueError(f'Frame type not found in the FITS header. Image number: {i}.')

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
                warnings.warn(f'Previous and current chop position ({cycle}) are the same. '
                              'Skipping the current image.')

            else:
                hdulist.close()
                raise ValueError(f'Frame type ({cycle}) not a valid value. Expecting HCYCLE1 or '
                                 'HCYCLE2 as value for ESO DET FRAM TYPE.')

        # check if the number of chop A and B images is equal, this error should never occur
        if count_chopa != count_chopb:
            warnings.warn('The number of images is not equal for chop A and chop B.')

        hdulist.close()

        return chopa, chopb

    @typechecked
    def run(self) -> None:
        """
        Run the module. The FITS files are collected from the input directory and uncompressed if
        needed. The images are then sorted by the two chop positions (chop A and chop B). The
        required FITS header keywords (which should be set in the configuration file) are also
        imported and stored as attributes to the two output datasets in the HDF5 database.

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
        assert(files), f'No FITS files found in {self.m_input_location}.'

        # if cropping chop A, get pixscale and convert crop_size to pixels and swap x/y
        if self.m_crop is not None:
            pixscale = self._m_config_port.get_attribute('PIXSCALE')
            self.m_crop = (self.m_crop[1], self.m_crop[0], int(math.ceil(self.m_crop[2]/pixscale)))

        start_time = time.time()
        for i, filename in enumerate(files):
            progress(i, len(files), 'Preprocessing NEAR data...', start_time)

            # get the primary header data and the image shape
            header, im_shape = self.read_header(filename)

            # get the images of chop A and chop B
            chopa, chopb = self.read_images(filename, im_shape)

            if self.m_subtract:
                chopa = chopa - chopb
                chopb = -1.*np.copy(chopa)

            if self.m_crop is not None:
                chopa = crop_image(chopa,
                                   center=self.m_crop[0:2],
                                   size=self.m_crop[2],
                                   copy=False)

                chopb = crop_image(chopb,
                                   center=self.m_crop[0:2],
                                   size=self.m_crop[2],
                                   copy=False)

            if self.m_combine is not None:

                if self.m_combine == 'mean':
                    chopa = np.mean(chopa, axis=0)
                    chopb = np.mean(chopb, axis=0)

                elif self.m_combine == 'median':
                    chopa = np.median(chopa, axis=0)
                    chopb = np.median(chopb, axis=0)

                header[self._m_config_port.get_attribute('NFRAMES')] = 1

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

        # add history information
        self.m_chopa_out_port.add_history('NearReadingModule', 'Chop A')
        self.m_chopb_out_port.add_history('NearReadingModule', 'Chop B')

        # close all connections to the database
        self.m_chopa_out_port.close_port()
