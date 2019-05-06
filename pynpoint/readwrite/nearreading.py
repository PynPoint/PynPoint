"""
Module for reading FITS files obtained with VLT/VISIR for the NEAR experiment.
"""

import os
import sys
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

    def __init__(self,
                 name_in='burst',
                 input_dir=None,
                 noda_chopa_tag='noda_chopa',
                 noda_chopb_tag='noda_chopb',
                 nodb_chopa_tag='nodb_chopa',
                 nodb_chopb_tag='nodb_chopb',
                 scheme='ABBA'):
        """
        Constructor of the NearReadingModule.

        Parameters
        ----------
        name_in : str
            Unique name of the instance.
        input_dir : str, None
            Input directory where the FITS files are located. The default input folder of the
            Pypeline is used if set to None.
        noda_chopa_tag : str
            Database entry where chop A from the nod A data will be stored.
        noda_chopb_tag : str
            Database entry where chop B from the nod A data will be stored.
        nodb_chopa_tag : str
            Database entry where chop A from the nod B data will be stored.
        nodb_chopb_tag : str
            Database entry where chop B from the nod B data will be stored.
        scheme : str
            Nodding scheme ('ABBA' or 'ABAB').

        Returns
        -------
        NoneType
            None
        """

        super(NearReadingModule, self).__init__(name_in, input_dir)

        out_tags = (noda_chopa_tag, noda_chopb_tag, nodb_chopa_tag, nodb_chopb_tag)

        # select all unique output tag names
        seen = set()
        unique_tags = [tag for tag in out_tags if tag not in seen and not seen.add(tag)]

        if len(unique_tags) != 4:
            raise ValueError('Output ports should have different name tags.')

        # create the 4 output ports for nod A/B and chop A/B
        self.m_image_out_port = []
        for tag in out_tags:
            self.m_image_out_port.append(self.add_output_port(tag))

        self.m_scheme = scheme

        if self.m_scheme != 'ABBA' and self.m_scheme != 'ABAB':
            raise ValueError('Nodding scheme argument should be set to \'ABBA\' or \'ABAB\'.')

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

            for i, _ in enumerate(indices[:-1]):
                progress(i, len(indices[:-1]), 'Uncompressing NEAR data...')

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

    def read_fits(self,
                  filename,
                  first_expno):
        """
        Function that opens a FITS file and separates the chop A and chop B images. The primary HDU
        contains only a general header. The subsequent HDUs contain a single image with a small
        extra header. The last HDU is the average of all images, which will be ignored.

        Parameters
        ----------
        filename : str
            FITS filename.
        first_expno : int
            First exposure number. Should be the first position of the nodding scheme.

        Returns
        -------
        numpy.array
            Array containing the images of chop A.
        numpy.array
            Array containing the images of chop B.
        astropy.io.fits.header.Header
            Primary header, which is valid for all images.
        str
            Nod position ('A' or 'B').
        int
            Number of images per chop position.
        """

        # open the FITS file
        hdulist = fits.open(os.path.join(self.m_input_location, filename))

        # number of images = total number of HDUs - primary HDU - last HDU (average image)
        nimages = len(hdulist) - 2

        # check if the file contains an even number of images, as expected with two chop positions
        if nimages%2 != 0:
            warnings.warn('FITS file contains odd number of images: {}'.format(filename))

            # increasing nimages to an even number such that nimages // 2 gives the correct size
            nimages += 1

        # primary header
        header = hdulist[0].header

        # get the exposure number
        expno = header['ESO TPL EXPNO']

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

        # determine the nod position (A or B) for the selected nodding scheme
        # relative to the exposure number of the first FITS file that is read
        if self.m_scheme == 'ABBA':
            if (expno-first_expno)%4 == 0 or (expno-first_expno)%4 == 3:
                nod = 'A'
            elif (expno-first_expno)%4 == 1 or (expno-first_expno)%4 == 2:
                nod = 'B'

        elif self.m_scheme == 'ABAB':
            if (expno-first_expno)%2 == 0:
                nod = 'A'
            elif (expno-first_expno)%2 == 1:
                nod = 'B'

        # initialize the image arrays for chop A and B
        chopa = np.zeros(im_shape, dtype=np.float32)
        chopb = np.zeros(im_shape, dtype=np.float32)

        count_chopa, count_chopb = 0, 0
        prev_cycle = None

        for i in range(nimages):
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

        return chopa, chopb, header, nod, nimages//2

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
        for port in self.m_image_out_port:
            port.del_all_data()
            port.del_all_attributes()

        # uncompress the FITS files if needed
        self.uncompress()

        # find and sort the FITS files
        files = []

        for filename in os.listdir(self.m_input_location):
            if filename.endswith('.fits'):
                files.append(filename)

        files.sort()

        # check if there are FITS files present in the input location
        assert(files), 'No FITS files found in {}.'.format(self.m_input_location)

        # get the first exposure number, which should be the first position of the nodding scheme
        header = fits.getheader(os.path.join(self.m_input_location, files[0]), ext=0)
        first_expno = header['ESO TPL EXPNO']

        for i, filename in enumerate(files):
            progress(i, len(files), 'Running NearReadingModule...')

            # get the images of chop A and B, the primary header data, the nod position,
            # and the number of images per chop position
            chopa, chopb, header, nod, nimages = self.read_fits(filename, first_expno)

            # select the output ports for nod A or B
            if nod == 'A':
                out_ports = self.m_image_out_port[0:2]
            elif nod == 'B':
                out_ports = self.m_image_out_port[2:4]

            # append the images of chop A and B
            out_ports[0].append(chopa, data_dim=3)
            out_ports[1].append(chopb, data_dim=3)

            # starting value for the INDEX attribute
            first_index = 0

            for port in out_ports:

                # set the static attributes
                set_static_attr(fits_file=filename,
                                header=header,
                                config_port=self._m_config_port,
                                image_out_port=port)

                # set the non-static attributes
                set_nonstatic_attr(header=header,
                                   config_port=self._m_config_port,
                                   image_out_port=port)

                # set the remaining attributes
                set_extra_attr(fits_file=filename,
                               location=self.m_input_location,
                               nimages=nimages,
                               config_port=self._m_config_port,
                               image_out_port=port,
                               first_index=first_index)

                # increase the first value of the INDEX attribute
                first_index += nimages

                # flush the output port
                port.flush()

        sys.stdout.write('Running NearReadingModule... [DONE]\n')
        sys.stdout.flush()

        # add history information
        self.m_image_out_port[0].add_history('NearReadingModule', 'Nod A, Chop A')
        self.m_image_out_port[1].add_history('NearReadingModule', 'Nod A, Chop B')
        self.m_image_out_port[2].add_history('NearReadingModule', 'Nod B, Chop A')
        self.m_image_out_port[3].add_history('NearReadingModule', 'Nod B, Chop B')

        # close all connections to the database
        self.m_image_out_port[0].close_port()
