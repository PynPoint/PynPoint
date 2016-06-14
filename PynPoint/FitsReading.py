# external modules
import numpy as np
import os
from astropy.io import fits
import warnings

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
                 **kwargs):

        super(ReadFitsCubesDirectory, self).__init__(name_in,
                                                     input_dir)
        self.m_image_tag = image_tag
        self.add_output_port(image_tag,
                             True)

        # read NACO static and non static keys
        static_keys_file = open("./config/NACO_static_header_keys.txt", "r")
        self.m_static_keys = static_keys_file.read().split(",\n")
        static_keys_file.close()

        non_static_keys_file = open("./config/NACO_non_static_header_keys.txt", "r")
        self.m_non_static_keys = non_static_keys_file.read().split(",\n")
        non_static_keys_file.close()

        # add additional keys
        if 'new_static' in kwargs:
            assert (os.path.isfile(kwargs['new_static'])), 'Error: Input file for static header keywords not found' \
                                                           ' - input requested: %s' % kwargs['new_static']
            static_keys_file = open(kwargs['new_static'], "r")
            self.m_static_keys.extend(static_keys_file.read().split(",\n"))
            static_keys_file.close()

        if 'new_non_static' in kwargs:
            assert (os.path.isfile(kwargs['new_non_static'])), 'Error: Input file for non static header keywords not ' \
                                                               'found - input requested: %s' % kwargs['new_non_static']
            non_static_keys_file = open(kwargs['new_non_static'], "r")
            self.m_non_static_keys.extend(non_static_keys_file.read().split(",\n"))
            non_static_keys_file.close()

        # create port for each non-static key
        for key in self.m_non_static_keys:
            self.add_output_port(key)

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
            print "Reading " + str(fits_file)

            hdulist = fits.open(tmp_location + fits_file)
            self._m_out_ports[self.m_image_tag].append(hdulist[0].data.byteswap().newbyteorder())

            # store header info
            tmp_header = hdulist[0].header

            # store static header information (e.g. Instrument name) as attributes
            # and non static using Ports
            # Use the NACO / VLT specific header tags
            for key in tmp_header:
                if key in self.m_static_keys:
                    check = self._m_out_ports[self.m_image_tag].check_attribute(key, tmp_header[key])
                    if check == -1:
                        warnings.warn('Static keyword %s has changed. Probably the current file %s does not '
                                      'belong to the data set %s of the PynPoint database. '
                                      'Updating Keyword...' % (key, fits_file, self.m_image_tag))
                    elif check == 0:
                        # Attribute is known and is still the same
                        pass
                    else:
                        # Attribute is new -> add
                        self._m_out_ports[self.m_image_tag].add_attribute(key, tmp_header[key])

                elif key in self.m_non_static_keys:
                    # use Port
                    self._m_out_ports[key].append(np.asarray([tmp_header[key],]))

            hdulist.close()
        self._m_out_ports[self.m_image_tag].close_port()
