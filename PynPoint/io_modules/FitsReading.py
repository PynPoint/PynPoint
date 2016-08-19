"""
Module for reading different kinds of .fits data
"""
# external modules
import os
import warnings

from astropy.io import fits

from PynPoint.core.Processing import ReadingModule

warnings.simplefilter("always")


class ReadFitsCubesDirectory(ReadingModule):
    """
    **THIS CLASS WAS MADE FOR NACO / VLT ADI DATA. USE CAREFULLY WITH DIFFERENT DATA**

    Reads .fits files from the given input_dir or the default directory of the Pypeline.
    The .fits files need to contain 2D images with same shape and type. The header of the .fits
    files is scanned for known static attributes (do not change from .fits file to .fits file) and
    non static attributes (e.g. the air mass ...). Known static and non static header entries are
    stored in the files:

        * config/NACO_non_static_header_keys.txt
        * config/NACO_static_header_keys.txt

    Static entries will be saved as .hdf5 attributes while changing entries will be saved as own
    data sets in a sub folder named *header_* + image_tag.
    (see :class:`PynPoint.io_modules.FitsReading.ReadFitsCubesDirectory`).
    Not known header entries will cause warning. It is possible to extend the known keywords by
    loading own static_header_keys and non_static_header_keys files. See
    :class:`PynPoint.io_modules.FitsReading.ReadFitsCubesDirectory` for more information.

    If the .fits files in the input directory have changing static attributes or the shape of the
    input images is changing a warning is caused.

    **Note** per default this module will overwrite all existing data with the same tags in the
    central Pynpoint database.
    """

    def __init__(self,
                 name_in=None,
                 input_dir=None,
                 image_tag="im_arr",
                 force_overwrite_in_databank=True,
                 **kwargs):
        """
        Constructor of a ReadFitsCubesDirectory instance. As all Reading Modules it has a name and
        input directory (:class:`PynPoint.core.Processing.ReadingModule`). In addition a image_tag
        can be chosen which will be the tag / key of the read data in the .hdf5 database.

        :param name_in: Name of the Module
        :type name_in: str
        :param input_dir: Input directory where the .fits files are located. If not specified the
                          Pypeline default directory is used. Needs to be a directory, if not raises
                          errors.
        :type input_dir: str
        :param image_tag: Tag of the read data in the .hdf5 data base. Non static header
                          information is stored with the tag: *header_* + image_tag /
                          #header_entry_name#
        :type image_tag: str
        :param force_overwrite_in_databank: If True existing data (header and the actual data set)
                                            in the .hdf5 data base will be overwritten.
        :type force_overwrite_in_databank: bool
        :param kwargs:
            **new_static:** Location of a file defining new static header entries used in the .fits
            file which is useful for non NACO / VLT data. The file needs to be formatted like
            this: ::

                <<KEY WORD>>,
                <<KEY WORD>>,
                <<KEY WORD>>

            while <<KEY WORD>> needs to be replaced with the header entry keyword.

            **new_non_static:** Location of a file defining new non static header entries used in
            the .fits file which is useful for non NACO / VLT data. The file needs to be formatted
            the same way as the new_static header entries.
        :return: None
        """

        super(ReadFitsCubesDirectory, self).__init__(name_in,
                                                     input_dir)

        self.m_image_out_port = self.add_output_port(image_tag,
                                                     True)
        self.m_image_tag = image_tag

        self._m_overwrite = force_overwrite_in_databank

        # get location of the Reading module
        script_dir = os.path.dirname(__file__)

        # read NACO static and non static keys
        static_keys_file = open(script_dir + "/config/NACO_static_header_keys.txt", "r")
        self.m_static_keys = static_keys_file.read().split(",\n")
        static_keys_file.close()

        non_static_keys_file = open(script_dir + "/config/NACO_non_static_header_keys.txt", "r")
        self.m_non_static_keys = non_static_keys_file.read().split(",\n")
        non_static_keys_file.close()

        # add additional keys
        if 'new_static' in kwargs:
            assert (os.path.isfile(kwargs['new_static'])), 'Error: Input file for static header ' \
                                                           'keywords not found - input requested:' \
                                                           ' %s' % kwargs['new_static']

            static_keys_file = open(kwargs['new_static'], "r")
            self.m_static_keys.extend(static_keys_file.read().split(",\n"))
            static_keys_file.close()

        if 'new_non_static' in kwargs:
            assert (os.path.isfile(kwargs['new_non_static'])), 'Error: Input file for non static ' \
                                                               'header keywords not found - input' \
                                                               ' requested: %s' \
                                                               % kwargs['new_non_static']

            non_static_keys_file = open(kwargs['new_non_static'], "r")
            self.m_non_static_keys.extend(non_static_keys_file.read().split(",\n"))
            non_static_keys_file.close()

    def _read_single_file(self,
                          fits_file,
                          tmp_location,
                          overwrite_keys):
        """
        Internal function which reads a single .fits file and appends it to the database. The
        function gets a list of overwriting_keys. If a new key (header entry or image data) is found
        that is not on this list the old entry is overwritten if self._m_overwrite is active. After
        replacing the old entry the key is added to the overwriting_keys. This procedure guaranties
        that all old data base information, that does not belong to the new data set that is read by
        ReadFitsCubesDirectory is replaced and the rest is kept.

        Note the function prints the name of the file it is currently reading

        :param fits_file: Name of the .fits file
        :type fits_file: String
        :param tmp_location: Directory where the .fits file is located.
        :type tmp_location: String
        :param overwrite_keys: The list of keys that will not be overwritten by the function
        :return: None
        """

        print "Reading " + str(fits_file)

        hdulist = fits.open(tmp_location + fits_file)

        if self._m_overwrite and self.m_image_tag not in overwrite_keys:
            # rest image array and all attached attributes
            self.m_image_out_port.set_all(hdulist[0].data.byteswap().newbyteorder(),
                                          data_dim=3)

            self.m_image_out_port.del_all_attributes()
            overwrite_keys.append(self.m_image_tag)
        else:
            self.m_image_out_port.append(hdulist[0].data.byteswap().newbyteorder(),
                                         data_dim=3)

        # store header info
        tmp_header = hdulist[0].header

        # store static header information (e.g. Instrument name) as attributes
        # and non static using Ports
        # Use the NACO / VLT specific header tags
        for key in tmp_header:
            if key in self.m_static_keys:
                check = self.m_image_out_port.check_static_attribute(key,
                                                                     tmp_header[key])
                if check == -1:
                    warnings.warn('Static keyword %s has changed. Probably the current '
                                  'file %s does not belong to the data set "%s" of the PynPoint'
                                  ' database. Updating Keyword...' \
                                  % (key, fits_file, self.m_image_tag))
                elif check == 0:
                    # Attribute is known and is still the same
                    pass
                else:
                    # Attribute is new -> add
                    self.m_image_out_port.add_attribute(key,
                                                        tmp_header[key],
                                                        static=True)

            elif key in self.m_non_static_keys:
                self.m_image_out_port.append_attribute_data(key,
                                                            tmp_header[key])

            elif key is not "":
                warnings.warn('Unknown Header "%s" key found' %str(key))

        hdulist.close()

        # append used file to header information
        self.m_image_out_port.append_attribute_data("Used_Files",
                                                    tmp_location + fits_file)
        self.m_image_out_port.flush()

    def run(self):
        """
        Run method of the module. It looks for all .fits files in the input directory and reads them
        using the internal function _read_single_file().
        Note if self._m_overwrite = force_overwrite_in_databank is True old data base information is
        overwritten by the module. Furthermore the module saves the number of files and the actual
        files used to create the database entry as attributes.

        :return: None
        """

        if not self.m_input_location.endswith('/'):
            tmp_location = self.m_input_location + '/'
        else:
            tmp_location = self.m_input_location

        # search for fits files
        files = []
        for tmp_file in os.listdir(tmp_location):
            if tmp_file.endswith('.fits'):
                files.append(tmp_file)
        files.sort()

        assert(len(files) > 0), 'Error no .fits files found in %s' % self.m_input_location

        # overwrite_keys save the database keys which were updated. Used for overwriting only
        overwrite_keys = []

        # read file and append data to storage
        for fits_file in files:
            self._read_single_file(fits_file,
                                   tmp_location,
                                   overwrite_keys)
            self.m_image_out_port.flush()
        self.m_image_out_port.close_port()

        # Update number of files used to create the data base entry
        new_files_num = len(files)
        if self._m_overwrite:
            # no old files have been used
            self.m_image_out_port.open_port()
            self.m_image_out_port.add_attribute("Num_Files", new_files_num)
        else:
            # appended data
            self.m_image_out_port.add_value_to_static_attribute("Num_Files",
                                                                new_files_num)
