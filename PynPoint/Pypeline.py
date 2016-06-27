"""
Module for the capsuling of different pipeline processing steps.
Main components:
    -> class Pypeline
"""
# external modules
import os
import collections
import warnings
import numpy as np

# own modules
from PynPoint.Processing import PypelineModule, WritingModule, ReadingModule
from PynPoint.DataIO import DataStorage


class Pypeline(object):
    """
    Pipeline class used for the management of various processing steps. The Pipeline has a central
    data storage on the hard drive which can be accessed by the different Modules using Ports.
    Furthermore a Pypeline instance inheres a internal dictionary of Pipeline steps (modules) and
    their names. This dictionary is ordered based on the order the steps have been added to the
    pypeline. It is possible to run the whole Pypeline (i.e. all modules / steps) or a single
    Modules by name.
    """

    def __init__(self,
                 working_place_in=None,
                 input_place_in=None,
                 output_place_in=None):
        """
        Constructor of a Pypeline object.

        :param working_place_in: Working location of the Pypeline which needs to be a folder on the
            hard drive. The given folder will be used to save the PynPoint data base as .hdf5.
            NOTE: Depending on the input this .hdf5 file can become to a very large file.
        :type working_place_in: String
        :param input_place_in: Default input directory of the Pypeline. All Reading Modules added to
         the Pypeline will use this directory to look for input data. It is possible to specify
            different locations for different Reading Modules using their constructors.
        :type input_place_in: String
        :param output_place_in: Default result directory used to save the output of all Writing
            Modules added to the Pypeline. It is possible to specify different locations for
            different Writing Modules using their constructors.
        :return: None
        """

        self._m_working_place = working_place_in
        self._m_input_place = input_place_in
        self._m_output_place = output_place_in
        self._m_modules = collections.OrderedDict()
        self.m_data_storage = DataStorage(working_place_in + '/PynPoint_database.hdf5')

    def __setattr__(self, key, value):
        """
        This method is called every time a member / attribute of the Pypeline is changed.
        It checks whether a chosen working / input / output directory exists.

        :param key: member / attribute name
        :param value: new value for the given attribute / member
        :return: None
        """

        # Error case directory does not exist
        if key in ["_m_working_place", "_m_input_place", "_m_output_place"]:
            assert (os.path.isdir(str(value))), 'Error: Input directory for ' + str(key) + \
                                                ' does not exist - input ' \
                                                'requested: %s' % value

        super(Pypeline, self).__setattr__(key, value)  # use the method of object

    def add_module(self,
                   pipeline_module):
        """
        Adds a given pipeline module to the internal Pypeline dictionary. The module is appended at
        the end of the ordered dictionary. If the input module is a reading or writing module
        without a specified input / output location the Pypeline default location is used. Moreover
        the input module is connected to the Pypeline internal data base.

        :param pipeline_module: The input module. Needs to be either a Processing, Reading or
            Writing Module.
        :type pipeline_module: Processing Module, Reading Module, Writing Module
        :return: None
        """
        assert isinstance(pipeline_module, PypelineModule), 'Error: the given pipeline_module is '\
                                                            'not a accepted Pypeline Module.'

        # if no specific output directory is given use the default
        if isinstance(pipeline_module, WritingModule):
            if pipeline_module.m_output_location is None:
                pipeline_module.m_output_location = self._m_output_place

        # if no specific input directory is given use the default
        if isinstance(pipeline_module, ReadingModule):
            if pipeline_module.m_input_location is None:
                pipeline_module.m_input_location = self._m_input_place

        pipeline_module.connect_database(self.m_data_storage)

        if pipeline_module.name in self._m_modules:
            warnings.warn('Processing module names need to be unique. Overwriting the old Module')

        self._m_modules[pipeline_module.name] = pipeline_module

    def remove_module(self,
                      name):
        """
        Removes a Pypeline module from the internal dictionary.

        :param name: The name (key) of the module which has to be removed.
        :type name: String
        :return: None
        """

        if name in self._m_modules:
            del self._m_modules[name]
            return True
        else:
            return False

    def get_module_names(self):
        """
        Function to get information about the stored Pypeline modules.

        :return: Ordered list of all pipeline names (String)
        """

        return self._m_modules.keys()

    def run(self):
        """
        Walks through all saved processing steps and calls their run methods. The order the steps
        are called depends on the order they have been added to the Pypeline instance.
        NOTE: The method prints information about the current process.

        :return: None
        """

        print "Start running Pypeline ..."
        for key in self._m_modules:
            print "Start running " + key + "..."
            self._m_modules[key].run()
            print "Finished running " + key
        print "Finished running Pypeline."

    def run_module(self, name):
        """
        Runs a specific processing module identified by name.

        :param name: Name of the module to be run.
        :type name: String
        :return: None
        """
        if name in self._m_modules:
            self._m_modules[name].run()
        else:
            warnings.warn('Module not found')

    def get_data(self,
                 tag):
        # TODO Documentation
        self.m_data_storage.open_connection()
        return np.asarray(self.m_data_storage.m_data_bank[tag])

    def get_attribute(self,
                      data_tag,
                      attr_name):
        # TODO Documentation
        self.m_data_storage.open_connection()
        return self.m_data_storage.m_data_bank[data_tag].attrs[attr_name]
