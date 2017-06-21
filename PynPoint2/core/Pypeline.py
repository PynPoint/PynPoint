"""
Module which capsules different pipeline processing steps.
"""
# external modules
import collections
import os
import atexit

import numpy as np
from PynPoint.core.DataIO import DataStorage

from PynPoint.core.Processing import PypelineModule, WritingModule, ReadingModule, ProcessingModule

import warnings


class Pypeline(object):
    """
    A Pypeline instance can be used to manage various processing steps. It inheres a internal
    dictionary of Pipeline steps (modules) and their names. A Pypeline has a central DataStorage on
    the hard drive which can be accessed by these different modules. The order of the modules
    depends on the order the steps have been added to the pypeline. It is possible to run the whole
    Pypeline (i.e. all modules / steps) or a single modules by name.
    """

    def __init__(self,
                 working_place_in=None,
                 input_place_in=None,
                 output_place_in=None):
        """
        Constructor of a Pypeline object.

        :param working_place_in: Working location of the Pypeline which needs to be a folder on the
                                 hard drive. The given folder will be used to save the central
                                 PynPoint database (a .hdf5 file). **NOTE**: Depending on the input
                                 this .hdf5 file can become very large!
        :type working_place_in: str
        :param input_place_in: Default input directory of the Pypeline. All ReadingModules added to
                               the Pypeline can use this directory to look for input data. It is
                               possible to specify a different location for each Reading Modules
                               using their constructors.
        :type input_place_in: str
        :param output_place_in: Default result directory used to save the output of all
                                WritingModules added to the Pypeline. It is possible to specify
                                a different locations for each WritingModule using their
                                constructors.
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

    @staticmethod
    def _validate(module,
                  existing_data_tags):
        """
        Internal function which is used for the validation of the pipeline.
        It validates a single module.

        :param module: The module
        :param existing_data_tags: Tags which exist in the database
        :return: validation
        :rtype: bool, str
        """

        if isinstance(module, ReadingModule):
            existing_data_tags.extend(module.get_all_output_tags())

        elif isinstance(module, WritingModule):
            for tag in module.get_all_input_tags():
                if tag not in existing_data_tags:
                    return False, module.name

        elif isinstance(module, ProcessingModule):
            existing_data_tags.extend(module.get_all_output_tags())
            for tag in module.get_all_input_tags():
                if tag not in existing_data_tags:
                    return False, module.name

        else: # pragma: no cover
            return False, None

        return True, None

    def add_module(self,
                   pipeline_module):
        """
        Adds a given pipeline module to the internal Pypeline step dictionary. The module is
        appended at the end of this ordered dictionary. If the input module is a reading or writing
        module without a specified input / output location the Pypeline default location is used.
        Moreover the given module is connected to the Pypeline internal data storage.

        :param pipeline_module: The input module. Needs to be either a Processing, Reading or
                                Writing Module.
        :type pipeline_module: ProcessingModule, ReadingModule, WritingModule
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
        Removes a Pypeline module from the internal dictionary. Returns

        :param name: The name (key) of the module which has to be removed.
        :type name: str
        :return: True if module was deleted False if module does not exist
        :rtype: bool
        """

        if name in self._m_modules:
            del self._m_modules[name]
            return True
        else:
            return False

    def get_module_names(self):
        """
        Function which returns a list of all module names.

        :return: Ordered list of all pipeline names
        :rtype: list[str]
        """

        return self._m_modules.keys()

    def validate_pipeline(self):
        """
        Function which checks if all input ports of the pipeline are lined to previous output ports.

        :return: True if pipeline is valid False of not. The second parameter gives the name of the
                 module which is not valid.
        :rtype: bool, str
        """

        self.m_data_storage.open_connection()
        existing_data_tags = self.m_data_storage.m_data_bank.keys()
        for module in self._m_modules.itervalues():

            validation = self._validate(module,
                                        existing_data_tags)

            if not validation[0]:
                return validation

        return True, None

    def validate_pipeline_module(self, name):
        """
        Checks if the data for the module with the name *name* exists.

        :param name: name of the module to be checked
        :type name: str
        :return: True if pipeline is valid False of not. The second parameter gives the name of the
                 module which is not valid.
        :rtype: bool, str
        """

        self.m_data_storage.open_connection()
        existing_data_tags = self.m_data_storage.m_data_bank.keys()

        if name in self._m_modules:
            module = self._m_modules[name]
        else:
            return

        return self._validate(module,
                              existing_data_tags)

    def run(self):
        """
        Walks through all saved processing steps and calls their run methods. The order the steps
        are called depends on the order they have been added to the Pypeline.
        **NOTE:** This method prints information about the current process.

        :return: None
        """

        print "validating Pipeline..."
        validation = self.validate_pipeline()
        if not validation[0]:
            raise AttributeError('Pipeline module %s is looking for data under a tag which is not '
                                 'created by a previous module or does not exist in the database.'
                                 % validation[1])

        print "Start running Pypeline ..."
        for key in self._m_modules:
            print "Start running " + key + "..."
            self._m_modules[key].run()
            print "Finished running " + key
        print "Finished running Pypeline."

    def run_module(self, name):
        """
        Runs a specific processing module identified by name.

        :param name: Name of the module.
        :type name: str
        :return: None
        """

        if name in self._m_modules:

            print "validating module..."
            validation = self.validate_pipeline_module(name)
            if not validation[0]:
                raise AttributeError(
                    'Pipeline module %s is looking for data under a tag which does not'
                    ' exist in the database.'
                    % validation[1])

            print "Start running module..."
            self._m_modules[name].run()
            print "finished running module..."
        else:
            warnings.warn('Module not found')

    def get_data(self,
                 tag):
        """
        Small function for easy data base access.

        :param tag: Dataset tag
        :type tag: str
        :return: The dataset
        :rtype: numpy array
        """
        self.m_data_storage.open_connection()
        return np.asarray(self.m_data_storage.m_data_bank[tag])

    def get_attribute(self,
                      data_tag,
                      attr_name):
        """
        Small function for easy attributes data access. Supports only static attributes.

        :param data_tag: Dataset tag
        :type data_tag: str
        :param attr_name: Name of the attribute
        :type attr_name: str
        :return: The attribute
        """
        self.m_data_storage.open_connection()
        return self.m_data_storage.m_data_bank[data_tag].attrs[attr_name]
