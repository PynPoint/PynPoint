"""
Module which capsules the methods of the Pypeline.
"""

from __future__ import absolute_import

import os
import sys
import warnings
import configparser
import collections
import multiprocessing

import six
import h5py
import numpy as np

import pynpoint

from pynpoint.core.attributes import get_attributes
from pynpoint.core.dataio import DataStorage
from pynpoint.core.processing import PypelineModule, WritingModule, ReadingModule, ProcessingModule


class Pypeline(object):
    """
    A Pypeline instance can be used to manage various processing steps. It inheres an internal
    dictionary of Pypeline steps (modules) and their names. A Pypeline has a central DataStorage on
    the hard drive which can be accessed by various modules. The order of the modules depends on
    the order the steps have been added to the pypeline. It is possible to run all modules attached
    to the Pypeline at once or run a single modules by name.
    """

    def __init__(self,
                 working_place_in=None,
                 input_place_in=None,
                 output_place_in=None):
        """
        Constructor of Pypeline.

        :param working_place_in: Working location of the Pypeline which needs to be a folder on the
                                 hard drive. The given folder will be used to save the central
                                 PynPoint database (an HDF5 file) in which all the intermediate
                                 processing steps are saved. Note that the HDF5 file can become
                                 very large depending on the size and number of input images.
        :type working_place_in: str
        :param input_place_in: Default input directory of the Pypeline. All ReadingModules added
                               to the Pypeline use this directory to look for input data. It is
                               possible to specify a different location for the ReadingModules
                               using their constructors.
        :type input_place_in: str
        :param output_place_in: Default result directory used to save the output of all
                                WritingModules added to the Pypeline. It is possible to specify
                                a different locations for the WritingModules by using their
                                constructors.

        :return: None
        """

        sys.stdout.write("Initiating PynPoint v"+pynpoint.__version__+"...")
        sys.stdout.flush()

        self._m_working_place = working_place_in
        self._m_input_place = input_place_in
        self._m_output_place = output_place_in

        self._m_modules = collections.OrderedDict()
        self.m_data_storage = DataStorage(os.path.join(working_place_in, 'PynPoint_database.hdf5'))

        self._config_init()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

    def __setattr__(self,
                    key,
                    value):
        """
        This method is called every time a member / attribute of the Pypeline is changed. It checks
        whether a chosen working / input / output directory exists.

        :param key: Member or attribute name.
        :param value: New value for the given member or attribute.

        :return: None
        """

        if key in ["_m_working_place", "_m_input_place", "_m_output_place"]:
            assert (os.path.isdir(str(value))), "Input directory for " + str(key) + " does not " \
                                                "exist - input requested: %s." % value

        super(Pypeline, self).__setattr__(key, value)

    @staticmethod
    def _validate(module,
                  tags):
        """
        Internal function which is used for the validation of the pipeline. Validates a
        single module.

        :param module: The module.
        :type module: ReadingModule, WritingModule, ProcessingModule
        :param tags: Tags in the database.
        :type tags: list, str

        :return: Module validation.
        :rtype: bool, str
        """

        if isinstance(module, ReadingModule):
            tags.extend(module.get_all_output_tags())

        elif isinstance(module, WritingModule):
            for tag in module.get_all_input_tags():
                if tag not in tags:
                    return False, module.name

        elif isinstance(module, ProcessingModule):
            tags.extend(module.get_all_output_tags())
            for tag in module.get_all_input_tags():
                if tag not in tags:
                    return False, module.name

        else:
            return False, None

        return True, None

    def _config_init(self):
        """
        Internal function which initializes the configuration file. It reads PynPoint_config.ini
        in the working folder and creates this file with the default (ESO/NACO) settings in case
        the file is not present.

        :return: None
        """

        def _create_config(filename, attributes):
            file_obj = open(filename, 'w')
            file_obj.write("[header]\n\n")

            for key, val in six.iteritems(attributes):
                if val['config'] == "header":
                    file_obj.write(key+': '+str(val['value'])+'\n')

            file_obj.write("\n[settings]\n\n")

            for key, val in six.iteritems(attributes):
                if val['config'] == "settings":
                    file_obj.write(key+': '+str(val['value'])+'\n')

            file_obj.close()

        def _read_config(config_file, attributes):
            config = configparser.ConfigParser()
            config.read_file(open(config_file))

            for key, val in six.iteritems(attributes):
                if config.has_option(val["config"], key):
                    if config.get(val["config"], key) == "None":
                        if val["config"] == "header":
                            attributes[key]["value"] = "None"

                        # elif val["type"] == "str":
                        #     attributes[key]["value"] = "None"

                        elif val["type"] == "float":
                            attributes[key]["value"] = float(0.)

                        elif val["type"] == "int":
                            attributes[key]["value"] = int(0)

                    else:
                        if val["config"] == "header":
                            attributes[key]["value"] = str(config.get(val["config"], key))

                        # elif val["type"] == "str":
                        #     attributes[key]["value"] = str(config.get(val["config"], key))

                        elif val["type"] == "float":
                            attributes[key]["value"] = float(config.get(val["config"], key))

                        elif val["type"] == "int":
                            attributes[key]["value"] = int(config.get(val["config"], key))

            return attributes

        def _write_config(attributes):
            hdf = h5py.File(self._m_working_place+'/PynPoint_database.hdf5', 'a')

            if "config" in hdf:
                del hdf["config"]

            config = hdf.create_group("config")

            for key, _ in six.iteritems(attributes):
                if attributes[key]["value"] is not None:
                    config.attrs[key] = attributes[key]["value"]

            hdf.close()

        config_file = self._m_working_place+"/PynPoint_config.ini"

        attributes = get_attributes()
        attributes["CPU"]["value"] = multiprocessing.cpu_count()

        if not os.path.isfile(config_file):
            warnings.warn("Configuration file not found. Creating PynPoint_config.ini with "
                          "default values in the working place.")

            _create_config(config_file, attributes)

        attributes = _read_config(config_file, attributes)

        _write_config(attributes)

    def add_module(self,
                   module):
        """
        Adds a Pypeline module to the internal Pypeline dictionary. The module is appended at the
        end of this ordered dictionary. If the input module is a reading or writing module without
        a specified input or output location then the Pypeline default location is used. Moreover,
        the given module is connected to the Pypeline internal data storage.

        :param module: Input module.
        :type module: ReadingModule, WritingModule, ProcessingModule

        :return: None
        """

        assert isinstance(module, PypelineModule), "The added module is not a valid " \
                                                   "Pypeline module."

        if isinstance(module, WritingModule):
            if module.m_output_location is None:
                module.m_output_location = self._m_output_place

        if isinstance(module, ReadingModule):
            if module.m_input_location is None:
                module.m_input_location = self._m_input_place

        module.connect_database(self.m_data_storage)

        if module.name in self._m_modules:
            warnings.warn("Processing module names need to be unique. Overwriting module '%s'."
                          % module.name)

        self._m_modules[module.name] = module

    def remove_module(self,
                      name):
        """
        Removes a Pypeline module from the internal dictionary.

        :param name: Name of the module which has to be removed.
        :type name: str

        :return: True if module was deleted and False if module does not exist.
        :rtype: bool
        """

        if name in self._m_modules:
            del self._m_modules[name]
            return True

        warnings.warn("Module name '"+name+"' not found in the Pypeline dictionary.")

        return False

    def get_module_names(self):
        """
        Function which returns a list of all module names.

        :return: Ordered list of all Pypeline modules.
        :rtype: list[str]
        """

        return list(self._m_modules.keys())

    def validate_pipeline(self):
        """
        Function which checks if all input ports of the Pypeline are pointing to previous output
        ports.

        :return: True if Pypeline is valid and False if not. The second parameter contains the name
                 of the module which is not valid.
        :rtype: bool, str
        """

        self.m_data_storage.open_connection()

        existing_data_tags = list(self.m_data_storage.m_data_bank.keys())

        for module in six.itervalues(self._m_modules):
            validation = self._validate(module, existing_data_tags)

            if not validation[0]:
                return validation

        return True, None

    def validate_pipeline_module(self, name):
        """
        Checks if the data exists for the module with label *name*.

        :param name: Name of the module that is checked.
        :type name: str

        :return: True if the Pypeline module is valid and False if not. The second parameter gives
                 the name of the module which is not valid.
        :rtype: bool, str
        """

        self.m_data_storage.open_connection()

        existing_data_tags = list(self.m_data_storage.m_data_bank.keys())

        if name in self._m_modules:
            module = self._m_modules[name]
            validate = self._validate(module, existing_data_tags)

        else:
            validate = None

        return validate

    def run(self):
        """
        Walks through all saved processing steps and calls their run methods. The order in which
        the steps are called depends on the order they have been added to the Pypeline.

        :return: None
        """

        sys.stdout.write("Validating Pypeline...")
        sys.stdout.flush()

        validation = self.validate_pipeline()

        if not validation[0]:
            raise AttributeError("Pipeline module '%s' is looking for data under a tag which is "
                                 "not created by a previous module or does not exist in the "
                                 "database." % validation[1])

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        for key in self._m_modules:
            self._m_modules[key].run()

    def run_module(self, name):
        """
        Runs a single processing module.

        :param name: Name of the module.
        :type name: str

        :return: None
        """

        if name in self._m_modules:
            sys.stdout.write("Validating module "+name+"...")
            sys.stdout.flush()

            validation = self.validate_pipeline_module(name)

            if not validation[0]:
                raise AttributeError("Pipeline module '%s' is looking for data under a tag which "
                                     "does not exist in the database." % validation[1])

            sys.stdout.write(" [DONE]\n")
            sys.stdout.flush()

            self._m_modules[name].run()

        else:
            warnings.warn("Module '"+name+"' not found.")

    def get_data(self,
                 tag):
        """
        Function for accessing data in the central database.

        :param tag: Database tag.
        :type tag: str

        :return: The selected dataset from the database.
        :rtype: numpy.ndarray
        """

        self.m_data_storage.open_connection()

        return np.asarray(self.m_data_storage.m_data_bank[tag])

    def get_attribute(self,
                      data_tag,
                      attr_name,
                      static=True):
        """
        Function for accessing attributes in the central database.

        :param data_tag: Database tag.
        :type data_tag: str
        :param attr_name: Name of the attribute.
        :type attr_name: str
        :param static: Static or non-static attribute.
        :type static: bool

        :return: The attribute value(s).
        """

        self.m_data_storage.open_connection()

        if static:
            attr = self.m_data_storage.m_data_bank[data_tag].attrs[attr_name]

        else:
            attr = self.m_data_storage.m_data_bank["header_"+data_tag+"/"+attr_name]
            attr = np.asarray(attr)

        return attr

    def set_attribute(self,
                      data_tag,
                      attr_name,
                      value,
                      static=True):
        """
        Function for writing attributes to the central database. Existing values will be
        overwritten.

        :param data_tag: Database tag.
        :type data_tag: str
        :param attr_name: Name of the attribute.
        :type attr_name: str
        :param value: Attribute value.
        :type value: float, int, str, tuple, numpy.ndarray
        :param static: Static or non-static attribute.
        :type static: bool

        :return: None
        """

        self.m_data_storage.open_connection()

        if static:
            self.m_data_storage.m_data_bank[data_tag].attrs[attr_name] = value

        else:
            if isinstance(value[0], str):
                value = np.array(value, dtype="|S")

            if attr_name in list(self.m_data_storage.m_data_bank["header_"+data_tag].keys()):
                del self.m_data_storage.m_data_bank["header_"+data_tag+"/"+attr_name]

            self.m_data_storage.m_data_bank["header_"+data_tag+"/"+attr_name] = np.asarray(value)

        self.m_data_storage.close_connection()

    def get_tags(self):
        """
        Function for listing the database tags, ignoring header and config tags.

        :return: Database tags.
        :rtype: numpy.asarray
        """

        self.m_data_storage.open_connection()
        tags = list(self.m_data_storage.m_data_bank.keys())

        select = []
        for _, item in enumerate(tags):

            if item == "config" or item == "fits_header" or item[0:7] == "header_":
                continue
            else:
                select.append(item)

        return np.asarray(select)

    def get_shape(self, tag):
        """
        Function getting the shape of a database entry.

        :param tag: Database tag.
        :type tag: str

        :return: Database tags.
        :rtype: tuple
        """

        self.m_data_storage.open_connection()

        return self.m_data_storage.m_data_bank[tag].shape
