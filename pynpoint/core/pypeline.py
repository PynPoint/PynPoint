"""
Module which capsules the methods of the Pypeline.
"""

import os
import warnings
import configparser
import collections
import multiprocessing

import h5py
import numpy as np

import pynpoint

from pynpoint.core.attributes import get_attributes
from pynpoint.core.dataio import DataStorage
from pynpoint.core.processing import PypelineModule, ReadingModule, WritingModule, ProcessingModule
from pynpoint.util.module import module_info, input_info, output_info


class Pypeline:
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

        Parameters
        ----------
        working_place_in : str
            Working location of the Pypeline which needs to be a folder on the hard drive. The
            given folder will be used to save the central PynPoint database (an HDF5 file) in
            which all the intermediate processing steps are saved. Note that the HDF5 file can
            become very large depending on the size and number of input images.
        input_place_in : str
            Default input directory of the Pypeline. All ReadingModules added to the Pypeline
            use this directory to look for input data. It is possible to specify a different
            location for the ReadingModules using their constructors.
        output_place_in : str
            Default result directory used to save the output of all WritingModules added to the
            Pypeline. It is possible to specify a different locations for the WritingModules by
            using their constructors.

        Returns
        -------
        NoneType
            None
        """

        pynpoint_version = 'PynPoint v' + pynpoint.__version__

        print(len(pynpoint_version) * '=')
        print(pynpoint_version)
        print(len(pynpoint_version) * '=' + '\n')

        self._m_working_place = working_place_in
        self._m_input_place = input_place_in
        self._m_output_place = output_place_in

        self._m_modules = collections.OrderedDict()

        self.m_data_storage = DataStorage(os.path.join(working_place_in, 'PynPoint_database.hdf5'))
        print(f'Database: {self.m_data_storage._m_location}')

        self._config_init()

    def __setattr__(self,
                    key,
                    value):
        """
        This method is called every time a member / attribute of the Pypeline is changed. It checks
        whether a chosen working / input / output directory exists.

        Parameters
        ----------
        key : str
            Member or attribute name.
        value : str
            New value for the given member or attribute.

        Returns
        -------
        NoneType
            None
        """

        if key in ['_m_working_place', '_m_input_place', '_m_output_place']:
            assert (os.path.isdir(str(value))), f'Input directory for {key} does not exist - ' \
                                                f'input requested: {value}.'

        super(Pypeline, self).__setattr__(key, value)

    @staticmethod
    def _validate(module,
                  tags):
        """
        Internal function which is used for the validation of the pipeline. Validates a
        single module.

        Parameters
        ----------
        module : ReadingModule, WritingModule, or ProcessingModule
            The pipeline module.
        tags : list(str, )
            Tags in the database.

        Returns
        -------
        bool
            Module validation.
        str
            Module name.
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

        Returns
        -------
        NoneType
            None
        """

        def _create_config(filename, attributes):
            file_obj = open(filename, 'w')
            file_obj.write('[header]\n\n')

            for key, val in attributes.items():
                if val['config'] == 'header':
                    file_obj.write(key+': '+str(val['value'])+'\n')

            file_obj.write('\n[settings]\n\n')

            for key, val in attributes.items():
                if val['config'] == 'settings':
                    file_obj.write(key+': '+str(val['value'])+'\n')

            file_obj.close()

        def _read_config(config_file, attributes):
            config = configparser.ConfigParser()
            with open(config_file) as cf_open:
                config.read_file(cf_open)

            for key, val in attributes.items():
                if config.has_option(val['config'], key):
                    if config.get(val['config'], key) == 'None':
                        if val['config'] == 'header':
                            attributes[key]['value'] = 'None'

                        # elif val['type'] == 'str':
                        #     attributes[key]['value'] = 'None'

                        elif val['type'] == 'float':
                            attributes[key]['value'] = float(0.)

                        elif val['type'] == 'int':
                            attributes[key]['value'] = int(0)

                    else:
                        if val['config'] == 'header':
                            attributes[key]['value'] = str(config.get(val['config'], key))

                        # elif val['type'] == 'str':
                        #     attributes[key]['value'] = str(config.get(val['config'], key))

                        elif val['type'] == 'float':
                            attributes[key]['value'] = float(config.get(val['config'], key))

                        elif val['type'] == 'int':
                            attributes[key]['value'] = int(config.get(val['config'], key))

            return attributes

        def _write_config(attributes):
            hdf = h5py.File(self._m_working_place+'/PynPoint_database.hdf5', 'a')

            if 'config' in hdf:
                del hdf['config']

            config = hdf.create_group('config')

            for key in attributes.keys():
                if attributes[key]['value'] is not None:
                    config.attrs[key] = attributes[key]['value']

            config.attrs['WORKING_PLACE'] = self._m_working_place

            hdf.close()

        config_file = os.path.join(self._m_working_place, 'PynPoint_config.ini')
        print(f'Configuration: {config_file}\n')

        attributes = get_attributes()
        attributes['CPU']['value'] = multiprocessing.cpu_count()

        if not os.path.isfile(config_file):
            warnings.warn('Configuration file not found. Creating PynPoint_config.ini with '
                          'default values in the working place.')

            _create_config(config_file, attributes)

        attributes = _read_config(config_file, attributes)

        _write_config(attributes)

        n_cpu = attributes['CPU']['value']

        if 'OMP_NUM_THREADS' in os.environ:
            n_thread = os.environ['OMP_NUM_THREADS']
        else:
            n_thread = 'not set'

        print(f'Number of CPUs: {n_cpu}')
        print(f'Number of threads: {n_thread}')

    def add_module(self,
                   module):
        """
        Adds a Pypeline module to the internal Pypeline dictionary. The module is appended at the
        end of this ordered dictionary. If the input module is a reading or writing module without
        a specified input or output location then the Pypeline default location is used. Moreover,
        the given module is connected to the Pypeline internal data storage.

        Parameters
        ----------
        module : ReadingModule, WritingModule, or ProcessingModule
            Input pipeline module.

        Returns
        -------
        NoneType
            None
        """

        assert isinstance(module, PypelineModule), 'The added module is not a valid ' \
                                                   'Pypeline module.'

        if isinstance(module, WritingModule):
            if module.m_output_location is None:
                module.m_output_location = self._m_output_place

        if isinstance(module, ReadingModule):
            if module.m_input_location is None:
                module.m_input_location = self._m_input_place

        module.connect_database(self.m_data_storage)

        if module.name in self._m_modules:
            warnings.warn(f'Pipeline module names need to be unique. Overwriting module '
                          f'\'{module.name}\'.')

        self._m_modules[module.name] = module

    def remove_module(self,
                      name):
        """
        Removes a Pypeline module from the internal dictionary.

        Parameters
        ----------
        name : str
            Name of the module that has to be removed.

        Returns
        -------
        bool
            Confirmation of removal.
        """

        if name in self._m_modules:
            del self._m_modules[name]
            removed = True

        else:
            warnings.warn(f'Module name \'{name}\' not found in the Pypeline dictionary.')
            removed = False

        return removed

    def get_module_names(self):
        """
        Function which returns a list of all module names.

        Returns
        -------
        list(str, )
            Ordered list of all Pypeline modules.
        """

        return list(self._m_modules.keys())

    def validate_pipeline(self):
        """
        Function which checks if all input ports of the Pypeline are pointing to previous output
        ports.

        Returns
        -------
        bool
            Confirmation of pipeline validation.
        str
            Module name that is not valid.
        """

        self.m_data_storage.open_connection()

        data_tags = list(self.m_data_storage.m_data_bank.keys())

        for module in self._m_modules.values():
            validation = self._validate(module, data_tags)

            if not validation[0]:
                break

        else:
            validation = True, None

        return validation

    def validate_pipeline_module(self,
                                 name):
        """
        Checks if the data exists for the module with label *name*.

        Parameters
        ----------
        name : str
            Name of the module that is checked.

        Returns
        -------
        bool
            Confirmation of pipeline module validation.
        str
            Module name that is not valid.
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

        Returns
        -------
        NoneType
            None
        """

        validation = self.validate_pipeline()

        if not validation[0]:
            raise AttributeError(f'Pipeline module \'{validation[1]}\' is looking for data '
                                 f'under a tag which is not created by a previous module or '
                                 f'does not exist in the database.')

        for name in self._m_modules:
            module_info(self._m_modules[name])

            if hasattr(self._m_modules[name], '_m_input_ports'):
                if len(self._m_modules[name]._m_input_ports) > 0:
                    input_info(self._m_modules[name])

            self._m_modules[name].run()

            if hasattr(self._m_modules[name], '_m_output_ports'):
                output_shape = {}
                for item in self._m_modules[name]._m_output_ports:
                    output_shape[item] = self.get_shape(item)

                output_info(self._m_modules[name], output_shape)

    def run_module(self,
                   name):
        """
        Runs a single processing module.

        Parameters
        ----------
        name : str
            Name of the pipeline module.

        Returns
        -------
        NoneType
            None
        """

        if name in self._m_modules:
            validation = self.validate_pipeline_module(name)

            if not validation[0]:
                raise AttributeError(f'Pipeline module \'{validation[1]}\' is looking for data '
                                     f'under a tag which does not exist in the database.')

            module_info(self._m_modules[name])

            if hasattr(self._m_modules[name], '_m_input_ports'):
                if len(self._m_modules[name]._m_input_ports) > 0:
                    input_info(self._m_modules[name])

            self._m_modules[name].run()

            if hasattr(self._m_modules[name], '_m_output_ports'):
                output_shape = {}
                for item in self._m_modules[name]._m_output_ports:
                    output_shape[item] = self.get_shape(item)

                output_info(self._m_modules[name], output_shape)

        else:
            warnings.warn(f'Module \'{name}\' not found.')

    def get_data(self,
                 tag,
                 data_range=None):
        """
        Function for accessing data in the central database.

        Parameters
        ----------
        tag : str
            Database tag.
        data_range : tuple(int, int), None
            Slicing range which can be used to select a subset of images from a 3D dataset. All
            data are selected if set to None.

        Returns
        -------
        numpy.ndarray
            The selected dataset from the database.
        """

        self.m_data_storage.open_connection()

        if data_range is None:
            data = np.asarray(self.m_data_storage.m_data_bank[tag])

        else:
            data = np.asarray(self.m_data_storage.m_data_bank[tag][data_range[0]:data_range[1], ])

        self.m_data_storage.close_connection()

        return data

    def delete_data(self,
                    tag):
        """
        Function for deleting a dataset and related attributes from the central database. Disk
        space does not seem to free up when using this function.

        Parameters
        ----------
        tag : str
            Database tag.

        Returns
        -------
        NoneType
            None
        """

        self.m_data_storage.open_connection()

        if tag in self.m_data_storage.m_data_bank:
            del self.m_data_storage.m_data_bank[tag]
        else:
            warnings.warn(f'Dataset \'{tag}\' not found in the database.')

        if 'header_' + tag + '/' in self.m_data_storage.m_data_bank:
            del self.m_data_storage.m_data_bank[f'header_{tag}']
        else:
            warnings.warn(f'Attributes of \'{tag}\' not found in the database.')

        self.m_data_storage.close_connection()

    def get_attribute(self,
                      data_tag,
                      attr_name,
                      static=True):
        """
        Function for accessing attributes in the central database.

        Parameters
        ----------
        data_tag : str
            Database tag.
        attr_name : str
            Name of the attribute.
        static : bool
            Static or non-static attribute.

        Returns
        -------
        numpy.ndarray
            The attribute values.
        """

        self.m_data_storage.open_connection()

        if static:
            attr = self.m_data_storage.m_data_bank[data_tag].attrs[attr_name]

        else:
            attr = self.m_data_storage.m_data_bank[f'header_{data_tag}/{attr_name}']
            attr = np.asarray(attr)

        self.m_data_storage.close_connection()

        return attr

    def set_attribute(self,
                      data_tag,
                      attr_name,
                      attr_value,
                      static=True):
        """
        Function for writing attributes to the central database. Existing values will be
        overwritten.

        Parameters
        ----------
        data_tag : str
            Database tag.
        attr_name : str
            Name of the attribute.
        attr_value : float, int, str, tuple, or numpy.ndarray
            Attribute value.
        static : bool
            Static or non-static attribute.

        Returns
        -------
        NoneType
            None
        """

        self.m_data_storage.open_connection()

        if static:
            self.m_data_storage.m_data_bank[data_tag].attrs[attr_name] = attr_value

        else:
            if isinstance(attr_value[0], str):
                attr_value = np.array(attr_value, dtype='|S')

            if attr_name in list(self.m_data_storage.m_data_bank[f'header_{data_tag}'].keys()):
                del self.m_data_storage.m_data_bank[f'header_{data_tag}/{attr_name}']

            attr_key = f'header_{data_tag}/{attr_name}'
            self.m_data_storage.m_data_bank[attr_key] = np.asarray(attr_value)

        self.m_data_storage.close_connection()

    def get_tags(self):
        """
        Function for listing the database tags, ignoring header and config tags.

        Returns
        -------
        numpy.asarray
            Database tags.
        """

        self.m_data_storage.open_connection()

        tags = list(self.m_data_storage.m_data_bank.keys())
        select = []

        for item in tags:
            if item in ('config', 'fits_header') or item[0:7] == 'header_':
                continue

            select.append(item)

        self.m_data_storage.close_connection()

        return np.asarray(select)

    def get_shape(self,
                  tag):
        """
        Function for getting the shape of a database entry.

        Parameters
        ----------
        tag : str
            Database tag.

        Returns
        -------
        tuple(int, )
            Dataset shape.
        """

        self.m_data_storage.open_connection()

        if tag in self.m_data_storage.m_data_bank:
            data_shape = self.m_data_storage.m_data_bank[tag].shape
        else:
            data_shape = None

        self.m_data_storage.close_connection()

        return data_shape
