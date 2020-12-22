"""
Module which capsules the methods of the Pypeline.
"""

import collections
import configparser
import json
import multiprocessing
import os
import urllib.request
import warnings

from typing import Any, List, Optional, Tuple, Union
from urllib.error import URLError

import h5py
import numpy as np

from typeguard import typechecked

import pynpoint

from pynpoint.core.attributes import get_attributes
from pynpoint.core.dataio import DataStorage
from pynpoint.core.processing import ProcessingModule, PypelineModule, \
    ReadingModule, WritingModule
from pynpoint.util.module import input_info, module_info, output_info
from pynpoint.util.types import NonStaticAttribute, StaticAttribute


class Pypeline:
    """
    A Pypeline instance can be used to manage various processing steps. It inheres an internal
    dictionary of Pypeline steps (modules) and their names. A Pypeline has a central DataStorage on
    the hard drive which can be accessed by various modules. The order of the modules depends on
    the order the steps have been added to the pypeline. It is possible to run all modules attached
    to the Pypeline at once or run a single modules by name.
    """

    @typechecked
    def __init__(self,
                 working_place_in: Optional[str] = None,
                 input_place_in: Optional[str] = None,
                 output_place_in: Optional[str] = None) -> None:
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

        try:
            contents = urllib.request.urlopen('https://pypi.org/pypi/pynpoint/json').read()
            data = json.loads(contents)
            latest_version = data['info']['version']

        except URLError:
            latest_version = None

        if latest_version is not None and pynpoint.__version__ != latest_version:
            print(f'A new version ({latest_version}) is available!\n')
            print('Want to stay informed about updates, bug fixes, and new features?')
            print('Please consider using the \'Watch\' button on the Github page:')
            print('https://github.com/PynPoint/PynPoint\n')

        self._m_working_place = working_place_in
        self._m_input_place = input_place_in
        self._m_output_place = output_place_in

        self._m_modules = collections.OrderedDict()

        self.m_data_storage = DataStorage(os.path.join(working_place_in, 'PynPoint_database.hdf5'))
        print(f'Database: {self.m_data_storage._m_location}')

        self._config_init()

    @typechecked
    def __setattr__(self,
                    key: str,
                    value: Any) -> None:
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

        super().__setattr__(key, value)

    @staticmethod
    @typechecked
    def _validate(module: Union[ReadingModule, WritingModule, ProcessingModule],
                  tags: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Internal function which is used for the validation of the pipeline. Validates a
        single module.

        Parameters
        ----------
        module : ReadingModule, WritingModule, ProcessingModule
            The pipeline module.
        tags : list(str)
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

        return True, None

    @typechecked
    def _config_init(self) -> None:
        """
        Internal function which initializes the configuration file. It reads PynPoint_config.ini
        in the working folder and creates this file with the default (ESO/NACO) settings in case
        the file is not present.

        Returns
        -------
        NoneType
            None
        """

        @typechecked
        def _create_config(filename: str,
                           attributes: dict) -> None:

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

        @typechecked
        def _read_config(config_file: str,
                         attributes: dict) -> dict:

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

        @typechecked
        def _write_config(attributes: dict) -> None:

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

    @typechecked
    def add_module(self,
                   module: PypelineModule) -> None:
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

    @typechecked
    def remove_module(self,
                      name: str) -> bool:
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
            warnings.warn(f'Pipeline module name \'{name}\' not found in the Pypeline dictionary.')
            removed = False

        return removed

    @typechecked
    def get_module_names(self) -> List[str]:
        """
        Function which returns a list of all module names.

        Returns
        -------
        list(str, )
            Ordered list of all Pypeline modules.
        """

        return list(self._m_modules.keys())

    @typechecked
    def validate_pipeline(self) -> Tuple[bool, Optional[str]]:
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

    @typechecked
    def validate_pipeline_module(self,
                                 name: str) -> Optional[Tuple[bool, Optional[str]]]:
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

    @typechecked
    def run(self) -> None:
        """
        Function for running all pipeline modules that are added to the
        :class:`~pynpoint.core.pypeline.Pypeline`.

        Returns
        -------
        NoneType
            None
        """

        # Validate the pipeline
        validation = self.validate_pipeline()

        if not validation[0]:
            # Check if the input data is available
            raise AttributeError(f'Pipeline module \'{validation[1]}\' is looking for data '
                                 f'under a tag which is not created by a previous module or '
                                 f'the data does not exist in the database.')

        # Loop over all pipeline modules and run them
        for name in self._m_modules:
            self.run_module(name)

    @typechecked
    def run_module(self,
                   name: str) -> None:
        """
        Function for running a pipeline module.

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
            # Validate the pipeline module
            validation = self.validate_pipeline_module(name)

            if not validation[0]:
                raise AttributeError(f'Pipeline module \'{validation[1]}\' is looking for data '
                                     f'under a tag which does not exist in the database.')

            # Print information about the pipeline module
            module_info(self._m_modules[name])

            # Check if the module has any input ports
            if hasattr(self._m_modules[name], '_m_input_ports'):
                # Check if the list of input ports is not empty
                if len(self._m_modules[name]._m_input_ports) > 0:
                    # Print information about the input ports
                    input_info(self._m_modules[name])

            # Check if the module has any output ports
            if hasattr(self._m_modules[name], '_m_output_ports'):
                for item in self._m_modules[name]._m_output_ports:
                    # Check if the module is a ProcessingModule
                    if isinstance(self._m_modules[name], ProcessingModule):
                        # Check if the database tag is already used
                        if item in self.m_data_storage.m_data_bank:
                            # Check if the output port is not used as input port
                            if hasattr(self._m_modules[name], '_m_input_ports') and \
                                    item not in self._m_modules[name]._m_input_ports:

                                print(f'Deleting data and attributes: {item}')

                                # Delete existing data and attributes
                                self._m_modules[name]._m_output_ports[item].del_all_data()
                                self._m_modules[name]._m_output_ports[item].del_all_attributes()

            # Run the pipeline module
            self._m_modules[name].run()

            # Check if the module has any output ports
            if hasattr(self._m_modules[name], '_m_output_ports'):
                output_shape = {}

                for item in self._m_modules[name]._m_output_ports:
                    # Get the shape of the output port
                    output_shape[item] = self.get_shape(item)

                # Print information about the output ports
                output_info(self._m_modules[name], output_shape)

        else:
            warnings.warn(f'Pipeline module \'{name}\' not found.')

    @typechecked
    def get_data(self,
                 tag: str,
                 data_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
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
        np.ndarray
            The selected dataset from the database.
        """

        self.m_data_storage.open_connection()

        if data_range is None:
            data = np.asarray(self.m_data_storage.m_data_bank[tag])

        else:
            data = np.asarray(self.m_data_storage.m_data_bank[tag][data_range[0]:data_range[1], ])

        self.m_data_storage.close_connection()

        return data

    @typechecked
    def delete_data(self,
                    tag: str) -> None:
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

    @typechecked
    def get_attribute(self,
                      data_tag: str,
                      attr_name: str,
                      static: bool = True) -> Union[StaticAttribute, NonStaticAttribute]:
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
        StaticAttribute, NonStaticAttribute
            The values of the attribute, which can either be static or non-static.
        """

        self.m_data_storage.open_connection()

        if static:
            attr = self.m_data_storage.m_data_bank[data_tag].attrs[attr_name]

        else:
            attr = self.m_data_storage.m_data_bank[f'header_{data_tag}/{attr_name}']
            attr = np.asarray(attr)

        self.m_data_storage.close_connection()

        return attr

    @typechecked
    def set_attribute(self,
                      data_tag: str,
                      attr_name: str,
                      attr_value: Union[StaticAttribute, NonStaticAttribute],
                      static: bool = True) -> None:
        """
        Function for writing attributes to the central database. Existing values will be
        overwritten.

        Parameters
        ----------
        data_tag : str
            Database tag.
        attr_name : str
            Name of the attribute.
        attr_value : StaticAttribute, NonStaticAttribute
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

    @typechecked
    def get_tags(self) -> np.ndarray:
        """
        Function for listing the database tags, ignoring header and config tags.

        Returns
        -------
        np.ndarray
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

    @typechecked
    def get_shape(self,
                  tag: str) -> Optional[Tuple[int, ...]]:
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
