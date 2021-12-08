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

from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import h5py
import numpy as np

from typeguard import typechecked

import pynpoint

from pynpoint.core.attributes import get_attributes
from pynpoint.core.dataio import DataStorage
from pynpoint.core.processing import ProcessingModule, PypelineModule, ReadingModule, WritingModule
from pynpoint.util.module import input_info, module_info, output_info
from pynpoint.util.type_aliases import NonStaticAttribute, StaticAttribute


class Pypeline:
    """
    The :class:`~pynpoint.core.pypeline.Pypeline` class manages the pipeline modules. It inheres an
    internal dictionary of pipeline modules and has a :class:`~pynpoint.core.dataio.DataStorage`
    which is accessed by the various modules. The order in which the pipeline modules are executed
    depends on the order they have been added to the :class:`~pynpoint.core.pypeline.Pypeline`. It
    is possible to run all modules at once or run a single module by name.
    """

    @typechecked
    def __init__(self,
                 working_place_in: Optional[str] = None,
                 input_place_in: Optional[str] = None,
                 output_place_in: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        working_place_in : str, None
            Working location where the central HDF5 database and the configuration file will be
            stored. Sufficient space is required in the working folder since each pipeline module
            stores a dataset in the HDF5 database. The current working folder of Python is used as
            working folder if the argument is set to None.
        input_place_in : str, None
            Default input folder where a :class:`~pynpoint.core.processing.ReadingModule` that is
            added to the :class:`~pynpoint.core.pypeline.Pypeline` will look for input data. The
            current working folder of Python is used as input folder if the argument is set to
            None.
        output_place_in : str, None
            Default output folder where a :class:`~pynpoint.core.processing.WritingModule` that is
            added to the :class:`~pynpoint.core.pypeline.Pypeline` will store output data. The
            current working folder of Python is used as output folder if the argument is set to
            None.

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

        if working_place_in is None:
            self._m_working_place = os.getcwd()
        else:
            self._m_working_place = working_place_in

        if input_place_in is None:
            self._m_input_place = os.getcwd()
        else:
            self._m_input_place = input_place_in

        if output_place_in is None:
            self._m_output_place = os.getcwd()
        else:
            self._m_output_place = output_place_in

        print(f'Working place: {self._m_working_place}')
        print(f'Input place: {self._m_input_place}')
        print(f'Output place: {self._m_output_place}\n')

        self._m_modules = collections.OrderedDict()

        hdf5_path = os.path.join(self._m_working_place, 'PynPoint_database.hdf5')
        self.m_data_storage = DataStorage(hdf5_path)

        print(f'Database: {self.m_data_storage._m_location}')

        self._config_init()

    @typechecked
    def __setattr__(self,
                    key: str,
                    value: Any) -> None:
        """
        Internal method which assigns a value to an object attribute. This method is called
        whenever and attribute of the :class:`~pynpoint.core.pypeline.Pypeline` is changed and
        checks if the chosen working, input, or output folder exists.

        Parameters
        ----------
        key : str
            Attribute name.
        value : str
            Value for the attribute.

        Returns
        -------
        NoneType
            None
        """

        if key == '_m_working_place':
            error_msg = f'The folder that was chosen for the working place does not exist: {value}.'
            assert os.path.isdir(str(value)), error_msg

        elif key == '_m_input_place':
            error_msg = f'The folder that was chosen for the input place does not exist: {value}.'
            assert os.path.isdir(str(value)), error_msg

        elif key == '_m_output_place':
            error_msg = f'The folder that was chosen for the output place does not exist: {value}.'
            assert os.path.isdir(str(value)), error_msg

        super().__setattr__(key, value)

    @staticmethod
    @typechecked
    def _validate(module: Union[ReadingModule, WritingModule, ProcessingModule],
                  tags: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Internal method to validate a :class:`~pynpoint.core.processing.PypelineModule`.

        Parameters
        ----------
        module : ReadingModule, WritingModule, ProcessingModule
            Pipeline module that will be validated.
        tags : list(str)
            Tags that are present in the database.

        Returns
        -------
        bool
            Validation of the pipeline module.
        str, None
            Pipeline module name in case it is not valid. Returns None if the module was validated.
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
        Internal method to initialize the configuration file. The configuration parameters are read
        from *PynPoint_config.ini* in the working folder. The file is created with default values
        (ESO/NACO) in case the file is not present.

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
        Method for adding a :class:`~pynpoint.core.processing.PypelineModule` to the internal
        dictionary of the :class:`~pynpoint.core.pypeline.Pypeline`. The module is appended at the
        end of this ordered dictionary. If the input module is a reading or writing module without
        a specified input or output location then the default location is used. The module is
        connected to the internal data storage of the :class:`~pynpoint.core.pypeline.Pypeline`.

        Parameters
        ----------
        module : ReadingModule, WritingModule, ProcessingModule
            Pipeline module that will be added to the :class:`~pynpoint.core.pypeline.Pypeline`.

        Returns
        -------
        NoneType
            None
        """

        if isinstance(module, ReadingModule):
            if module.m_input_location is None:
                module.m_input_location = self._m_input_place

        if isinstance(module, WritingModule):
            if module.m_output_location is None:
                module.m_output_location = self._m_output_place

        module.connect_database(self.m_data_storage)

        if module.name in self._m_modules:
            warnings.warn(f'Names of pipeline modules that are added to the Pypeline need to '
                          f'be unique. The current pipeline module, \'{module.name}\', does '
                          f'already exist in the Pypeline dictionary so the previous module '
                          f'with the same name will be overwritten.')

        self._m_modules[module.name] = module

    @typechecked
    def remove_module(self,
                      name: str) -> bool:
        """
        Method to remove a :class:`~pynpoint.core.processing.PypelineModule` from the internal
        dictionary with pipeline modules that are added to the
        :class:`~pynpoint.core.pypeline.Pypeline`.

        Parameters
        ----------
        name : str
            Name of the module that has to be removed.

        Returns
        -------
        bool
            Confirmation of removing the :class:`~pynpoint.core.processing.PypelineModule`.
        """

        if name in self._m_modules:
            del self._m_modules[name]

            removed = True

        else:
            warnings.warn(f'Pipeline module \'{name}\' is not found in the Pypeline dictionary '
                          f'so it could not be removed. The dictionary contains the following '
                          f'modules: {list(self._m_modules.keys())}.')

            removed = False

        return removed

    @typechecked
    def get_module_names(self) -> List[str]:
        """
        Method to return a list with the names of all pipeline modules that are added to the
        :class:`~pynpoint.core.pypeline.Pypeline`.

        Returns
        -------
        list(str)
            Ordered list of all Pypeline modules.
        """

        return list(self._m_modules.keys())

    @typechecked
    def validate_pipeline(self) -> Tuple[bool, Optional[str]]:
        """
        Method to check if each :class:`~pynpoint.core.dataio.InputPort` is pointing to an
        :class:`~pynpoint.core.dataio.OutputPort` of a previously added
        :class:`~pynpoint.core.processing.PypelineModule`.

        Returns
        -------
        bool
            Validation of the pipeline.
        str, None
            Name of the pipeline module that can not be validated. Returns None if all modules
            were validated.
        """

        self.m_data_storage.open_connection()

        # Create list with all datasets that are stored in the database
        data_tags = list(self.m_data_storage.m_data_bank.keys())

        # Initiate the validation in case self._m_modules.values() is empty
        validation = (True, None)

        # Loop over all pipline modules in the ordered dictionary
        for module in self._m_modules.values():
            # Validate the pipeline module
            validation = self._validate(module, data_tags)

            if not validation[0]:
                # Break the for loop if a module could not be validated
                break

        return validation

    @typechecked
    def validate_pipeline_module(self,
                                 name: str) -> Tuple[bool, Optional[str]]:
        """
        Method to check if each :class:`~pynpoint.core.dataio.InputPort` of a
        :class:`~pynpoint.core.processing.PypelineModule` with label ``name`` points to an
        existing dataset in the database.

        Parameters
        ----------
        name : str
            Name of the pipeline module instance that will be validated.

        Returns
        -------
        bool
            Validation of the pipeline module.
        str, None
            Pipeline module name in case it is not valid. Returns None if the module was validated.
        """

        self.m_data_storage.open_connection()

        # Create list with all datasets that are stored in the database
        data_tags = list(self.m_data_storage.m_data_bank.keys())

        # Check if the name is included in the internal dictionary with added modules
        if name in self._m_modules:
            # Validate the pipeline module
            validate = self._validate(self._m_modules[name], data_tags)

        else:
            validate = (False, name)

        return validate

    @typechecked
    def run(self) -> None:
        """
        Method for running all pipeline modules that are added to the
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
        Method for running a pipeline module.

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
        Method for reading data from the database.

        Parameters
        ----------
        tag : str
            Database tag.
        data_range : tuple(int, int), None
            Slicing range for the first axis of a dataset. This argument can be used to select a
            subset of images from dataset. The full dataset is read if the argument is set to None.

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
        Method for deleting a dataset and related attributes from the central database. Disk
        space does not seem to free up when using this method.

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
        Method for reading an attribute from the database.

        Parameters
        ----------
        data_tag : str
            Database tag.
        attr_name : str
            Name of the attribute.
        static : bool
            Static (True) or non-static attribute (False).

        Returns
        -------
        StaticAttribute, NonStaticAttribute
            Attribute value. For a static attribute, a single value is returned. For a non-static
            attribute, an array of values is returned.
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
        Method for writing an attribute to the database. Existing values will be overwritten.

        Parameters
        ----------
        data_tag : str
            Database tag.
        attr_name : str
            Name of the attribute.
        attr_value : StaticAttribute, NonStaticAttribute
            Attribute value.
        static : bool
            Static (True) or non-static attribute (False).

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
    def get_tags(self) -> List[str]:
        """
        Method for returning a list with all database tags, except header and configuration tags.

        Returns
        -------
        list(str)
            Database tags.
        """

        self.m_data_storage.open_connection()

        tags = list(self.m_data_storage.m_data_bank.keys())

        selected_tags = []

        for item in tags:
            if item in ['config', 'fits_header'] or item[0:7] == 'header_':
                continue

            selected_tags.append(item)

        self.m_data_storage.close_connection()

        return selected_tags

    @typechecked
    def get_shape(self,
                  tag: str) -> Optional[Tuple[int, ...]]:
        """
        Method for returning the shape of a database entry.

        Parameters
        ----------
        tag : str
            Database tag.

        Returns
        -------
        tuple(int, ...), None
            Shape of the dataset. None is returned if the database tag is not found.
        """

        self.m_data_storage.open_connection()

        if tag in self.m_data_storage.m_data_bank:
            data_shape = self.m_data_storage.m_data_bank[tag].shape
        else:
            data_shape = None

        self.m_data_storage.close_connection()

        return data_shape

    @typechecked
    def list_attributes(self,
                        data_tag: str) -> Dict[str, Union[str, np.float64, np.ndarray]]:
        """
        Method for printing and returning an overview of all attributes of a dataset.

        Parameters
        ----------
        data_tag : str
            Database tag of which the attributes will be extracted.

        Returns
        -------
        dict(str, bool)
            Dictionary with all attributes, both static and non-static.
        """

        print_text = f'Attribute overview of {data_tag}'

        print('\n' + len(print_text) * '-')
        print(print_text)
        print(len(print_text) * '-' + '\n')

        self.m_data_storage.open_connection()

        attributes = {}

        print('Static attributes:')

        for key, value in self.m_data_storage.m_data_bank[data_tag].attrs.items():
            attributes[key] = value
            print(f'\n   - {key} = {value}')

        print('\nNon-static attributes:')

        for key, value in self.m_data_storage.m_data_bank[f'header_{data_tag}'].items():
            attributes[key] = list(value)
            print(f'\n   - {key} = {list(value)}')

        self.m_data_storage.close_connection()

        return attributes
