"""
Modules for accessing data and attributes in the central database.
"""

import os
import warnings

from abc import ABCMeta, abstractmethod

import h5py
import numpy as np


class DataStorage:
    """
    Instances of DataStorage manage to open and close the Pypeline HDF5 databases. They have an
    internal h5py data bank (self.m_data_bank) which gives direct access to the data if the storage
    is open (self.m_open == True).
    """

    def __init__(self,
                 location_in):
        """
        Constructor of a DataStorage instance. It needs the location of the HDF5 file (Pypeline
        database) as input. If the file already exists it is opened and extended, if not a new File
        will be created.

        Parameters
        ----------
        location_in : str
            Location (directory + filename) of the HDF5 database.

        Returns
        -------
        NoneType
            None
        """

        assert (os.path.isdir(os.path.split(location_in)[0])), 'Input directory for DataStorage ' \
                                                               'does not exist - input requested:'\
                                                               ' %s.' % location_in

        self._m_location = location_in
        self.m_data_bank = None
        self.m_open = False

    def open_connection(self):
        """
        Opens the connection to the HDF5 file by opening an old file or creating a new one.

        Returns
        -------
        NoneType
            None
        """

        if not self.m_open:
            self.m_data_bank = h5py.File(self._m_location, mode='a')
            self.m_open = True

    def close_connection(self):
        """
        Closes the connection to the HDF5 file. All entries of the data bank will be stored on the
        hard drive and the memory is cleaned.

        Returns
        -------
        NoneType
            None
        """

        if self.m_open:
            self.m_data_bank.close()
            self.m_data_bank = None
            self.m_open = False


class Port(metaclass=ABCMeta):
    """
    Abstract interface and implementation of common functionality of the InputPort, OutputPort, and
    ConfigPort. Each Port has a internal tag which is its key to a dataset in the DataStorage. If
    for example data is stored under the entry ``im_arr`` in the central data storage only a port
    with the tag (``self._m_tag = im_arr``) can access and change that data. A port knows exactly
    one DataStorage instance, whether it is active or not (``self._m_data_base_active``).
    """

    @abstractmethod
    def __init__(self,
                 tag,
                 data_storage_in=None):
        """
        Abstract constructor of a Port. As input the tag / key is expected which is needed to build
        the connection to the database entry with the same tag / key. It is possible to give the
        Port a DataStorage. If this storage is not given the Pypeline module has to set it or the
        connection needs to be added manually using
        :func:`~pynpoint.core.dataio.Port.set_database_connection`.

        Parameters
        ----------
        tag : str
            Input Tag.
        data_storage_in : pynpoint.core.dataio.DataStorage
            The data storage to which the port is connected.

        Returns
        -------
        NoneType
            None
        """

        assert isinstance(tag, str), 'Port tag needs to be a string.'

        self._m_tag = tag
        self._m_data_storage = data_storage_in
        self._m_data_base_active = False

    @property
    def tag(self):
        """
        Getter for the internal tag (no setter).

        Returns
        -------
        str
            Database tag name.
        """

        return self._m_tag

    def open_port(self):
        """
        Opens the connection to the :class:`~pynpoint.core.dataio.DataStorage` and activates its
        data bank.

        Returns
        -------
        NoneType
            None
        """

        if not self._m_data_base_active:
            self._m_data_storage.open_connection()
            self._m_data_base_active = True

    def close_port(self):
        """
        Closes the connection to the :class:`~pynpoint.core.dataio.DataStorage` and forces it to
        save the data to the hard drive. All data that was accessed using the port is cleaned from
        the memory.

        Returns
        -------
        NoneType
            None
        """

        if self._m_data_base_active:
            self._m_data_storage.close_connection()
            self._m_data_base_active = False

    def set_database_connection(self,
                                data_base_in):
        """
        Sets the internal DataStorage instance.

        Parameters
        ----------
        data_base_in: pynpoint.core.dataio.DataStorage
            The input DataStorage.

        Returns
        -------
        NoneType
            None
        """

        self._m_data_storage = data_base_in


class ConfigPort(Port):
    """
    ConfigPort can be used to read the 'config' tag from a (HDF5) database. This tag contains
    the central settings used by PynPoint, as well as the relevant FITS header keywords. You can
    use a ConfigPort instance to access a single attribute of the dataset using get_attribute().
    """

    def __init__(self,
                 tag,
                 data_storage_in=None):
        """
        Constructor of the ConfigPort class which creates the config port instance which can read
        the settings stored in the central database under the tag `config`. An instance of the
        ConfigPort is created in the constructor of PypelineModule such that the attributes in
        the ConfigPort can be accessed from within all type of modules. For example:

        .. code-block:: python

            memory = self._m_config_port.get_attribute('MEMORY')

        Parameters
        ----------
        tag : str
            The tag name of the port. The port can be used to get data from the dataset with the
            key `config`.
        data_storage_in : pynpoint.core.dataio.DataStorage
            The input DataStorage. It is possible to give the constructor of an ConfigPort a
            DataStorage instance which will link the port to that DataStorage. Usually the
            DataStorage is set later by calling
            :func:`~pynpoint.core.dataio.Port.set_database_connection`.

        Returns
        -------
        NoneType
            None
        """

        super(ConfigPort, self).__init__(tag, data_storage_in)

        if tag != 'config':
            raise ValueError('The tag name of the central configuration should be \'config\'.')

    def _check_status_and_activate(self):
        """
        Internal function which checks if the ConfigPort is ready to use and open it.

        Returns
        -------
        bool
            Returns True if the ConfigPort can be used, False if not.
        """

        if self._m_data_storage is None:
            warnings.warn('ConfigPort can not load data unless a database is connected.')
            status = False

        else:
            if not self._m_data_base_active:
                self.open_port()

            status = True

        return status

    def _check_if_data_exists(self):
        """
        Internal function which checks if data exists for the 'config' tag.

        Returns
        -------
        bool
            Returns True if data exists, False if not.
        """

        return 'config' in self._m_data_storage.m_data_bank

    def _check_error_cases(self):
        """'
        Internal function which checks the error cases.
        """

        if not self._check_status_and_activate():
            status = False

        elif self._check_if_data_exists() is False:
            warnings.warn('No data under the tag which is linked by the ConfigPort.')
            status = False

        else:
            status = True

        return status

    def get_attribute(self,
                      name):
        """
        Returns a static attribute which is connected to the dataset of the ConfigPort.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        str, float, or int
            The attribute value. Returns None if the attribute does not exist.
        """

        if not self._check_error_cases():
            attr_val = None

        elif name in self._m_data_storage.m_data_bank['config'].attrs:
            attr_val = self._m_data_storage.m_data_bank['config'].attrs[name]

        else:
            warnings.warn(f'The attribute \'{name}\' was not found.')
            attr_val = None

        return attr_val


class InputPort(Port):
    """
    InputPorts can be used to read datasets with a specific tag from the HDF5 database. This type
    of port can be used to access:

        * A complete dataset using the get_all() method.
        * A single attribute of the dataset using get_attribute().
        * All attributes of the dataset using get_all_static_attributes() and
          get_all_non_static_attributes().
        * A part of a dataset using slicing. For example:

        .. code-block:: python

            in_port = InputPort('tag')
            data = in_port[0, :, :] # returns the first 2D image of a 3D image stack.

    (More information about how 1D, 2D, and 3D data is organized can be found in the documentation
    of OutputPort (:func:`~pynpoint.core.dataio.OutputPort.append` and
    :func:`~pynpoint.core.dataio.OutputPort.set_all`)

    InputPorts can load two types of attributes which give additional information about
    a dataset the port is linked to:

        * Static attributes: contain global information about a dataset which is not changing
          through a dataset in the database (e.g. the instrument name or pixel scale).
        * Non-static attributes: contain information which changes for different parts of the
          dataset (e.g. the parallactic angles or dithering positions).
    """

    def __init__(self,
                 tag,
                 data_storage_in=None):
        """
        Constructor of InputPort. An input port can read data from the central database under the
        key `tag`. Instances of InputPort should not be created manually inside a PypelineModule
        but should be created with the add_input_port() function.

        Parameters
        ----------
        tag : str
            The tag of the port. The port can be used in order to get data from the dataset with
            the key `tag`.
        data_storage_in : pynpoint.core.dataio.DataStorage
            It is possible to give the constructor of an InputPort a DataStorage instance which
            will link the port to that DataStorage. Usually the DataStorage is set later by calling
            :func:`~pynpoint.core.dataio.Port.set_database_connection`.

        Returns
        -------
        NoneType
            None
        """

        super(InputPort, self).__init__(tag, data_storage_in)

        if tag == 'config':
            raise ValueError('The tag name \'config\' is reserved for the central configuration '
                             'of PynPoint.')

        if tag == 'fits_header':
            raise ValueError('The tag name \'fits_header\' is reserved for storage of the FITS '
                             'headers.')

    def _check_status_and_activate(self):
        """
        Internal function which checks if the InputPort is ready to use and open it.

        Returns
        -------
        bool
            Returns True if the InputPort can be used, False if not.
        """

        if self._m_data_storage is None:
            warnings.warn('InputPort can not load data unless a database is connected.')
            status = False

        else:
            status = True

            if not self._m_data_base_active:
                self.open_port()

        return status

    def _check_if_data_exists(self):
        """
        Internal function which checks if data exists for the Port specific tag.

        Returns
        -------
        bool
            Returns True if data exists, False if not.
        """

        return self._m_tag in self._m_data_storage.m_data_bank

    def _check_error_cases(self):

        if not self._check_status_and_activate():
            status = False

        elif self._check_if_data_exists() is False:
            warnings.warn('No data under the tag which is linked by the InputPort.')
            status = False

        else:
            status = True

        return status

    def __getitem__(self,
                    item):
        """
        Internal function which handles the data access using slicing. See class documentation for a
        example (:class:`~pynpoint.core.dataio.InputPort`). None if the data does not exist.

        Parameters
        ----------
        item : tuple
            Slicing parameter.

        Returns
        -------
        numpy.ndarray
            The selected data. Returns None if no data exists under the tag of thePort.
        """

        if not self._check_error_cases():
            data = None

        else:
            data = self._m_data_storage.m_data_bank[self._m_tag][item]

        return data

    def get_shape(self):
        """
        Returns the shape of the dataset the port is linked to. This can be useful if you need the
        shape without loading the whole data.

        Returns
        -------
        tuple(int, )
            Shape of the dataset. Returns None if the dataset does not exist.
        """

        if not self._check_error_cases():
            data_shape = None

        else:
            self.open_port()
            data_shape = self._m_data_storage.m_data_bank[self._m_tag].shape

        return data_shape

    def get_ndim(self):
        """
        Returns the number of dimensions of the dataset the port is linked to.

        Returns
        -------
        int
            Number of dimensions of the dataset. Returns None if the dataset does not exist.
        """

        if not self._check_error_cases():
            ndim = None

        else:
            self.open_port()
            ndim = self._m_data_storage.m_data_bank[self._m_tag].ndim

        return ndim

    def get_all(self):
        """
        Returns the whole dataset stored in the data bank under the tag of the Port. Be careful
        using this function for loading large datasets. The data type is inferred from the data
        with numpy.asarray. A 32 bit array will be returned in case the input data is a
        combination of float32 and float64 arrays.

        Returns
        -------
        numpy.ndarray
            The full dataset. Returns None if the data does not exist.
        """

        if not self._check_error_cases():
            data = None

        else:
            data = np.asarray(self._m_data_storage.m_data_bank[self._m_tag][...])

        return data

    def get_attribute(self,
                      name):
        """
        Returns an attribute which is connected to the dataset of the port. The function can return
        static and non-static attributes (static attributes have priority). More information about
        static and non-static attributes can be found in the class documentation of
        :class:`~pynpoint.core.dataio.InputPort`.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        str, float, int, or numpy.ndarray
            The attribute value. Returns None if the attribute does not exist.
        """

        if not self._check_error_cases():
            attr_val = None

        else:
            if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
                # static attribute
                attr_val = self._m_data_storage.m_data_bank[self._m_tag].attrs[name]

            elif 'header_' + self._m_tag + '/' + name in self._m_data_storage.m_data_bank:
                # non-static attribute
                attribute = 'header_' + self._m_tag + '/' + name
                attr_val = np.asarray(self._m_data_storage.m_data_bank[attribute][...])

            else:
                warnings.warn(f'The attribute \'{name}\' was not found.')
                attr_val = None

        return attr_val

    def get_all_static_attributes(self):
        """
        Get all static attributes of the dataset which are linked to the Port tag.

        Returns
        -------
        dict
            Dictionary of all attributes, as `{attr_name:attr_value}`.
        """

        if not self._check_error_cases():
            attr_dict = None

        else:
            attr_dict = dict(self._m_data_storage.m_data_bank[self._m_tag].attrs)

        return attr_dict

    def get_all_non_static_attributes(self):
        """
        Returns a list of all non-static attribute keys.  More information about
        static and non-static attributes can be found in the class documentation of
        :class:`~pynpoint.core.dataio.InputPort`.

        Returns
        -------
        list(str, )
            List of all existing non-static attribute keys.
        """

        if not self._check_error_cases():
            attr_key = None

        else:
            attr_key = []

            if 'header_' + self._m_tag + '/' in self._m_data_storage.m_data_bank:
                for key in self._m_data_storage.m_data_bank['header_' + self._m_tag + '/']:
                    attr_key.append(key)

            else:
                attr_key = None

        return attr_key


class OutputPort(Port):
    """
    Output ports can be used to save results under a given tag to the HDF5 DataStorage. An instance
    of OutputPort with self.tag=`tag` can store data under the key `tag` by using one of the
    following methods:

        * set_all(...) - replaces and sets the whole dataset
        * append(...) - appends data to the existing data set. For more information see
          function documentation (:func:`~pynpoint.core.dataio.OutputPort.append`).
        * slicing - sets a part of the actual dataset. Example:

        .. code-block:: python

            out_port = OutputPort('Some_tag')
            data = np.ones(200, 200) # 2D image filled with ones
            out_port[0,:,:] = data # Sets the first 2D image of a 3D image stack

        * add_attribute(...) - modifies or creates a attribute of the dataset
        * del_attribute(...) - deletes a attribute
        * del_all_attributes(...) - deletes all attributes
        * append_attribute_data(...) - appends information to non-static attributes. See
          add_attribute() (:func:`~pynpoint.core.dataio.OutputPort.add_attribute`) for more
          information about static and non-static attributes.
        * check_static_attribute(...) - checks if a static attribute exists and if it is equal to a
          given value
        * other functions listed below

    For more information about how data is organized inside the central database have a look at the
    function documentation of the function :func:`~pynpoint.core.dataio.OutputPort.set_all` and
    :func:`~pynpoint.core.dataio.OutputPort.append`.

    Furthermore it is possible to deactivate a OutputPort to stop him saving data.
    """

    def __init__(self,
                 tag,
                 data_storage_in=None,
                 activate_init=True):
        """
        Constructor of the OutputPort class which creates an output port instance which can write
        data to the the central database under the tag `tag`. If you write a PypelineModule you
        should not create instances manually! Use the add_output_port() function instead.

        Parameters
        ----------
        tag : str
            The tag of the port. The port can be used in order to write data to the dataset with
            the key = `tag`.
        data_storage_in : pynpoint.core.dataio.DataStorage
            It is possible to give the constructor of an OutputPort a DataStorage instance which
            will link the port to that DataStorage. Usually the DataStorage is set later by calling
            :func:`~pynpoint.core.dataio.Port.set_database_connection`.

        Returns
        -------
        NoneType
            None
        """

        super(OutputPort, self).__init__(tag, data_storage_in)

        self.m_activate = activate_init

        if tag == 'config':
            raise ValueError('The tag name \'config\' is reserved for the central configuration '
                             'of PynPoint.')

        if tag == 'fits_header':
            raise ValueError('The tag name \'fits_header\' is reserved for storage of the FITS '
                             'headers.')

    def _check_status_and_activate(self):
        """
        Internal function which checks if the OutputPort is ready to use and open it.

        Returns
        -------
        :return: Returns True if the OutputPort can be used, False if not.
        :rtype: bool
        """

        if not self.m_activate:
            status = False

        elif self._m_data_storage is None:
            warnings.warn('OutputPort can not store data unless a database is connected.')
            status = False

        else:
            if not self._m_data_base_active:
                self.open_port()

            status = True

        return status

    def _initialize_database(self,
                             first_data,
                             tag,
                             data_dim=None):
        """
        Internal function which is used to initialize the HDF5 database.

        Parameters
        ----------
        first_data : numpy.ndarray
            The initial data.
        tag : str
            Database tag.
        data_dim : int
            Number of dimensions. The dimensions of *first_data* is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        def _ndim_check(data_dim, first_dim):
            if first_dim > 3 or first_dim < 1:
                raise ValueError('Output port can only save numpy arrays from 1D to 3D. Use Port '
                                 'attributes to save as int, float, or string.')

            if data_dim > 3 or data_dim < 1:
                raise ValueError('The data dimensions should be 1D, 2D, or 3D.')

            if data_dim < first_dim:
                raise ValueError('The dimensions of the data should be equal to or larger than the '
                                 'dimensions of the input data.')

            if data_dim == 3 and first_dim == 1:
                raise ValueError('Cannot initialize 1D data in 3D data container.')

        first_data = np.asarray(first_data)

        if data_dim is None:
            data_dim = first_data.ndim

        _ndim_check(data_dim, first_data.ndim)

        if data_dim == first_data.ndim:
            if first_data.ndim == 1:  # case (1,1)
                data_shape = (None, )

            elif first_data.ndim == 2:  # case (2,2)
                data_shape = (None, first_data.shape[1])

            elif first_data.ndim == 3:  # case (3,3)
                data_shape = (None, first_data.shape[1], first_data.shape[2])

        else:
            if data_dim == 2:  # case (2,1)
                data_shape = (None, first_data.shape[0])
                first_data = first_data[np.newaxis, :]

            elif data_dim == 3:  # case (3,2)
                data_shape = (None, first_data.shape[0], first_data.shape[1])
                first_data = first_data[np.newaxis, :, :]

        if first_data.size == 0:
            warnings.warn(f'The new dataset that is stored under the tag name \'{tag}\' is empty.')

        else:
            if isinstance(first_data[0], str):
                first_data = np.array(first_data, dtype='|S')

        self._m_data_storage.m_data_bank.create_dataset(tag,
                                                        data=first_data,
                                                        maxshape=data_shape)

    def _set_all_key(self,
                     tag,
                     data,
                     data_dim=None,
                     keep_attributes=False):
        """
        Internal function which sets the values of a dataset under the *tag* name in the database.
        If old data exists it will be overwritten. This function is used by
        :func:`~pynpoint.core.dataio.OutputPort.set_all` and for setting non-static attributes.

        Parameters
        ----------
        tag : str
            Database tag of the data that will be modified.
        data : numpy.ndarray
            The data that will be stored and replace any old data.
        data_dim : int
            Number of dimension of the data.
        keep_attributes : bool
            Keep all static attributes of the dataset if set to True. Non-static attributes will be
            kept anyway so not needed for setting non-static attributes.

        Returns
        -------
        NoneType
            None
        """

        tmp_attributes = {}

        # check if database entry is new...
        if tag in self._m_data_storage.m_data_bank:
            # NO -> database entry exists
            if keep_attributes:
                # we have to copy all attributes since deepcopy is not supported
                for key, value in self._m_data_storage.m_data_bank[tag].attrs.items():
                    tmp_attributes[key] = value

            # remove database entry
            del self._m_data_storage.m_data_bank[tag]

        # make new database entry
        self._initialize_database(data, tag, data_dim=data_dim)

        if keep_attributes:
            for key, value in tmp_attributes.items():
                self._m_data_storage.m_data_bank[tag].attrs[key] = value

    def _append_key(self,
                    tag,
                    data,
                    data_dim=None,
                    force=False):
        """
        Internal function for appending data to a dataset or appending non-static attribute
        information. See :func:`~pynpoint.core.dataio.OutputPort.append` for more information.

        Parameters
        ----------
        tag : str
            Database tag of the data that will be modified.
        data : numpy.ndarray
            The data that will be stored and replace any old data.
        data_dim : int
            Number of dimension of the data.
        force : bool
            The existing data will be overwritten if shape or type does not match.

        Returns
        -------
        NoneType
            None
        """

        # check if database entry is new...
        if tag not in self._m_data_storage.m_data_bank:
            # YES -> database entry is new
            self._initialize_database(data, tag, data_dim=data_dim)
            return None

        # NO -> database entry exists
        # check if the existing data has the same dim and datatype
        tmp_shape = self._m_data_storage.m_data_bank[tag].shape
        tmp_dim = len(tmp_shape)

        if data_dim is None:
            data_dim = tmp_dim

        # convert input data to numpy array
        data = np.asarray(data)

        # if the dimension offset is 1 add that dimension (e.g. save 2D image in 3D image stack)
        if data.ndim + 1 == data_dim:

            if data_dim == 2:
                data = data[np.newaxis, :]
            elif data_dim == 3:
                data = data[np.newaxis, :, :]

        def _type_check():
            check_result = False

            if tmp_dim == data.ndim:

                if tmp_dim == 1:
                    check_result = True
                elif tmp_dim == 2:
                    check_result = tmp_shape[1] == data.shape[1]
                elif tmp_dim == 3:
                    check_result = (tmp_shape[1] == data.shape[1]) and \
                                   (tmp_shape[2] == data.shape[2])

            return check_result

        if _type_check():
            # YES -> dim and type match
            # we always append in axis one independent of the dimension
            # 1D case

            if data.size == 0:
                warnings.warn(f'The dataset that is appended under the tag name \'{tag}\' '
                              f'is empty.')

            else:
                if isinstance(data[0], str):
                    data = np.array(data, dtype='|S')

            self._m_data_storage.m_data_bank[tag].resize(tmp_shape[0] + data.shape[0], axis=0)
            self._m_data_storage.m_data_bank[tag][tmp_shape[0]::] = data

            return None

        # NO -> shape or type is different
        # Check force
        if force:
            # YES -> Force is true
            self._set_all_key(tag, data=data)
            return None

        # NO -> Error message
        raise ValueError(f'The port tag \'{self._m_tag}\' is already used with a different data '
                         f'type. The \'force\' parameter can be used to replace the tag.')

    def __setitem__(self,
                    key,
                    value):
        """
        Internal function needed to change data using slicing. See class documentation for an
        example (:class:`~pynpoint.core.dataio.OutputPort`).

        Parameters
        ----------
        key : slice
            Index slice to be changed.
        value : numpy.ndarray
            New data.

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate():
            self._m_data_storage.m_data_bank[self._m_tag][key] = value

    def del_all_data(self):
        """
        Delete all data belonging to the database tag.
        """

        if self._check_status_and_activate():

            if self._m_tag in self._m_data_storage.m_data_bank:
                del self._m_data_storage.m_data_bank[self._m_tag]

    def set_all(self,
                data,
                data_dim=None,
                keep_attributes=False):
        """
        Set the data in the database by replacing all old values with the values of the input data.
        If no old values exists the data is just stored. Since it is not possible to change the
        number of dimensions of a data set later in the processing history one can choose a
        dimension different to the input data. The following cases are implemented:

            * (#dimension of the first input data#, #desired data_dim#)
            * (1, 1) 1D input or single value will be stored as list in HDF5
            * (1, 2) 1D input, but 2D array stored inside (i.e. a list of lists with a fixed size).
            * (2, 2) 2D input (single image) and 2D array stored inside (i.e. a list of lists with a
              fixed size).
            * (2, 3) 2D input (single image) but 3D array stored inside (i.e. a stack of images with
              a fixed size).
            * (3, 3) 3D input and 3D array stored inside (i.e. a stack of images with a fixed size).

        For 2D and 3D data the first dimension always represents the list / stack (variable size)
        while the second (or third) dimension has a fixed size. After creation it is possible to
        extend a data set using :func:`~pynpoint.core.dataio.OutputPort.append` along the first
        dimension.

        **Example 1:**

        Input 2D array with size (200, 200). Desired dimension 3D. The result is a 3D dataset with
        the dimension (1, 200, 200). It is possible to append other images with the size (200, 200)
        or other stacks of images with the size (:, 200, 200).

        **Example 2:**

        Input 2D array with size (200, 200). Desired dimension 2D. The result is a 2D dataset with
        the dimension (200, 200). It is possible to append other list with the length 200
        or other stacks of lines with the size (:, 200). However it is not possible to append other
        2D images along a third dimension.

        Parameters
        ----------
        data : numpy.ndarray
            The data to be saved.
        data_dim : int
            Number of data dimensions. The dimension of the *first_data* is used if set to None.
        keep_attributes : bool
            All attributes of the old dataset will remain the same if set to True.

        Returns
        -------
        NoneType
            None
        """

        data = np.asarray(data)

        if self._check_status_and_activate():

            self._set_all_key(tag=self._m_tag,
                              data=data,
                              data_dim=data_dim,
                              keep_attributes=keep_attributes)

    def append(self,
               data,
               data_dim=None,
               force=False):
        """
        Appends data to an existing dataset with the tag of the Port along the first
        dimension. If no data exists with the tag of the Port a new data set is created.
        For more information about how the dimensions are organized see documentation of
        the function :func:`~pynpoint.core.dataio.OutputPort.set_all`. Note it is not possible to
        append data with a different shape or data type to the existing dataset.

        **Example:** An internal data set is 3D (storing a stack of 2D images) with shape
        (233, 300, 300) which mean it contains 233 images with a resolution of 300 x 300 pixel.
        Thus it is only possible to extend along the first dimension by appending new images with
        a size of (300, 300) or by appending a stack of images (:, 300, 300). Everything else will
        raise exceptions.

        It is possible to force the function to overwrite the existing data set if and only if the
        shape or type of the input data does not match the existing data. **Warning**: This can
        delete the existing data.

        Parameters
        ----------
        data : numpy.ndarray
            The data which will be appended.
        data_dim : int
            Number of data dimensions used if a new data set is created. The dimension of the
            *data* is used if set to None.
        force : bool
            The existing data will be overwritten if shape or type does not match if set to True.

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate():

            self._append_key(self._m_tag,
                             data=data,
                             data_dim=data_dim,
                             force=force)

    def activate(self):
        """
        Activates the port. A non activated port will not save data.

        Returns
        -------
        NoneType
            None
        """

        self.m_activate = True

    def deactivate(self):
        """
        Deactivates the port. A non activated port will not save data.

        Returns
        -------
        NoneType
            None
        """

        self.m_activate = False

    def add_attribute(self,
                      name,
                      value,
                      static=True):
        """
        Adds an attribute to the dataset of the Port with the attribute name = `name` and the
        value = `value`. If the attribute already exists it will be overwritten. Two different
        types of attributes are supported:

            1. **static attributes**:
               Contain a single value or name (e.g. The name of the used Instrument).
            2. **non-static attributes**:
               Contain a dataset which is connected to the actual data set (e.g. Instrument
               temperature). It is possible to append additional information to non-static
               attributes later (:func:`~pynpoint.core.dataio.OutputPort.append_attribute_data`).
               This is not supported by static attributes.

        Static and non-static attributes are stored in a different way using the HDF5 file format.
        Static attributes will be direct attributes while non-static attributes are stored in a
        group with the name *header_* + name of the dataset.

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : str, float, or int
            Value of the attribute.
        static : bool
            Indicate if the attribute is static (True) or non-static (False).

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate():

            if self._m_tag not in self._m_data_storage.m_data_bank:
                warnings.warn('Can not store attribute if data tag does not exist.')

            else:
                if static:
                    self._m_data_storage.m_data_bank[self._m_tag].attrs[name] = value

                else:
                    self._set_all_key(tag=('header_' + self._m_tag + '/' + name),
                                      data=np.asarray(value))

    def append_attribute_data(self,
                              name,
                              value):
        """
        Function which appends a single data value to non-static attributes.

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : str, float, or int
            Value which will be appended to the attribute dataset.

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate():

            self._append_key(tag=('header_' + self._m_tag + '/' + name),
                             data=np.asarray([value, ]))

    def copy_attributes(self,
                        input_port):
        """
        Copies all static and non-static attributes from a given InputPort. Attributes which already
        exist will be overwritten. Non-static attributes will be linked not copied. If the InputPort
        tag = OutputPort tag (self.tag) nothing will be changed. Use this function in all modules
        to keep the header information.

        Parameters
        ----------
        input_port : pynpoint.core.dataio.InputPort
            The InputPort with the header information.

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate() and input_port.tag != self._m_tag:

            # link non-static attributes
            if 'header_' + input_port.tag + '/' in self._m_data_storage.m_data_bank:

                for attr_name, attr_data in self._m_data_storage\
                        .m_data_bank['header_' + input_port.tag + '/'].items():

                    database_name = 'header_'+self._m_tag+'/'+attr_name

                    # overwrite existing header information in the database
                    if database_name in self._m_data_storage.m_data_bank:
                        del self._m_data_storage.m_data_bank[database_name]

                    self._m_data_storage.m_data_bank[database_name] = attr_data

            # copy static attributes
            attributes = input_port.get_all_static_attributes()
            for attr_name, attr_val in attributes.items():
                self.add_attribute(attr_name, attr_val)

            self._m_data_storage.m_data_bank.flush()

    def del_attribute(self,
                      name):
        """
        Deletes the attribute of the dataset with the given name. Finds and removes static and
        non-static attributes.

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate():

            # check if attribute is static
            if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
                del self._m_data_storage.m_data_bank[self._m_tag].attrs[name]

            elif 'header_'+self._m_tag+'/'+name in self._m_data_storage.m_data_bank:
                # remove non-static attribute
                del self._m_data_storage.m_data_bank[('header_' + self._m_tag + '/' + name)]

            else:
                warnings.warn(f'Attribute \'{name}\' does not exist and could not be deleted.')

    def del_all_attributes(self):
        """
        Deletes all static and non-static attributes of the dataset.

        Returns
        -------
        NoneType
            None
        """

        if self._check_status_and_activate():

            # static attributes
            if self._m_tag in self._m_data_storage.m_data_bank:
                self._m_data_storage.m_data_bank[self._m_tag].attrs.clear()

            # non-static attributes
            if 'header_' + self._m_tag + '/' in self._m_data_storage.m_data_bank:
                del self._m_data_storage.m_data_bank[('header_' + self._m_tag + '/')]

    def check_static_attribute(self,
                               name,
                               comparison_value):
        """
        Checks if a static attribute exists and if it is equal to a comparison value.

        Parameters
        ----------
        name : str
            Name of the static attribute.
        comparison_value : str, float, or int
            Comparison value.

        Returns
        -------
        int
            Status: 1 if the static attribute does not exist, 0 if the static attribute exists
            and is equal, and -1 if the static attribute exists but is not equal.
        """

        if not self._check_status_and_activate():
            return None

        if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            if self._m_data_storage.m_data_bank[self._m_tag].attrs[name] == comparison_value:
                return 0

            return -1

        return 1

    def check_non_static_attribute(self,
                                   name,
                                   comparison_value):
        """
        Checks if a non-static attribute exists and if it is equal to a comparison value.

        Parameters
        ----------
        name : str
            Name of the non-static attribute.
        comparison_value : numpy.ndarray
            Comparison values

        Returns
        -------
        int
            Status: 1 if the non-static attribute does not exist, 0 if the non-static attribute
            exists and is equal, and -1 if the non-static attribute exists but is not equal.
        """

        if not self._check_status_and_activate():
            return None

        group = 'header_' + self._m_tag + '/'

        if group in self._m_data_storage.m_data_bank:
            if name in self._m_data_storage.m_data_bank[group]:
                if np.array_equal(self._m_data_storage.m_data_bank[group+name][:],
                                  comparison_value):
                    return 0

                return -1

            return 1

        return 1

    def add_history(self,
                    module,
                    history):
        """
        Adds an attribute with history information about the pipeline module.

        Parameters
        ----------
        module : str
            Name of the pipeline module which was executed.
        history : str
            History information.

        Returns
        -------
        NoneType
            None
        """

        self.add_attribute('History: ' + module, history)

    def flush(self):
        """
        Forces the :class:`~pynpoint.core.dataio.DataStorage` to save all data from the memory to
        the hard drive without closing the :class:`~pynpoint.core.dataio.OutputPort`.

        Returns
        -------
        NoneType
            None
        """

        self._m_data_storage.m_data_bank.flush()
