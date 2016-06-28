"""
Modules to save and load data as / from a .hdf5 database.
"""
from abc import ABCMeta, abstractmethod
import warnings
import copy
import h5py
import numpy as np

# TODO decorator for check status of Port


class DataStorage(object):
    """
    Instances of DataStorage manage the opening and closing of .hdf5 databases. They have a internal
    data bank (self.m_data_bank) which gives direct access to the data bank if the storage is open
    (self.m_open == True). Furthermore it knows the location of the .hdf5 file.
    """

    def __init__(self,
                 location_in):
        """
        Constructor of a DataStorage instance. Needs the location of the .hdf5 file as input. If the
        file already exists it is opened and extended if needed, if not a new File is created.

        :param location_in: Location (directory + name) of the .hdf5 data bank
        :type location_in: String
        :return: None
        """
        self._m_location = location_in
        self.m_data_bank = None
        self.m_open = False

    def open_connection(self):
        """
        Opens the connection to the .hdf5 file by opening / creating it.

        :return: None
        """
        if self.m_open:
            return
        self.m_data_bank = h5py.File(self._m_location, mode='a')
        self.m_open = True

    def close_connection(self):
        """
        Closes the connection to the .hdf5 file. All entries of the data bank will be stored on the
        hard drive and the memory is cleaned.

        :return: None
        """
        if not self.m_open:
            return
        self.m_data_bank.close()
        self.m_open = False


class Port:
    """
    Abstract interface and implementation of common functionality of the Input and Output Port.
    Each Port has a internal tag which is the its key to the database. If for example data is stored
    under the entry "im_arr" in the central data storage only a port with the tag (self._m_tag) can
    access or change that data. Furthermore a port knows exact one Data Storage instance and if it
    is active or not (self._m_data_base_active).
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 tag,
                 data_storage_in=None):
        """
        Abstract constructor of a Port. As input the tag / key is expected with is needed to build
        the connection to the database entry with the same tag / key. It is possible to give the
        Port a data storage. If this storage is not given the Pypeline module has to set it or the
        connection needs to be added manually.

        :param tag: Input Tag
        :type tag: String
        :param data_storage_in: The data storage the port is connected with
        :type data_storage_in: DataStorage
        :return: None
        """
        assert (isinstance(tag, str)), "Error: Port tags need to be strings."
        self._m_tag = tag
        self._m_data_storage = data_storage_in
        self._m_data_base_active = False

    def close_port(self):
        """
        Closes the connection to the Data Storage and force it to save the data to the hard drive.
        All data that was accessed using the port is cleaned from the memory.

        :return: None
        """
        if self._m_data_base_active:
            self._m_data_storage.close_connection()
            self._m_data_base_active = False

    def open_port(self):
        """
        Opens the connection to the Data Storage and activates its data bank.

        :return: None
        """
        if not self._m_data_base_active:
            self._m_data_storage.open_connection()
            self._m_data_base_active = True

    def set_database_connection(self,
                                data_base_in):
        """
        Sets the internal Data Storage instance.

        :param data_base_in: The inout data storage
        :type data_base_in: DataStorage
        :return: None
        """
        self._m_data_storage = data_base_in


class InputPort(Port):
    """
    InputPorts can be used to read data with a specific tag from a (.hdf5) Data Storage. With a
    input port one can access:
        -> the complete actual dataset using the get_all() method
        -> a single attribute of the dataset using get_attribute()
        -> all attributes of the dataset using get_all_attributes()
        -> a part of the actual dataset using slicing
            for example:
            tmp_in_port = InputPort("Some_tag")
            data = tmp_in_port[0,:,:] # returns the first 2D image of a 3D image stack
            (See more information about how 1D 2D and 3D data is organized in the documentation of
            Output Port (append, set_all).
    Note that all data accessed using a input port in copied to the memory to avoid InputPorts
    changing data of the database.
    """

    def __init__(self,
                 tag,
                 data_storage_in=None):
        """
        Constructor of the InputPort class crating a input port instance with the tag self._m_tag.
        This function is just calling the super constructor (i.e. __init__() of Port).

        :param tag: Input Tag
        :type tag: Sting
        :return: None
        """
        super(InputPort, self).__init__(tag, data_storage_in)

    def _check_status_and_activate(self):
        """
        Internal function which checks if the port is ready to use and open it.

        :return: Returns True if the Port can be used, False if not.
        :rtype: Boolean
        """
        if self._m_data_storage is None:
            warnings.warn("Port can not load data unless a database is connected")
            return False

        if not self._m_data_base_active:
            self.open_port()

        return True

    def __getitem__(self, item):
        """
        Internal function needed to access data using slicing. See class documentation for an
        example.
        Note this function returns a copy of the data and not the actual data bank data.
        Changing this copy will not change the data bank data.

        :param item: Slicing input
        :return: A array of the ordered data.
        """

        if not self._check_status_and_activate():
            return

        result = self._m_data_storage.m_data_bank[self._m_tag][item]

        if isinstance(result, bytearray):
            return np.asarray(result, dtype=np.float64)

        return result

    def get_all(self):
        """
        Returns the whole data set stored in the data bank under the Port tag (self._m_tag).
        Note this function returns a copy of the data and not the actual data bank data.
        Changing this copy will not change the data bank data.

        :return: The data set
        """
        self._check_status_and_activate()
        return np.asarray(self._m_data_storage.m_data_bank[self._m_tag], dtype=np.float64)

    def get_attribute(self,
                      name):
        """
        first looking for statics
        :param name:
        :return:
        """
        # TODO Documentation

        self._check_status_and_activate()

        # check if attribute is static
        if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            # item unpacks numpy types to python types hdf5 only uses numpy types
            return self._m_data_storage.m_data_bank[self._m_tag].attrs[name].item()
        elif ("header_" + self._m_tag + "/" + name) in self._m_data_storage.m_data_bank:
            return np.asarray(self._m_data_storage.m_data_bank
                              [("header_" + self._m_tag + "/" + name)])
        else:
            return None

    def get_all_static_attributes(self):
        """
        Returns all attributes of the data set stored in the data bank under the Port tag
        (self._m_tag) as dictionary {attr_name: attr_value}.

        :return: dictionary of all attributes
        """
        self._check_status_and_activate()
        return self._m_data_storage.m_data_bank[self._m_tag].attrs

    def get_all_non_static_attributes(self):
        result = []

        # check if header Group exists
        if ("header_" + self._m_tag + "/") in self._m_data_storage.m_data_bank:
            for key in self._m_data_storage.m_data_bank["header_" + self._m_tag + "/"]:
                result.append(key)
            return result
        else:
            return None


class OutputPort(Port):
    """
    Output Ports can be used to save data with a specific tag to a (.hdf5) Data Storage. The
    following methods allow different ways of saving data:
        -> replace and set the whole data set with set_all(...)
        -> append data to the existing data set using append(...). For more information see function
           documentation.
        -> set or create a single attribute of the data set using add_attribute()
        -> delete a single attribute using del_attribute(...)
        -> delete all attributes using del_all_attributes()
        -> set a part of the actual data set using slicing
            for example:
            tmp_in_port = OutputPort("Some_tag")
            data = np.ones(200, 200) # 2D image filled with ones
            tmp_in_port[0,:,:] = data # Sets the first 2D image of a 3D image stack
            For more information about how data is organized inside and can be modified see function
            documentation of the function set_all() and append().
    It is possible to deactivate a Output Port to stop him saving data.
    """

    def __init__(self,
                 tag,
                 data_storage_in=None,
                 activate_init=True):
        """
        Constructor for a OutputPort class instance with the tag self._m_tag.

        :param tag: Tag of the Port
        :type tag: String
        :param activate_init: start activation status. If False the Port will not save data until
            it is activated.
        :type activate_init: Boolean
        :return: None
        """

        super(OutputPort, self).__init__(tag, data_storage_in)
        self.m_activate = activate_init

    # internal functions
    def _check_status_and_activate(self):
        """
        Internal function which checks if the port is ready to use and open it.

        :return: Returns True if the Port can be used, False if not.
        :rtype: Boolean
        """
        if not self.m_activate:
            return False

        if self._m_data_storage is None:
            warnings.warn("Port can not store data unless a database is connected")
            return False

        if not self._m_data_base_active:
            self.open_port()

        return True

    def _initialize_database_entry(self,
                                   first_data,
                                   tag,
                                   data_dim=None):
        """
        Internal function which is used to create (initialize) a data base in .hdf5 data base. Since
        it is not possible to change the number of dimensions later in the processing history the
        right dimension has to be specified during the creation of the data set. The function needs
        first data as input and optional a desired dimension of the internal data set in the .hdf5
        file. The following options are available:
        - (#dimension of the first input data#, #desired data_dim#)
        -> (1, 1) 1D input or single value stored as list in .hdf5
        -> (1, 2) 1D input, but 2D array stored inside (i.e. a list of lists with a fixed size).
        -> (2, 2) 2D input and 2D array stored inside (i.e. a list of lists with a fixed size).
        -> (2, 3) 2D input but 3D array stored inside (i.e. a stack of images with a fixed size).
        -> (3, 3) 3D input and 3D array stored inside (i.e. a stack of images with a fixed size).

        For 2D and 3D data the first dimension always represents the list / stack (variable size)
        while the second (or third) dimension has a fixed size.
            Example: Input 2D array with size (200, 200) Desired dimension 3D
                The result is a 3D data set with the dimension (1, 200, 200). It is possible to
                append other images with the size (200, 200) or other stacks of images with the
                size (:, 200, 200).

        After creation it is possible to extend a data set using append(...) along the first
        dimension.

        :param first_data: The initial data
        :type first_data: bytearray
        :param data_dim: number of desired dimensions. If None the dimension of the first_data is
            used.
        :return: None
        """

        # check Error cases
        if isinstance(first_data, bytearray) or first_data.ndim > 3 or first_data.ndim < 1:
            raise ValueError('Output port can only save numpy arrays from 1D to 3D. If you want '
                             'to save a int, float, string ... use Port attributes instead.')

        if data_dim is None:
            data_dim = first_data.ndim

        if data_dim > 3 or data_dim < 1:
            raise ValueError('data_dim needs to be in [1,3].')

        if data_dim < first_data.ndim:
            raise ValueError('data_dim needs to have at least the same dim as the input.')

        if data_dim == 3 and first_data.ndim == 1:
            raise ValueError('Cannot initialize 1D data in 3D data container.')

        # if no data_dim is given check the input data
        if data_dim == first_data.ndim:
            if first_data.ndim == 1:  # case (1,1)
                data_shape = (None,)
            elif first_data.ndim == 2:  # case (2,2)
                data_shape = (None, first_data.shape[1])
            elif first_data.ndim == 3:  # case (3,3)
                data_shape = (None, first_data.shape[1], first_data.shape[2])
            else:
                raise ValueError('Input shape not supported')  # this case should never be reached
        else:
            if data_dim == 2:  # case (2, 1)
                data_shape = (None, first_data.shape[0])
                first_data = first_data[np.newaxis, :]
            elif data_dim == 3:  # case (3, 2)
                data_shape = (None, first_data.shape[0], first_data.shape[1])
                first_data = first_data[np.newaxis, :, :]
            else:
                raise ValueError('Input shape not supported')  # this case should never be reached

        self._m_data_storage.m_data_bank.create_dataset(tag,
                                                        data=first_data,
                                                        maxshape=data_shape)

    def _set_all_key(self,
                     tag,
                     data,
                     data_dim=None,
                     keep_attributes = False):

        tmp_attributes = {}
        # check if database entry is new...
        if tag in self._m_data_storage.m_data_bank:
            # NO -> database entry exists
            if keep_attributes:
                # we have to copy all attributes since deepcopy is not supported
                for key, value in self._m_data_storage.m_data_bank[tag].attrs.iteritems():
                    tmp_attributes[key] = value

            # remove database entry
            del self._m_data_storage.m_data_bank[tag]

        # make new database entry
        self._initialize_database_entry(data,
                                        tag,
                                        data_dim=data_dim)
        if keep_attributes:
            for key, value in tmp_attributes.iteritems():
                self._m_data_storage.m_data_bank[tag].attrs[key] = value
        return

    def _append_key(self,
                    tag,
                    data,
                    data_dim=None,
                    force=False):

        # check if database entry is new...
        if tag not in self._m_data_storage.m_data_bank:
            # YES -> database entry is new
            self._initialize_database_entry(data,
                                            tag,
                                            data_dim=data_dim)
            return

        # NO -> database entry exists
        # check if the existing data has the same dim and datatype
        tmp_shape = self._m_data_storage.m_data_bank[tag].shape
        tmp_dim = len(tmp_shape)

        # if the dimension offset is 1 add that dimension (e.g. save 2D image in 3D image stack)
        if data.ndim + 1 == data_dim:
            if data_dim == 3:
                data = data[np.newaxis, :, :]
            if data_dim == 2:
                data = data[np.newaxis, :]

        def _type_check():
            if tmp_dim == data.ndim:
                if tmp_dim == 3:
                    return (tmp_shape[1] == data.shape[1]) \
                        and (tmp_shape[2] == data.shape[2])
                elif tmp_dim == 2:
                    return tmp_shape[1] == data.shape[1]
                else:
                    return True
            else:
                return False

        if _type_check():

            # YES -> dim and type match
            # we always append in axis one independent of the dimension
            # 1D case
            self._m_data_storage.m_data_bank[tag].resize(tmp_shape[0] + data.shape[0],
                                                                 axis=0)
            self._m_data_storage.m_data_bank[tag][tmp_shape[0]::] = data
            return

        # NO -> shape or type is different
        # Check force
        if force:
            # YES -> Force is true
            self._set_all_key(tag,
                              data=data,
                              data_dim=data_dim)
            return

        # NO -> Error message
        raise ValueError('The port tag %s is already used with a different data type. If you want '
                         'to replace it use force = True.' % self._m_tag)

    def __setitem__(self, key, value):
        """
        Internal function needed to change data using slicing. See class documentation for an
        example.

        :param key: Slicing indices to be changed
        :param value: New values
        :return: None
        """
        if not self._check_status_and_activate():
            return

        self._m_data_storage.m_data_bank[self._m_tag][key] = value

    def set_all(self,
                data,
                data_dim=None,
                keep_attributes=False):
        """
        Set the data in the data base by replacing all old values with the values of the input data.
        If no old values exists the data is just stored. Since it is not possible to change the
        number of dimensions of a data set later in the processing history one can choose a
        dimension different to the input data of the function.
        - (#dimension of the first input data#, #desired data_dim#)
        -> (1, 1) 1D input or single value stored as list in .hdf5
        -> (1, 2) 1D input, but 2D array stored inside (i.e. a list of lists with a fixed size).
        -> (2, 2) 2D input and 2D array stored inside (i.e. a list of lists with a fixed size).
        -> (2, 3) 2D input but 3D array stored inside (i.e. a stack of images with a fixed size).
        -> (3, 3) 3D input and 3D array stored inside (i.e. a stack of images with a fixed size).

        For 2D and 3D data the first dimension always represents the list / stack (variable size)
        while the second (or third) dimension has a fixed size.
            Example: Input 2D array with size (200, 200) Desired dimension 3D
                The result is a 3D data set with the dimension (1, 200, 200). It is possible to
                append other images with the size (200, 200) or other stacks of images with the
                size (:, 200, 200).

        After creation it is possible to extend a data set using append(...) along the first
        dimension.

        :param data: The data to be saved
        :param data_dim: number of desired dimensions. If None the dimension of the first_data is
            used.
        :return: None
        """

        # check if port is ready to use
        if not self._check_status_and_activate():
            return

        self._set_all_key(self._m_tag,
                          data,
                          data_dim,
                          keep_attributes)

    def append(self,
               data,
               data_dim=None,
               force=False):
        """
        Appends input data to the existing data set with the tag of the Port along the first
        dimension. If no data exists with the tag of the Port a new data set is created.
        For more information about how the dimensions are organized see documentation of
        the function set_all(). Note it is not possible to append data with a different shape or
        data type to the existing data set.
            Example: Internal data set is 3D (storing a stack of 2D images) with shape
            (233, 300, 300) which mean it contains 233 images with 300 x 300 pixel values. It is
            only possible to extend along the first dimension by appending new images with the size
            of (300, 300) or by appending a stack of images (:, 300, 300). Everything else will
            raise exceptions.
        It is possible to force the function to overwrite the existing data set if or only if the
        shape or type of the input data does not match the existing data. Warning: This can delete
        all existing data.

        :param data: The data which will be appended
        :type data: bytearray
        :param data_dim: number of desired dimensions used if a new data set is created. If None
            the dimension of the first_data is used.
        :type data_dim: int
        :param force: If true existing data will be overwritten of shape or type does not match.
        :type force: Boolean
        :return: None
        """

        # check if port is ready to use
        if not self._check_status_and_activate():
            return

        self._append_key(self._m_tag,
                         data=data,
                         data_dim=data_dim,
                         force=force)

    def activate(self):
        """
        Activates the port. A non activated port will not save data.

        :return: None
        """
        self.m_activate = True

    def deactivate(self):
        """
        Deactivates the port. A non activated port will not save data.

        :return: None
        """
        self.m_activate = False

    def add_attribute(self,
                      name,
                      value,
                      static=True):

        if static:
            self._m_data_storage.m_data_bank[self._m_tag].attrs[name] = value
        else:
            # add information in sub Group
            self._set_all_key(tag=("header_" + self._m_tag + "/" + name),
                              data=np.asarray(value))

    def append_attribute_data(self,
                              name,
                              value):
        self._append_key(tag=("header_" + self._m_tag + "/" + name),
                         data=np.asarray([value,]))

    def del_attribute(self,
                      name):
        """
        Deletes the attribute of the data set by a given name.

        :param name: Name of the attribute.
        :return: None
        """
        # check if attribute is static
        if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            del self._m_data_storage.m_data_bank[self._m_tag].attrs[name]
        else:
            # remove non static attribute
            del self._m_data_storage.m_data_bank[("header_" + self._m_tag + "/" + name)]

    def del_all_attributes(self):
        """
        Deletes all attributes of the data set.

        :return: None
        """
        # static attributes
        for attr in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            del attr

        # non static attributes
        if ("header_" + self._m_tag + "/") in self._m_data_storage.m_data_bank:
            del self._m_data_storage.m_data_bank[("header_" + self._m_tag + "/")]

    def check_static_attribute(self,
                               name,
                               comparison_value):
        """
        Checks if a attribute exists and if it is equal to a comparison value.

        :param name: Name of the attribute
        :param comparison_value: Value for comparison
        :return: 1 if the attribute does not exist
                 0 if the attribute exists and is equal,
                 -1 if the attribute exists but is not equal
        """
        if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            if self._m_data_storage.m_data_bank[self._m_tag].attrs[name] == comparison_value:
                return 0
            else:
                return -1
        else:
            return 1

    def flush(self):
        """
        Forces the Data Storage to save all data from the memory to the hard drive without closing
        it.

        :return: None
        """
        self._m_data_storage.m_data_bank.flush()
