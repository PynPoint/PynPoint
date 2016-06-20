"""
Modules to save and load data in / from a .hdf5 database.
"""
from abc import ABCMeta, abstractmethod
import warnings
import h5py
import numpy as np


class DataStorage(object):

    def __init__(self,
                 location_in):
        self._m_location = location_in
        self.m_data_bank = None
        self.m_open = False

    def open_connection(self):
        if self.m_open:
            return
        self.m_data_bank = h5py.File(self._m_location, mode='a')
        self.m_open = True

    def close_connection(self):
        if not self.m_open:
            return
        self.m_data_bank.close()
        self.m_open = False


class Port:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 tag):
        assert (isinstance(tag, str)), "Error: Port tags need to be strings."
        self._m_tag = tag
        self._m_data_storage = None
        self._m_data_base_active = False

    def close_port(self):
        if self._m_data_base_active:
            self._m_data_storage.close_connection()
            self._m_data_base_active = False

    def open_port(self):
        if not self._m_data_base_active:
            self._m_data_storage.open_connection()
            self._m_data_base_active = True

    def set_database_connection(self,
                                data_base_in):
        self._m_data_storage = data_base_in


class InputPort(Port):

    def __init__(self,
                 tag):
        super(InputPort, self).__init__(tag)

    def _check_status_and_activate(self):
        if self._m_data_storage is None:
            warnings.warn("Port can not load data unless a database is connected")
            return False

        if not self._m_data_base_active:
            self.open_port()

        return True

    def __getitem__(self, item):

        if not self._check_status_and_activate():
            return

        return self._m_data_storage.m_data_bank[self._m_tag][item]

    def get_all(self):
        self._check_status_and_activate()
        return self._m_data_storage.m_data_bank[self._m_tag]

    def get_attribute(self,
                      name):
        self._check_status_and_activate()
        return self._m_data_storage.m_data_bank[self._m_tag].attrs[name]

    def get_all_attributes(self):
        self._check_status_and_activate()
        return self._m_data_storage.m_data_bank[self._m_tag].attrs


class OutputPort(Port):

    def __init__(self,
                 tag,
                 activate_init=True):

        super(OutputPort, self).__init__(tag)
        self.m_activate = activate_init

    # internal functions
    def _check_status_and_activate(self):
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
                                   data_dim=None):
        """
        We only except the following inputs
            (data_dim, data dimension) = (1,1), (2,1), (2,2), (3,2), (3,3)
        """

        # check Error cases
        if isinstance(first_data, np.array) or first_data.ndim > 3 or first_data.ndim < 1:
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

        self._m_data_storage.m_data_bank.create_dataset(self._m_tag,
                                                        data=first_data,
                                                        maxshape=data_shape)

    def __setitem__(self, key, value):
        if not self._check_status_and_activate():
            return

        self._m_data_storage.m_data_bank[self._m_tag][key] = value

    def set_all(self,
                data,
                data_dim=None):

        # check if port is ready to use
        if not self._check_status_and_activate():
            return

        # check if database entry is new...
        if self._m_tag in self._m_data_storage.m_data_bank:
            # NO -> database entry exists
            # remove database entry
            del self._m_data_storage.m_data_bank[self._m_tag]

        # make new database entry
        self._initialize_database_entry(data,
                                        data_dim=data_dim)
        return

    def append(self,
               data,
               data_dim=None,
               force=False):

        # check if port is ready to use
        if not self._check_status_and_activate():
            return

        # check if database entry is new...
        if self._m_tag not in self._m_data_storage.m_data_bank:
            # YES -> database entry is new
            self._initialize_database_entry(data,
                                            data_dim=data_dim)
            return

        # NO -> database entry exists
        # check if the existing data has the same dim and datatype
        tmp_shape = self._m_data_storage.m_data_bank[self._m_tag].shape
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
            self._m_data_storage.m_data_bank[self._m_tag].resize(tmp_shape[0] + data.shape[0],
                                                                 axis=0)
            self._m_data_storage.m_data_bank[self._m_tag][tmp_shape[0]::] = data
            return

        # NO -> shape or type is different
        # Check force
        if force:
            # YES -> Force is true
            self.set_all(data=data,
                         data_dim=data_dim)
            return

        # NO -> Error message
        raise ValueError('The port tag %s is already used with a different data type. If you want '
                         'to replace it use force = True.' % self._m_tag)

    def activate(self):
        self.m_activate = True

    def deactivate(self):
        self.m_activate = False

    def add_attribute(self,
                      name,
                      value):
        self._m_data_storage.m_data_bank[self._m_tag].attrs[name] = value

    def check_attribute(self,
                        name,
                        comparison_value):
        """

        :param name:
        :param comparison_value:
        :return: 1 does not exist, 0 exists and is equal, -1 exists but is not equal
        """
        if name in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            if self._m_data_storage.m_data_bank[self._m_tag].attrs[name] == comparison_value:
                return 0
            else:
                return -1
        else:
            return 1

    def del_all_attributes(self):
        for attr in self._m_data_storage.m_data_bank[self._m_tag].attrs:
            del attr

    def flush(self):
        self._m_data_storage.m_data_bank.flush()
