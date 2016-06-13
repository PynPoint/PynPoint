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
        assert (type(tag) == str), "Error: Port tags need to be strings."
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

    def __init__(self):
        super(InputPort,self).__init__()
        pass

    def get_index(self,
                  index):
        pass

    def get_all(self):
        pass


class OutputPort(Port):

    def __init__(self,
                 tag,
                 activate_init=True):

        super(OutputPort, self).__init__(tag)
        self.m_activate = activate_init
        self.m_dim = None
        self.m_dtype = None

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
                                   data_shape=None):
        """
        Initializes the connection to the database for the given tag. If the tag exists its type and shape is copied, if
        not a new dataset is created using the input first_data. Per default the inputs are stored using the following
        data structures:

            Input -> Internal structure
            1D array -> 1D array which can be extended using the append method.
            2D array -> 2D array with fixed size (no extension using the append method)
            3D array -> Stack of 2D images with the same fixed image size. New images can be extended using the append
                        method. NOTE it is not possible to store images with different sizes

            TODO: explain shape and type

        :param first_data:
        :param type_in:
        :param shape_in:
        :return:
        """

        # check if the input is a numpy array
        if type(first_data) is not np.ndarray or first_data.ndim > 3:
            raise ValueError('Outport port can only save numpy arrays from 1D to 3D. If you want to save a int,'
                                 'float, string ... use Port attributes instead.')

        # if no data_shape is given check the input data
        if data_shape is None:
            if first_data.ndim == 1:
                data_shape = (None,)
            elif first_data.ndim == 2:
                data_shape = first_data.shape
            elif first_data.ndim == 3:
                data_shape = (None, first_data.shape[1], first_data.shape[2])
            else:
                raise ValueError('Input shape not supported')

        self.m_dim = first_data.ndim
        self.m_dtype = first_data.dtype
        self._m_data_storage.m_data_bank.create_dataset(self._m_tag,
                                                        data=first_data,
                                                        maxshape=data_shape)

    def set_index(self,
                  index,
                  data):
        raise NotImplementedError('Missing, Sorry')


    def set_all(self,
                data,
                force=False):

        '''if not self._check_status():
        return

        # check if port was used before
        # NO -> create dataset with data
        # Yes -> delete all data and add new data
        pass'''
        raise NotImplementedError('Missing, Sorry')


    def append(self,
               data,
               data_shape=None,
               force = False):

        # check if port is ready to use
        if not self._check_status_and_activate():
            return

        # check if database entry is new...
        if self._m_tag not in self._m_data_storage.m_data_bank:
            # YES -> database entry is new
            self._initialize_database_entry(data,
                                            data_shape=data_shape)
            return

        # NO -> database entry exists
        # check if the existing data has the same shape and datatype
        if (len(self._m_data_storage.m_data_bank[self._m_tag].shape) == data.ndim) \
                and self._m_data_storage.m_data_bank[self._m_tag].dtype == data.dtype:

            # YES -> shape and type matches
            self.m_dtype = self._m_data_storage.m_data_bank[self._m_tag].dtype
            actual_shape = self._m_data_storage.m_data_bank[self._m_tag].shape
            self.m_dim = len(actual_shape)

            # 1D case
            if self.m_dim == 1:
                self._m_data_storage.m_data_bank[self._m_tag].resize(actual_shape[0] + data.shape[0], axis=0)
                self._m_data_storage.m_data_bank[self._m_tag][actual_shape[0]::] = data
            # 2D case
            if self.m_dim == 2:
                raise NotImplementedError('Missing, Sorry')

            # 3D case
            if self.m_dim == 3:
                print self._m_data_storage.m_data_bank[self._m_tag].shape
                self._m_data_storage.m_data_bank[self._m_tag].resize(actual_shape[0] + data.shape[0], axis=0)
                print self._m_data_storage.m_data_bank[self._m_tag].shape
                self._m_data_storage.m_data_bank[self._m_tag][actual_shape[0]::] = data
            return

        # NO -> shape or type is different
        # Check force
        if force:
            # YES -> Force is true
            raise NotImplementedError('Missing, Sorry')

        # NO -> Error message
        raise ValueError('The port tag %s is already used with a different data type. If you want to replace it use '
                         'force = True.' % self._m_tag)

    def activate(self):
        self.m_activate = True

    def deactivate(self):
        self.m_activate = False