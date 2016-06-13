import os
import collections
import warnings
import h5py
from Processing import PypelineModule, WritingModule, ReadingModule
from DataIO import DataStorage


class Pypeline(object):

    def __init__(self,
                 working_place_in=None,
                 input_place_in=None,
                 output_place_in=None):
        # TODO: Documentation

        self._m_working_place = working_place_in
        self._m_input_place = input_place_in
        self._m_output_place = output_place_in
        self._m_modules = collections.OrderedDict()
        self._m_data_storage = DataStorage(working_place_in + '/PynPoint_database.hdf5')

    def __setattr__(self, key, value):

        if key in ["_m_working_place", "_m_input_place", "_m_output_place"]:
            assert (os.path.isdir(str(value))), 'Error: Input directory for ' + str(key) + ' does not exist - input ' \
                                                'requested: %s' % value

        super(Pypeline, self).__setattr__(key, value)

    def add_module(self,
                   pipeline_module):
        assert isinstance(pipeline_module, PypelineModule), 'Error: the given pipeline_module is not a' \
                                                            ' accepted PypelineModule.'

        # if no specific output directory is given use the default
        if isinstance(pipeline_module, WritingModule):
            if pipeline_module.m_output_location is None:
                pipeline_module.m_output_location = self._m_output_place

        # if no specific input directory is given use the default
        if isinstance(pipeline_module, ReadingModule):
            if pipeline_module.m_input_location is None:
                pipeline_module.m_input_location = self._m_input_place

        pipeline_module.connect_database(self._m_data_storage)

        if pipeline_module.name in self._m_modules:
            warnings.warn('Processing module names need to be unique. Overwriting the old Module')

        self._m_modules[pipeline_module.name] = pipeline_module

    def remove_module(self,
                      name):

        if name in self._m_modules:
            del self._m_modules[name]
            return True
        else:
            return False

    def get_module_names(self):
        return self._m_modules.keys()

    def run(self):
        print "Start running Pypeline ..."
        for key in self._m_modules:
            print "Start running " + key + "..."
            self._m_modules[key].run()
            print "Finished running " + key
        print "Finished running Pypeline."

    def run_module(self, name):
        if name in self._m_modules:
            self._m_modules[name].run()
        else:
            warnings.warn('Module not found')
