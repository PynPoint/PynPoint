import os
from Processing import PypelineModule


class Pypeline(object):

    def __init__(self,
                 working_place_in=None,
                 input_place_in=None,
                 output_place_in=None):
        # TODO: Documentation

        self._m_working_place = working_place_in
        self._m_input_place = input_place_in
        self._m_output_place = output_place_in
        self._m_modules = []

    def __setattr__(self, key, value):

        if key in ["_m_working_place", "_m_input_place", "_m_output_place"]:
            assert (os.path.isdir(str(value))), 'Error: Input directory for ' + str(key) + ' does not exist - input ' \
                                                'requested: %s' % value

        super(Pypeline, self).__setattr__(key, value)

    def add_module(self,
                   pipeline_module):
        assert isinstance(pipeline_module, PypelineModule), 'Error: the given pipeline_module is not a' \
                                                            ' accepted PypelineModule.'
        self._m_modules.append(pipeline_module)

    def remove_module_by_index(self,
                               index):
        del self._m_modules[index]

    def remove_module_by_name(self,
                              name):
        result = False
        for module in self._m_modules:
            if module.name == str(name):
                del module
                result = True

        return result

    def get_modules(self):
        modules = []
        for module in self._m_modules:
            modules.append(module.name)
        return modules

    def run(self):
        print "Start running Pypeline ..."
        for module in self._m_modules:
            print "Start running " + module.name + "..."
            module.run()
            print "Finished running " + module.name
        print "Finished running Pypeline."
