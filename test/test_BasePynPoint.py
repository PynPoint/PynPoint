
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `base_pynpoint` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import PynPoint

class TestBasePynpoint(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        self.base_pynpoint = PynPoint.base_pynpoint()
        pass

    def test_initialise(self):
        assert self.base_pynpoint.obj_type == 'PynPoint_parent' 

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass