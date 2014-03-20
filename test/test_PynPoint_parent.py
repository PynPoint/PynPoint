
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `PynPoint_v1_5` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
# import PynPoint_v1_5 as PynPoint
import PynPoint

class TestPynpointParent(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        self.pynpoint_parent = PynPoint.pynpoint_parent()
        pass

    def test_initialise(self):
        assert self.pynpoint_parent.obj_type == 'PynPoint_parent' 

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass