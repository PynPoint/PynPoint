
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `_Cache` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from PynPoint import _Cache


class TestCache(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        self.tempdata = np.array([1.,2.,3.])
        self.im_arr_cache =  _Cache.im_arr_store(self.tempdata)
        pass

    def test_initialise(self):

        assert np.array_equal(self.im_arr_cache.get() , self.tempdata)

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass