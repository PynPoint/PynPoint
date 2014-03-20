
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `PynPoint_v1_5` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
# import PynPoint_v1_5 as PynPoint
import PynPoint
import numpy as np
# from PynPoint_v1_5 import PynPoint_ctx as pynpoint_ctx
from PynPoint import ctx as pynpoint_ctx


class TestPynpointParent(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        # self.tempdata = np.array([1.,2.,3.])
          # PynPoint.pynpointctx.ctx()#.im_arr_store(self.tempdata)
        self.pynpointctx =pynpoint_ctx.Ctx()
        pass

    def test_initialise(self):
        assert 1==1
        # assert np.array_equal(self.im_arr_cache.get() , self.tempdata)

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass