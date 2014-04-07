
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `_Ctx` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

from PynPoint import _Ctx
import pytest


pytest.skip("Nothing being tested yet")
class TestCtx(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        # self.tempdata = np.array([1.,2.,3.])
        # PynPoint.pynpointctx.ctx()#.im_arr_store(self.tempdata)
        self.pynpointctx = _Ctx.Ctx()
        pass

    def test_initialise(self):
        assert 1==1
        # assert np.array_equal(self.im_arr_cache.get() , self.tempdata)

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass