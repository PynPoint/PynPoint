# Copyright (C) 2014 ETH Zurich, Institute for Astronomy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/.


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