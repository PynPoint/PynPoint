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
Tests for `base_pynpoint` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import PynPoint as PynPoint

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