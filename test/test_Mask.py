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
Tests for `_Mask` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from PynPoint import _Mask


limit1 = 1.0e-10
limit2 = 1.0e-7


class TestMask(object):

    def setup(self):

        print("setting up " + __name__)


    def test_mk_circle(self):
        im_temp = _Mask.mk_circle(400,400,100.,200.,20.)
        assert im_temp.shape[0] == 400
        assert im_temp.shape[1] == 400
        assert im_temp.max() == 1.0
        assert im_temp.min() == 0.0
        assert im_temp[200,100] == 0.0
        assert im_temp[200,121] == 1.0
        assert im_temp[200,79] == 1.0


    def test_mk_cent_remove(self):
        temp_arr = np.ones([10,400,400])*np.pi
        im_arr_omask,im_arr_imask,cent_mask = _Mask.mk_cent_remove(temp_arr,cent_size=0.2,edge_size=1.0)
        assert temp_arr.shape[0] == 10
        assert im_arr_omask.shape[0] == 10
        assert im_arr_omask.shape[1] == 400
        assert im_arr_omask.shape[2] == 400
        assert np.allclose(im_arr_omask[0,],cent_mask*np.pi,rtol=limit1)
        assert np.allclose(im_arr_imask[0,],(1.0 - cent_mask)*np.pi,rtol=limit1)
        assert np.allclose(im_arr_omask[5,],cent_mask*np.pi,rtol=limit1)
        assert np.allclose(im_arr_imask[5,],(1.0 - cent_mask)*np.pi,rtol=limit1)
        assert im_arr_omask[0,200,200] == 0.0
        assert im_arr_imask[0,200,200] == np.pi
        assert im_arr_omask[0,0,0] == 0.0
        assert im_arr_imask[0,0,0] == np.pi
        assert im_arr_omask[0,200,20] == np.pi
        assert im_arr_imask[0,200,20] == 0.0
        assert cent_mask.max() == 1.0
        assert cent_mask.min() == 0.0
        

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass

