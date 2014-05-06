#!/opt/local/bin/python
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

# This script was used to generate the test data sets that
# are used as part of the quality controls in the unit tests.
#

import PynPoint
import os

test_dir = (os.path.dirname(__file__))+'/test_data/'
#test_dir = 'test_data/'

print(test_dir)

#setting up a standard images file:
filename_images = test_dir+'test_data_images_v001.hdf5'
images = PynPoint.images.create_wdir(test_dir,
            cent_remove=False,resize=False,ran_sub=None,recent=False)
images.save(filename_images)

#setting up a standard basis save file:
filename_basis = test_dir+'test_data_basis_v001.hdf5'
basis = PynPoint.basis.create_wdir(test_dir,
            cent_remove=False,resize=False,ran_sub=None,recent=False)
basis.save(filename_basis)

#setting up a standard residuals
filename_res = test_dir+'test_data_residuals_v001.hdf5'
res = PynPoint.residuals.create_winstances(images,basis)
res.save(filename_res)


