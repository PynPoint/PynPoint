#!/opt/local/bin/python
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
            cent_remove=False,resize=False,ran_sub=False,recent=False)
images.save(filename_images)

#setting up a standard basis save file:
filename_basis = test_dir+'test_data_basis_v001.hdf5'
basis = PynPoint.basis.create_wdir(test_dir,
            cent_remove=False,resize=False,ran_sub=False,recent=False)
basis.save(filename_basis)

#setting up a standard residuals
filename_res = test_dir+'test_data_residuals_v001.hdf5'
res = PynPoint.residuals.create_winstances(images,basis)
res.save(filename_res)


