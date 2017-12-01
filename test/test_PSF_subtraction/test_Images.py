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
Tests for `Images` module.
"""
#from __future__ import print_function, division, absolute_import, unicode_literals
import os
import pytest
import numpy as np

import PynPoint as PynPoint
import PynPoint.old_version

limit0 = 1e-20
limit1 = 1e-10
limit2 = 2e-4



class TestImages(object):

    def setup(self):
        
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        test_data = (os.path.dirname(__file__))+'/test_data/'
        self.test_data_dir = test_data
        self.files_fits = [test_data+'Cube_000_Frame_0002_zoom_2.0.fits_shift.fits_planet.fits',
        test_data+'Cube_001_Frame_0130_zoom_2.0.fits_shift.fits_planet.fits',
        test_data+'Cube_000_Frame_0166_zoom_2.0.fits_shift.fits_planet.fits',
        test_data+'Cube_003_Frame_0160_zoom_2.0.fits_shift.fits_planet.fits']
        self.basis_file = self.test_data_dir + 'test_data_basis_v001.hdf5'
        
        self.files_fits_sorted = [self.files_fits[0],self.files_fits[2],self.files_fits[1],self.files_fits[3]]

        self.images1 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=False,ran_sub=None,recent=False)

        self.images2 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=False,ran_sub=2,recent=False)
        
        self.images3 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=None,recent=False,cent_size=0.2)

        self.images4 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=True,F_int=4.0,F_final=2.0,ran_sub=None,recent=True)

        self.images5 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=True,ran_sub=None,recent=False)
                                
        hdf5file = PynPoint.old_version._Util.filename4mdir(self.test_data_dir)

        # not used
        '''self.images6 = PynPoint.images.create_whdf5input(hdf5file,
                                cent_remove=False,resize=False,ran_sub=None,recent=False)'''
        
        self.basis = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=None,recent=False,cent_size=0.2)
                                
        self.imagesfits = PynPoint.images.create_wfitsfiles(self.files_fits,
                                cent_remove=True,resize=False,ran_sub=None,recent=False,cent_size=0.2)

    def test_overall_images3(self):
        assert np.array_equal(self.images3.files , self.files_fits_sorted)
        assert self.images3.num_files == len(self.images3.files) 
        assert np.array_equal(self.images3.im_size, (146,146))
        assert self.images3.cent_remove is True
        assert self.images3.cent_size == 0.2
        assert self.images3.im_arr.shape == (4,146,146) 
        assert np.allclose(self.images3.im_arr.min(), -0.0018710878410451591,rtol=limit1)
        assert np.allclose(self.images3.im_arr.max(), 0.0053438268740249742,rtol=limit1)
        assert np.allclose(self.images3.im_arr.var(), 3.0953862425934607e-07,rtol=limit1)
        assert np.allclose(self.images3.im_norm , np.array([79863.8203125,  82103.890625 ,  76156.6484375,  66806.0546875]),rtol=limit1)
        assert np.array_equal(self.images3.para , np.array([-17.3261, -17.172 , -17.0143, -16.6004]))
        assert self.images3.cent_mask.shape == (146,146) 
        assert self.images3.cent_mask.min() == 0.0 
        assert self.images3.cent_mask.max() == 1.0
        assert np.allclose(self.images3.cent_mask.var() , 0.22491619287271775 ,rtol=limit1)

        assert np.array_equal(self.images3.im_arr_mask.shape , (4,146,146))
        assert np.allclose(self.images3.im_arr_mask.min() , -0.0020284527946860184,rtol=limit1)
        assert np.allclose(self.images3.im_arr_mask.max() ,0.13108563152819708,rtol=limit1)
        assert np.allclose(self.images3.im_arr_mask.var() , 4.5119275320392779e-05 ,rtol=limit1)

    def test_overall_images1(self):
        images = self.images1
        images_base = self.images3

        assert np.array_equal(images.files, images_base.files)
        assert images.num_files == images_base.num_files
        assert np.array_equal(images.im_size, images_base.im_size)
        assert images.im_arr.shape == images_base.im_arr.shape
        assert np.array_equal(images.im_norm , images_base.im_norm)

        assert images.im_arr.shape == (4,146,146) 
        assert np.allclose(images.im_arr.min(), -0.0020284527946860184,rtol=limit1)
        assert np.allclose(images.im_arr.max() , 0.13108563152819708,rtol=limit1)
        assert np.allclose(images.im_arr.var() , 4.4735294551387646e-05 ,rtol=limit1)
        assert images.cent_remove is False
        assert np.array_equal(images.cent_mask , np.ones(shape=(146,146)))
                 
    def func4test_overall_same(self,images,images_base):
        assert np.array_equal(images.files , images_base.files )
        assert np.array_equal(images.num_files , images_base.num_files)
        assert np.array_equal(images.im_size , images_base.im_size)
        assert np.array_equal(images.cent_remove  , images_base.cent_remove) 
        assert np.array_equal(images.im_arr.shape , images_base.im_arr.shape)
        assert np.array_equal(images.im_norm , images_base.im_norm)
        assert np.array_equal(images.im_arr , images_base.im_arr)#,atol=limit0)
        assert np.array_equal(images.cent_mask,images_base.cent_mask)


    def test_images_save_restore(self,tmpdir):
        temp_file = str(tmpdir.join('tmp_images_hdf5.h5'))

        self.images3.save(temp_file)

        temp_images = PynPoint.images.create_restore(temp_file)
        self.func4test_overall_same(self.images3,temp_images)
    

    def test_mk_psfmodel(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        # assert self.images3._have_psf_coeffs == True

        assert self.images3._psf_coeff.shape == (4,4)
        assert np.allclose(self.images3._psf_coeff.mean() ,-9.1344831261194862e-20,rtol=limit1)
        assert np.allclose(self.images3._psf_coeff.min() , -0.045193361270144991,rtol=limit1)
        assert np.allclose(self.images3._psf_coeff.max() , 0.0348221927176,rtol=limit1)
        assert np.allclose(self.images3._psf_coeff.var() , 0.000506169218776,rtol=limit1)
        assert self.images3.psf_im_arr.shape == (4, 146, 146)
        assert np.allclose(self.images3.psf_im_arr.mean() , 0.000293242276843,rtol=limit1)
        assert np.allclose(self.images3.psf_im_arr.min() , -0.00187108784105,rtol=limit1)
        assert np.allclose(self.images3.psf_im_arr.max() , 0.00534382687402,rtol=limit1)
        assert np.allclose(self.images3.psf_im_arr.var() , 3.09538624259e-07,rtol=limit1)
        
    
    def test_mk_psf_realisation(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        im_temp =  self.images3.mk_psf_realisation(1,full=False)
        assert im_temp.shape  == (146, 146)

        assert np.allclose(im_temp.mean() , 0.000274016013066,rtol=limit1)
        assert np.allclose(im_temp.min() , -0.00159553949981,rtol=limit1)
        assert np.allclose(im_temp.max() , 0.00425071210256,rtol=limit1)
        assert np.allclose(im_temp.var() , 3.06145293065e-07,rtol=limit1)
        
    def test_mk_psf_realisation2(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        im_temp =  self.images3.mk_psf_realisation(1,full=True)
        assert im_temp.shape  == (146, 146)

        assert np.allclose(im_temp.mean() , 0.00141957914019,rtol=limit1)
        assert np.allclose(im_temp.min() , -0.00159553949981,rtol=limit1)
        assert np.allclose(im_temp.max() , 0.127411745859,rtol=limit1)
        assert np.allclose(im_temp.var() , 4.48979119722e-05,rtol=limit1)


    def test_random_subsample(self):

        images2 = self.images2

        images_temp = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=False,ran_sub=3,recent=False)

        assert images2.im_arr.shape == (2, 146, 146)

    def teardown(self):
        #tidy up
        if os.path.isfile(self.test_data_dir + "PynPoint_database.hdf5"):
            os.remove(self.test_data_dir + "PynPoint_database.hdf5")
    

if __name__ == '__main__':
    pytest.main("-k TestImages")
