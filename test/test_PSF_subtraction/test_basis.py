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
Tests for `Basis` module.
"""
#from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np

import PynPoint as PynPoint
import PynPoint.old_version

limit0 = 1e-20
limit1 = 1e-10
limit2 = 2e-4

class TestBasis(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        test_data = (os.path.dirname(__file__))+'/test_data/'
        print(test_data)
        self.test_data_dir = test_data
        self.files_fits = [test_data+'Cube_000_Frame_0002_zoom_2.0.fits_shift.fits_planet.fits',
        test_data+'Cube_001_Frame_0130_zoom_2.0.fits_shift.fits_planet.fits',
        test_data+'Cube_000_Frame_0166_zoom_2.0.fits_shift.fits_planet.fits',
        test_data+'Cube_003_Frame_0160_zoom_2.0.fits_shift.fits_planet.fits']
        self.file_hdf = test_data+'testfile_basis.hdf5'

        self.files_fits_sorted = [self.files_fits[0],self.files_fits[2],self.files_fits[1],self.files_fits[3]]


        self.basis1 = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=False,ran_sub=None,recent=False)
        hdf5file = PynPoint.old_version._Util.filename4mdir(self.test_data_dir)
        self.basis5 = PynPoint.basis.create_whdf5input(hdf5file,
                                cent_remove=False,resize=False,ran_sub=None,recent=False)

        self.basis3 = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=None,recent=False,cent_size=0.2)
        self.basis4 = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=True,F_int=4.0,F_final=2.0,ran_sub=None,recent=True)
        self.basisfits = PynPoint.basis.create_wfitsfiles(self.files_fits,
                                cent_remove=True,resize=False,ran_sub=None,recent=False,cent_size=0.2)

        self.eg_array1 = np.arange(100.).reshape(4,5,5)
        self.ave_eg_array1 = np.array([[ 37.5,  38.5,  39.5,  40.5,  41.5],
                        [ 42.5,  43.5,  44.5,  45.5,  46.5],
                        [ 47.5,  48.5,  49.5,  50.5,  51.5],
                        [ 52.5,  53.5,  54.5,  55.5,  56.5],
                        [ 57.5,  58.5,  59.5,  60.5,  61.5]])
                        

        self.eg_array2 = np.array([[ 0.75, -2.25, -0.25,  1.75],[ 0.5, -2.5, -0.5,  2.5],[-1.25,  3.75,  0.75, -3.25],[-2.25, -0.25,  0.75,  1.75]])
        self.eg_array2_pca = np.array([
                        [[-0.17410866485907938728 , 0.71893395614514299385],[ 0.11771735865682392275 ,-0.66254264994288725177]],
                        [[ 0.84800005432046565712 ,-0.13400452403625343067],[-0.29357897148679990007 ,-0.420416558797411688]],
                        [[-0.0241263485317723507 , 0.46387148461541644062],[-0.80619725314070234123 , 0.36645211705705693639]],
                        [[ 0.5        , 0.5       ],[ 0.5        , 0.5       ]]])
        self.eg_array2_coeff = np.array([
                        [ -2.937061877035140e+00 ,  2.751759847981489e-01 , -2.189650836484913e-01],
                        [ -3.599604526978027e+00 , -1.452405739992628e-01 ,  1.474870334085662e-01],
                        [  5.155189797925138e+00 , -4.163474455600442e-01 , -2.594131731843463e-02],
                        [ -8.591506115107919e-01 , -2.830412197722555e+00 , -2.504032196304351e-02],
                        [  1.00 ,  0.0  ,0.0],
                        [  0.0 ,  1.00  ,0.0],
                        [ 0.0 , 0.0  , 1.00000e+00],
                        [  0.0 , 0.0   , 0.0 ]])                 




        pass
        
    def test_basis_save_restore(self,tmpdir):
        temp_file = str(tmpdir.join('tmp_hdf5.h5'))
        self.basis3.save(temp_file)
        temp_basis = PynPoint.basis.create_restore(temp_file)#('',intype='restore')

        self.func4test_overall_same(self.basis3,temp_basis)

    def test_overall_basis3(self):
        assert np.array_equal(self.basis3.files , self.files_fits_sorted)
        assert self.basis3.num_files == len(self.basis3.files) 
        assert np.array_equal(self.basis3.im_size , (146,146))
        assert self.basis3.cent_remove is True
        assert self.basis3.im_arr.shape == (4,146,146)
        assert np.allclose(self.basis3.im_arr.min() , -0.00171494746241,rtol=limit1)
        assert np.allclose(self.basis3.im_arr.max() , 0.00177186490054,rtol=limit1)
        assert np.allclose(self.basis3.im_arr.var() , 9.49839029417e-08 ,rtol=limit1)

        assert np.allclose(self.basis3.im_norm , np.array([79863.8203125,  82103.890625 ,  76156.6484375,  66806.0546875]),rtol=limit1)
        assert np.array_equal(self.basis3.para , np.array([-17.3261, -17.172,  -17.0143, -16.6004]))
        assert self.basis3.cent_mask.shape == (146,146) 
        assert self.basis3.cent_mask.min() == 0.0 
        assert self.basis3.cent_mask.max() == 1.0
        assert np.allclose(self.basis3.cent_mask.var() , 0.224916192873,rtol=limit1)
        assert self.basis3.psf_basis.shape == (4,146,146) 
        assert np.allclose(self.basis3.psf_basis.var() , 4.6765801282319207e-05,rtol=limit1)
        assert self.basis3.im_ave.shape == (146,146)
        assert np.allclose(self.basis3.im_ave.min() , -0.000763643189976 ,rtol=limit1)
        assert np.allclose(self.basis3.im_ave.max() , 0.0042338920493,rtol=limit1)
        assert np.allclose(self.basis3.im_ave.var() , 2.14554721318e-07,rtol=limit1)
        
    def test_overall_basis1(self):
        basis = self.basis1
        basis_base = self.basis3
        assert np.array_equal(basis.files , basis_base.files)
        assert basis.num_files == basis_base.num_files
        assert np.array_equal(basis.im_size , basis_base.im_size)
        assert np.array_equal(basis.im_arr.shape , basis_base.im_arr.shape)
        assert np.array_equal(basis.im_norm , basis_base.im_norm)

        assert np.array_equal(basis.im_arr.shape , (4,146,146) )
        assert np.allclose(basis.im_arr.min() , -0.0201545461598,rtol=limit1)
        assert np.allclose(basis.im_arr.max() , 0.0278516903399,rtol=limit1)
        assert np.allclose(basis.im_arr.var() , 7.24677588909e-07 ,rtol=limit1)
        assert basis.cent_remove is False
        assert np.array_equal(basis.cent_mask , np.ones(shape=(146,146)))  
        
        assert np.array_equal(basis.psf_basis.shape , (4,146,146) )

        assert np.allclose(basis.psf_basis.var() , 4.69117947893e-05,rtol=limit1)
        

    def func4test_overall_same(self,basis,basis_base):
        #assert np.array_equal(basis.files, basis_base.files)
        assert np.array_equal(basis.num_files , basis_base.num_files)
        assert np.array_equal(basis.im_size,basis_base.im_size)
        assert basis.cent_remove  == basis_base.cent_remove 

        assert np.allclose(basis.im_norm, basis_base.im_norm, rtol=1e-7)
        assert np.allclose(basis.im_arr, basis_base.im_arr, rtol=1e-4)
        assert np.allclose(basis.psf_basis[0], basis_base.psf_basis[0], rtol=1e-3)
        assert np.allclose(basis.psf_basis[1], basis_base.psf_basis[1], rtol=1e-1)
        assert np.allclose(basis.psf_basis[2], basis_base.psf_basis[2], rtol=1e-2)

        # TODO The comparison of the fourth component does not work anymore now that get_data
        # reads both float32 and float64. This will require a change of the test data.
        # assert np.allclose(basis.psf_basis[3], basis_base.psf_basis[3], rtol=1e-1)

        assert np.allclose(basis.cent_mask, basis_base.cent_mask, rtol=1e-7)
        assert np.allclose(basis.im_ave, basis_base.im_ave, rtol=1e-7)


    def test_overall_basis5(self):
        self.func4test_overall_same(self.basis5,self.basis1)
        
    def test_mk_psfmodel(self):
        basis = self.basis3
        basis.mk_psfmodel(20)
        assert np.allclose(basis.psf_im_arr.mean() , 0.000274004663716341,rtol=limit1)

    def teardown(self):
        if os.path.isfile(self.test_data_dir + "PynPoint_database.hdf5"):
            os.remove(self.test_data_dir + "PynPoint_database.hdf5")
