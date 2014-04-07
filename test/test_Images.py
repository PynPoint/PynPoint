
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `Images` module.
"""
#from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np

import PynPoint

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
        self.basis_file = self.test_data_dir  + 'test_data_basis_v001.hdf5'
        
        self.files_fits_sorted = [self.files_fits[0],self.files_fits[2],self.files_fits[1],self.files_fits[3]]

        self.images1 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=False,ran_sub=False,recent=False)

        #KEY WORK RAN_SUB NOT YET IMPLIMENTED!!!!
        self.images2 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=False,ran_sub=2,recent=False)
        
        self.images3 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)

        self.images4 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=True,F_int=4.0,F_final=2.0,ran_sub=False,recent=True)

        self.images5 = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=False,resize=True,ran_sub=False,recent=False)
                                
        hdf5file = PynPoint._Util.filename4mdir(self.test_data_dir)
        self.images6 = PynPoint.images.create_whdf5input(hdf5file,
                                cent_remove=False,resize=False,ran_sub=False,recent=False)
        
        
        #self.images5 = PynPoint.images(self.files_fits_sorted,intype='files',cent_remove=False,resize=False,ran_sub=False,recent=False)

        # self.basis = PynPoint.basis.create_restore(self.basis_file)
        self.basis = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)
                                
        self.imagesfits = PynPoint.images.create_wfitsfiles(self.files_fits,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)
        
        
        #self.basis_hdf = PynPoint.basis(self.file_hdf,intype='HDF5')
        
        pass

    def test_overall_images3(self):
        print('-----')
        print(self.files_fits_sorted)
        print('-----')
        print(self.images3.files)
        print('-----')
        assert np.array_equal(self.images3.files , self.files_fits_sorted)
        assert self.images3.num_files == len(self.images3.files) 
        assert np.array_equal(self.images3.im_size, (146,146))
        assert self.images3.cent_remove is True
        assert self.images3.im_arr.shape == (4,146,146) 
        assert np.allclose(self.images3.im_arr.min(), -5.7673052651807666e-05,rtol=limit1)
        assert np.allclose(self.images3.im_arr.max(), 0.00016471423441544175,rtol=limit1)
        assert np.allclose(self.images3.im_arr.var(), 3.149548727488477e-10,rtol=limit1)
        assert np.allclose(self.images3.im_norm , np.array([ 2339855.10735457,  2484443.10731339 ,  2576155.10408142,  2167391.10663852]),rtol=limit1)
        assert np.array_equal(self.images3.para , np.array([-17.3261, -17.172 , -17.0143, -16.6004]))
        assert self.images3.cent_mask.shape == (146,146) 
        assert self.images3.cent_mask.min() == 0.0 
        assert self.images3.cent_mask.max() == 1.0
        assert self.images3.cent_mask.var() == 0.22491619287271775 
        # assert self.images3.im_ave.shape == (146,146)
        # assert self.images3.im_ave.min() == -2.4491993372066645e-05 
        # assert self.images3.im_ave.max() == 0.00013430662147584371
        # assert self.images3.im_ave.var() == 2.1823141818009155e-10
        assert np.array_equal(self.images3.im_arr_mask.shape , (4,146,146)) 
        assert np.allclose(self.images3.im_arr_mask.min() , -6.9235109549481422e-05,rtol=limit1)
        assert np.allclose(self.images3.im_arr_mask.max() , 0.0044742124155163765,rtol=limit1)
        assert np.allclose(self.images3.im_arr_mask.var() , 4.604643923044455e-08 ,rtol=limit1)
        

    def test_overall_images1(self):
        images = self.images1
        images_base = self.images3
        #general comparisons with the baseline base instance
        assert np.array_equal(images.files, images_base.files) 
        assert images.num_files == images_base.num_files
        assert np.array_equal(images.im_size, images_base.im_size)
        assert images.im_arr.shape == images_base.im_arr.shape
        assert np.array_equal(images.im_norm , images_base.im_norm)

        assert images.im_arr.shape == (4,146,146) 
        assert np.allclose(images.im_arr.min(), -6.9235109549481422e-05,rtol=limit1)
        assert np.allclose(images.im_arr.max() , 0.0044742124155163765,rtol=limit1)
        assert np.allclose(images.im_arr.var() , 4.5663498111183116e-08 ,rtol=limit1)
        assert images.cent_remove is False
        assert np.array_equal(images.cent_mask , np.ones(shape=(146,146))) 
                 
    # def test_overall_images2(self):
    #     images = self.images2
    #     images_base = self.images3
    #     #general comparisons with the baseline base instance
    #     assert len(images.files) == 2 
    #     assert images.num_files == len(images.files)
    #     #could add extra tests
 
    def func4test_overall_same(self,images,images_base):
        assert np.array_equal(images.files , images_base.files )
        assert np.array_equal(images.num_files , images_base.num_files)
        assert np.array_equal(images.im_size , images_base.im_size)
        assert np.array_equal(images.cent_remove  , images_base.cent_remove) 
        assert np.array_equal(images.im_arr.shape , images_base.im_arr.shape)
        assert np.array_equal(images.im_norm , images_base.im_norm)
        assert np.array_equal(images.im_arr , images_base.im_arr)#,atol=limit0)
        assert np.array_equal(images.cent_mask,images_base.cent_mask)
        #assert np.array_equal(images.im_ave,images_base.im_ave)#,atol=limit1)
        
    # def test_overall_images5(self):
    #     self.func4test_overall_same(self.images5,self.images1)

#    def test_overall_images_hdf(self):
#        self.func4test_overall_same(self.images_hdf,self.images3)

    def test_images_save_restore(self,tmpdir):
        temp_file = str(tmpdir.join('tmp_images_hdf5.h5'))
        
        self.images3.save(temp_file)
        #temp_images = PynPoint.images()#'',intype='empty')
        #temp_images.restore(temp_file)
        temp_images = PynPoint.images.create_restore(temp_file)        
        self.func4test_overall_same(self.images3,temp_images)
    

    def test_mk_psfmodel(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        assert self.images3._have_psf_coeffs == True
        assert self.images3._psf_coeff.shape == (4,4)
        assert np.allclose(self.images3._psf_coeff.mean() ,-9.1344831261194862e-20,rtol=limit1)
        assert np.allclose(self.images3._psf_coeff.min() , -0.0015266346011625687,rtol=limit1)
        assert np.allclose(self.images3._psf_coeff.max() , 0.00094515408517407723,rtol=limit1)
        assert np.allclose(self.images3._psf_coeff.var() , 5.1543928939621446e-07,rtol=limit1)
        assert self.images3.psf_im_arr.shape == (4, 146, 146)
        assert np.allclose(self.images3.psf_im_arr.mean() , 9.269862414087591e-06,rtol=limit1)
        assert np.allclose(self.images3.psf_im_arr.min() , -5.7673052651807639e-05,rtol=limit1)
        assert np.allclose(self.images3.psf_im_arr.max() , 0.00016471423441544178,rtol=limit1)
        assert np.allclose(self.images3.psf_im_arr.var() , 3.1495487274884796e-10,rtol=limit1)
        
    
    # def test_mk_psfmodel(self):
    #     basis = self.basis_restore
    #     self.images3.mk_psfmodel2(basis,3,mask=None)
    #     self.images3.mk_psfmodel(basis,3,mask=None)
    # 

    def test_mk_psf_realisation(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        im_temp =  self.images3.mk_psf_realisation(1,full=False)
        assert im_temp.shape  == (146, 146)
        assert np.allclose(im_temp.mean() , 9.0554680538744472e-06,rtol=limit1)
        assert np.allclose(im_temp.min() , -5.2728148148162236e-05,rtol=limit1)
        assert np.allclose(im_temp.max() , 0.00014047422155272216,rtol=limit1)
        assert np.allclose(im_temp.var() , 3.3434705799034875e-10,rtol=limit1)
        
    def test_mk_psf_realisation2(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        im_temp =  self.images3.mk_psf_realisation(1,full=True)
        assert im_temp.shape  == (146, 146)
        assert np.allclose(im_temp.mean() , 4.6913116907487437e-05,rtol=limit1)
        assert np.allclose(im_temp.min() , -5.2728148148162236e-05,rtol=limit1)
        assert np.allclose(im_temp.max() , 0.0042106015505874173,rtol=limit1)
        assert np.allclose(im_temp.var() , 4.9033794533516369e-08,rtol=limit1)
        


    def test_plt_stackave(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        self.images3.plt_stackave()

    def test_plt_resid(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        self.images3.plt_resid(0)

    def test_plt_psf(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        self.images3.plt_psf(0)

    def test_plt_im(self):
        self.images3.plt_im(0)

    def test_anim_im(self):
        self.images3.anim_im()

    def test_anim_active(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis,3)#,mask=None)
        self.images3.anim_active()

        
    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        tempfilename = PynPoint._Util.filename4mdir(self.test_data_dir,filetype='convert')
        if os.path.isfile(tempfilename):
            os.remove(tempfilename)
        
        pass