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


# System imports
from __future__ import print_function, division

from PynPoint.old_version._BasePynPoint import base_pynpoint

from PynPoint.old_version import _Creators
from PynPoint.old_version import _Util


#Basis Class
class basis(base_pynpoint):
    """
    For creating a basis set from a given set of postage stamp images
	"""
    
    def __init__(self):
        """
        Initialise an instance of the bais class.
        """
        self.obj_type = 'PynPoint_basis'
        
    @classmethod
    def create_wdir(cls, dir_in,**kwargs):#dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
        """
        Creates an instance of the basis class.

        :param dir_in: name of the directory with fits files
        :param recent: if True, the images will be re-centered
        :param resize: if True, the final images will be increased by a factor set by F final
        :param cent_remove: if True, the central region will be masked (size set by cent size).
        :param F_final: factor increase in resolution of final images (resize must be True).
        :param ran_sub: a random subset is used if a number is passed.
        :param para_sort: if True, the images will be sorted so that the parallax angles increase through the stack.
        :param cent_size: radius of the central mask as fraction of the full image size.
        :param edge_size: diameter of the outer mask in fraction of the image size.
        :param stackave: if set to an integer (N), then the stack will be reduced by averaging over adjacent N images.
        :return: instance of the basis class


        """

        obj = cls()
        _Creators.pynpoint_create_wdir(obj, dir_in, **kwargs)
        basis_save = obj.mk_basis_set()
        return obj

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance of basis from hdf5 file.

        :param file_in: path to the hdf5 file containing the images
        :param kwargs: accepts the same keyword options as :py:func:`create_wdir`
        :return: instance of the basis class


        """

        obj = cls()
        _Creators.pynpoint_create_whdf5input(obj, file_in, **kwargs)
        obj.mk_basis_set()
        return obj

        
    @classmethod
    def create_restore(cls, filename):
        """
        Restores data from a hdf5 file previously created using the save method of a basis instance.

        :param filename: name of the inputfile
        :return: instance of the basis class

        """

        obj = cls()
        _Creators.restore(obj, filename)
        return obj

    @classmethod
    def create_wfitsfiles(cls, files,**kwargs):
        """
        Creates an instance of basis from a list of fits files.

        :param files: list of strings with fits filenames
        :param kwargs: accepts the same keyword options as :py:func:`create_wdir`
        :return: instance of the basis class



        """
        
        obj = cls()
        _Creators.pynpoint_create_wfitsfiles(obj, files, **kwargs)
        obj.mk_basis_set()
        return obj
        



    def mk_basis_set(self):#,fileout = None):
        """
        creates basis set attributes using the images stored in im_arr
        """
        # if fileout is None:
        #     dir_in = os.path.dirname(self.files[0])
        #     filename = _Util.filename4mdir(dir_in,filetype='basis')
        
        basis_info_full = _Util.mk_basis_pca(self.im_arr)#,ave_sub=True)
        self.im_ave = basis_info_full['im_ave']
        self.im_arr = basis_info_full['im_arr']
        self.psf_basis = basis_info_full['basis']
        self.psf_basis_type = basis_info_full['basis_type']


    def mk_orig(self,ind):
        """Function for producing an original input image

        :param ind: index of the image to returned
        :return: 2D numpy array with the original input image


        """
        if self.cent_remove is True:
            imtemp = (self.im_arr[ind,] + self.im_ave + self.im_arr_mask[ind,]) * self.im_norm[ind]
        else:
            imtemp = (self.im_arr[ind,] + self.im_ave) * self.im_norm[ind]

        return imtemp

 
    def mk_psfmodel(self, num):
        """
        Makes a model of the PSF using its PSF basis.

        :param num: number of basis coefficients used in the fit

        """

        super(basis, self).mk_psfmodel(self, num)