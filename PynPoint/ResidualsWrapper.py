from PynPoint.WrapperUtils import BasePynpointWrapper


class ResidualsWrapper(BasePynpointWrapper):

    def __init__(self):
        pass

    @classmethod
    def create_restore(cls, filename):
        pass

    @classmethod
    def create_winstances(cls, images,basis):
        pass

    def res_arr(self,num_coeff):
        pass

    def res_rot(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_mean(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_median(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_mean_clip(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_var(self,num_coeff,extra_rot = 0.0):
        pass

    def _psf_im(self,num_coeff):
        pass

    def mk_psfmodel(self, num):
        pass
