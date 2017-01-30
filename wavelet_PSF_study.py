from PynPoint import Pypeline
import numpy as np
from PynPoint.processing_modules import PSFSubtractionModule
from PynPoint.io_modules import WriteAsSingleFitsFile

from PynPoint.processing_modules.PSFSubtraction import MakePSFModelModule, CreateResidualsModule


part = raw_input("Please enter denoising value: ")

pipeline = Pypeline("/scratch/user/mbonse/HR8799_2012_08_26_ND/working_files/08_klein_" + str(part) + "/",
                    "/scratch/user/mbonse/HR8799_2012_08_26_ND/working_files/08_klein_" + str(part) + "/",
                    "/scratch/user/mbonse/HR8799_2012_08_26_ND/results/")

# 08 PSF Subtraction and preparation

if str(part) == "0":
    wavelet_names = ["07_wavelet_denoised_0_0"]
elif str(part) == "1":
    wavelet_names = ["07_wavelet_denoised_0_8"]
elif str(part) == "2":
    wavelet_names = ["07_wavelet_denoised_1_0"]
else:
    wavelet_names = ["07_wavelet_denoised_1_2"]


'''
pca_numbers = range(1, 20, 1)
pca_numbers.extend(range(25, 90, 5))'''

pca_numbers = range(1, 30, 1)
pca_numbers.extend(range(32, 152, 2))

for denoising_result in wavelet_names:

    psf_subtraction = PSFSubtractionModule(0,
                                           name_in="PSF_sub_for_" + denoising_result,
                                           images_in_tag=denoising_result,
                                           reference_in_tag=denoising_result,
                                           res_mean_tag="08_res_mean_for_" + denoising_result +
                                                        "_" + str(0).zfill(2),  # 0 PCAs
                                           res_median_tag="08_res_median_for_" + denoising_result +
                                                        "_" + str(0).zfill(2),  # 0 PCAs
                                           res_rot_mean_clip_tag="08_res_rot_mean_clip_for_"
                                                                 + denoising_result +
                                                                 "_" + str(0).zfill(2),  # 0 PCAs
                                           prep_tag="prep_data",
                                           basis_out_tag="pca_basis",
                                           image_ave_tag="im_ave",
                                           cent_mask_tag="cent_mask",
                                           cent_size=0.07)
    pipeline.add_module(psf_subtraction)

    writing = WriteAsSingleFitsFile("08_res_mean_for_" + denoising_result +
                                    "_" + str(0).zfill(2) + ".fits",
                                    name_in="writing_" + "08_res_mean_for_" + denoising_result +
                                            "_" + str(0).zfill(2),
                                    data_tag="08_res_mean_for_" + denoising_result +
                                             "_" + str(0).zfill(2))
    pipeline.add_module(writing)

    for pca_number in pca_numbers:

        tmp_make_psf_model = MakePSFModelModule(num=pca_number,
                                                name_in="PSF_model_creation_for_" +
                                                        denoising_result + "_" + str(pca_number),
                                                im_arr_in_tag="prep_data",
                                                basis_in_tag="pca_basis",
                                                basis_average_in_tag="im_ave",
                                                psf_basis_out_tag="tmp_psf_model")

        tmp_residuals_module = \
            CreateResidualsModule(name_in="Creating_residuals_for_" +
                                          denoising_result + "_" + str(pca_number),
                                  im_arr_in_tag="prep_data",
                                  psf_im_in_tag="tmp_psf_model",
                                  mask_in_tag="cent_mask",
                                  res_mean_tag="08_res_mean_for_" + denoising_result +
                                               "_" + str(pca_number).zfill(2),
                                  res_median_tag="08_res_median_for_" + denoising_result +
                                               "_" + str(pca_number).zfill(2),
                                  res_rot_mean_clip_tag="08_res_rot_mean_clip_for_" +
                                                        denoising_result + "_" +
                                                        str(pca_number).zfill(2),)

        pipeline.add_module(tmp_make_psf_model)
        pipeline.add_module(tmp_residuals_module)

        # write out result as fits

        writing = WriteAsSingleFitsFile("08_res_mean_for_" + denoising_result +
                                        "_" + str(pca_number).zfill(2) + ".fits",
                                        name_in="writing_" + "08_res_mean_for_" + denoising_result +
                                                "_" + str(pca_number).zfill(2),
                                        data_tag="08_res_mean_for_" + denoising_result +
                                                 "_" + str(pca_number).zfill(2))
        pipeline.add_module(writing)

pipeline.run()
