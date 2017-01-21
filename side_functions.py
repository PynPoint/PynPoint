from PynPoint import Pypeline

__author__ = 'Arianna'
from skimage.transform import rescale
import math
import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate,zoom
from PynPoint import Pypeline
from PynPoint.processing_modules.PSFsubPreparation import PSFdataPreparation
from PynPoint.core.Processing import ProcessingModule, PypelineModule
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile, Hdf5Reading
#from DatabaseAccess import DatabaseAccessModule
from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
DarkSubtractionModule, FlatSubtractionModule, CutTopTwoLinesModule, \
AngleCalculationModule, SimpleBackgroundSubtractionModule, \
StarExtractionModule, StarAlignmentModule, PSFSubtractionModule, \
StackAndSubsetModule

# ################################################################################################################################
# ################################################################################################################################


#2D gaussian fitter:
def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

################################################################################################################################
################################################################################################################################


#define a function that takes a point in the cartesian plane, takes and angle and calculate the rotated point around
#the origin point
def angular_coords_float(origin_point,start_point, degrees):
    rotation = math.radians(degrees)
    outx = origin_point[0]+(start_point[0]-origin_point[0])*math.cos(rotation) - (start_point[1]-origin_point[1])*math.sin(rotation)
    outy = origin_point[1]+(start_point[0]-origin_point[0])*math.sin(rotation) + (start_point[1]-origin_point[1])*math.cos(rotation)

    return np.array([outx,outy])

################################################################################################################################
################################################################################################################################


    ####Function that finds planets based on local maximum values
def planets_finder(image_dir,filetype,method,planet_position =[0,0],range_of_search = 0):

    ##### PARAMETER EXPLANATION #######
    ## image_dir = ARRAY or STRING = directory with the image stored as a fits file or array
    ## filetype = STRING = filetype of the input image, either 'array' or 'fits'
    ## method = STRING = method with which search for the planets, either 'global_max' (search the maximum of the entire
                        # image) or 'local_max' search for the maximum inside a region of given size and centered on
                        # planet_location
    ## planet_position = ARRAY = [x,y] = position around which search for the local maximum
    ## range_of_search = INTEGER = size of the region (centered on planet_position) inside which search for the maximum

    #Open the image depending on the filetype:
    if filetype=='array':
        image=image_dir
    if filetype=='fits':
        data = fits.open(image_dir)
        image=data[0].data
        # hdr = data[0].header

    # # Store the image dimension:
    # length_x = len(image[0])
    # length_y = len(image[1])

    #Find the maximum depending on the method input:
    if method=='local_max':
        resized_image = image[int(planet_position[1]-range_of_search/2.):
                              int(planet_position[1]+range_of_search/2.),

                              int(planet_position[0]-range_of_search/2.):
                              int(planet_position[0]+range_of_search/2.)]

        # scale image
        resized_image =  rescale(image=np.asarray(resized_image,
                                 dtype=np.float64),
                                 scale=(100,
                                        100),
                                 order=3,
                                 mode="reflect")

        position = [planet_position[1]-(range_of_search/2. - np.where(resized_image==resized_image.max())[0]/100.),
                    planet_position[0]-(range_of_search/2. - np.where(resized_image==resized_image.max())[1]/100.)]

        maximum = image[int(position[0][0]), int(position[1][0])]

    if method=='global_max':
        position=np.where(image==image.max())
        maximum = image[position][0]
        # resized_image=image

    # return both the position (n.b: the np.where function returns the x and y inverted, since it actually returns the
    # position as row and column, so they need to be interchanged before returning) of the maximum and its value:

    real_position = [position[1][0],position[0][0]]

    return real_position, maximum

################################################################################################################################
################################################################################################################################

#Let's create a function that, for a given magnitude contrast and planet position, create and insert the fake planet:

def create_fake_planet(dir_raw,psf_fits,dir_fake_planets,fake_planet_pos,mag,subpix_precision,
                    negative_flux,pc_number=0.,inner_mask_cent_size=0.,bad_frames=np.array([]),cutting_psf=False,
                    psf_cut=20.,plot=False,returnim=False,save=False,savefolder='None',first_angle_ext='None',psf_type='array'):
    ##################################################################

    #Let's inform the user:
    if negative_flux==True:
        print 'Inserting a fake negative planet at pixel position '+str(fake_planet_pos)+', with a magnitude contrast of '+str(mag)\
              + '\n(The image will be enlarged by a factor of '+str(subpix_precision)+')'
    if negative_flux==False:
        print 'Inserting a fake positive planet at pixel position '+str(fake_planet_pos)+', with a magnitude contrast of '+str(mag)\
              + '\n(The image will be enlarged by a factor of '+str(subpix_precision)+')'

    # Define Directories
    working_place_in = "/Users/Alex/polybox/Studium/Material/HS2016/Master_Thesis/Coding/Workplace"
    input_place_in = "/Users/Alex/polybox/Studium/Material/HS2016/Master_Thesis/Data/EpsEri/2007_12_17/Science0_33"
    output_place_in = "/Users/Alex/polybox/Studium/Material/HS2016/Master_Thesis/Data/EpsEri/2007_12_17/Results/PSF3px"

    # New Pipeline
    pipeline = Pypeline(working_place_in, input_place_in, output_place_in)

    #Let's import the data as a list of strings with the fits files names:
    fits_files_list=[fits_file for fits_file in os.listdir(dir_raw) if fits_file.endswith('.fits')]

    if first_angle_ext=='None':
        #Let's save the first angle (i.e.: the smallest angle:
        angles=np.zeros((len(fits_files_list)))
        # Uncomment for ADI
        # for i_angle in range(len(fits_files_list)):
        #     angle_i_i=fits.open(str(dir_raw)+str(fits_files_list[i_angle]))[0].header['NEW_PARA']
        #     angles[i_angle]=angle_i_i

        first_angle=np.min(angles)
    else:
        first_angle=first_angle_ext

    #Lets find the size of the image:
    image_size = np.shape(pipeline.get_data('im_arr_aligned0'))[1]

    #Let's calculate the sub pixel position, given the subpixel precision:
    sub_pos=np.array([fake_planet_pos[0] * subpix_precision,fake_planet_pos[1]*subpix_precision])

    print 'sub_pos = '+str(sub_pos[0])+' , '+str(sub_pos[1])

    #Let's empty the folder that (may) contains older fake planets:
    filelist_delete=os.listdir(dir_fake_planets)
    for filename_delete in filelist_delete:
        os.remove(dir_fake_planets+filename_delete)
    print 'The folder '+str(dir_fake_planets)+' has been emptied'

    #Calculate the flux reduction:
    if negative_flux==True:
        flux_red=-10**(-mag/2.5)
    if negative_flux==False: #To do the 3 sigma upper limit non detection
        flux_red=10**(-mag/2.5)

    # #import the PSF:
    if psf_type=='array':
        psf=psf_fits
    if psf_type=='fits':
        psf = pipeline.get_data('psf_model')

    center=[len(psf)/2.,len(psf)/2.] #Assumes centering already done
    #Cut the PSF if requested:
    if cutting_psf==True:
        star_psf=psf[center[1]-psf_cut/2.:center[1]+psf_cut/2.,center[0]-psf_cut/2.:center[0]+psf_cut/2.]
    else:
        star_psf=psf

    #Multiply the psf by the reduction factor:
    planet_PSF = flux_red*star_psf

    #Let's do subpixel sampling: decide the zoom factor
    return_factor=1./subpix_precision

    #Insert the planet PSF:
    #Enlarge the planet psf image:
    planet_PSF_enlarged=zoom(planet_PSF,subpix_precision)
    print 'I have enlarged the planet PSF by a factor of '+str(subpix_precision)
    # print '\n'

    #Create an image with the same size of the fits files, but filled with zeros (to be used to insert the fake planet):
    fake_image= np.zeros([image_size*subpix_precision,image_size*subpix_precision])
    # print 'I have created a void image of size '+str(np.shape(fake_image))

    fake_image[sub_pos[1]-len(planet_PSF_enlarged[1])/2:sub_pos[1]+len(planet_PSF_enlarged[1])/2,
    sub_pos[0]-len(planet_PSF_enlarged[0])/2:sub_pos[0]+len(planet_PSF_enlarged[0])/2]=planet_PSF_enlarged[:,:]

    # print 'I have inserted the planet at the position '+str(pos_int)
    print 'first angle : '+str(first_angle)
    for i in range(len(fits_files_list)): #(--> THIS CAN BE PARALLELIZED)
        #Ignore the bad frames if there are some:
        if i in bad_frames:
            print 'Frame '+str(i)+' will be ignored'
        else:
            #let's read in the angle of rotation for this frame from its header:

            angle_i = 0. #fits.open(str(dir_raw)+str(fits_files_list[i]))[0].header['NEW_PARA']-first_angle

            #Let's rotate the images:
            fake_image_i=rotate(fake_image,angle_i,reshape=False)

            #Let's add this to the raw data:
            data_all=fits.open(str(dir_raw)+str(fits_files_list[i]))
            data=data_all[0].data
            data_header=data_all[0].header

            #Let's enlarge the image of the desired zoom factor:
            data_enlarge=zoom(data,subpix_precision)

            #Let's add the raw data (enlarged) and the void image with the fake planet:
            new_data=data_enlarge+fake_image_i

            #Shrink everything down again:
            data_shrinked_again=zoom(new_data,return_factor)

            #Save to Database:
            # os.chdir(dir_fake_planets)
            # savefits=fits.PrimaryHDU(data_shrinked_again,header=data_header)
            # name='fake_'+str(i)+'.fits'
            # savefits.writeto(name,clobber=1)
            #access = DatabaseAccessModule()

    print 'The fake planets have been inserted into the raw data, and saved into the folder '+str(dir_fake_planets)

    if returnim==True or plot==True or save==True:

        print '\nRunning PynPoint on the new set of fits file with the fake planet...\n'

        ###############################
        # Old Version of PnyPoint     #
        ###############################
        #Let's run Pynpoint on the fits file with the fake planet:
        basis=PynPoint.basis.create_wdir(dir_fake_planets,cent_size=inner_mask_cent_size)
        # basis.save(dir_fake_planets+'basis_savefile.hdf5')
        # print('basis created')

        images=PynPoint.images.create_wdir(dir_fake_planets,cent_size=inner_mask_cent_size)
        # images.save(dir_fake_planets+'images_savefile.hdf5')
        # print 'Images created'

        res_fake=PynPoint.residuals.create_winstances(images,basis)
        # res_fake.save(dir_fake_planets+'res_savefile.hdf5')
        # print 'residuals created'

        res_for_conv=PynPoint.residuals.res_rot(res_fake,pc_number)
        res_for_conv_int=res_for_conv[:]
        # res_fake_to_use=np.delete(res_for_conv_int,bad_frames_int,axis=0)
        # print 'I am using '+str(len(res_for_conv_int))+' frames'
        # print str(bad_frames)

        mean_image=np.mean(res_for_conv_int,axis=0)

        ###############################
        # New Version of PnyPoint     #
        ###############################
        psf_sub = PSFSubtractionModule(pca_number=pc_number,
                                       name_in="PSF_subtraction",
                                       images_in_tag="im_arr_aligned33",
                                       reference_in_tag="im_arr_aligned0",
                                       res_mean_tag="res_mean")

    #Plot (if requested):
    if plot==True:
        plt.figure()
        plt.subplot(111)
        plt.imshow(mean_image,origin='lower')
        plt.show()

    if save==True:
        saveimage=fits.PrimaryHDU(mean_image)
        if negative_flux==True:
            savename='fake_neg_'+str(fake_planet_pos[0])+'_'+str(fake_planet_pos[1])+'_'+str(mag)+'.fits'
        if negative_flux==False:
            savename='fake_pos_'+str(fake_planet_pos[0])+'_'+str(fake_planet_pos[1])+'_'+str(mag)+'.fits'

        saveimage.writeto(str(savefolder)+str(savename),clobber=1)

        print 'The image with the fake planet has been saved as a fits file in '+str(savefolder)+str(savename)+'\n'

    if returnim==True:
        return (mean_image)
    else:
        return ()


####################################################################################################################
####################################################################################################################


