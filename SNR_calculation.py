import numpy as np
import glob
from matplotlib import pyplot as plt
import pyfits
import copy as cp
from skimage.transform import rescale
import numpy as np

class SignalToNoise(object):

    """
    Noise calculation according to Mawet (2014). It calculates the S/N value
    and the associated p-value for the one sided t-test. The p-value
    corresponds to the false-alarm-probability calculated from the cdf of the
    t-distribution.
    p-values for Gaussian S/N test:
        3-sigma detection: 0.0013 false-alarm-probability
        5-sigma detection: 2.9e-07 false-alarm-probability
    The program finds all files in the input directory and then creates a
    signal and a noise pixels mask for the first image. The same masks are then
    used to analyze all images. Only *.fits files in the input directory are
    allowed. The star needs to be centered in the middle of the image.

        :param dir_in: path to the directory containing the images
        :param pl_pos: position of the planet in the images
        :param pl_size: size of the planet PSF (usually corresponds the FWHM or 1 lambda/D)
        :param tot_size: the area around the planet position which should be excluded from the noise
         analysis (for example due to oversubtraction in this area). Should always be equal to or
         larger than pl_size
    """

    def __init__(self,
                 dir_in,
                 pl_pos,
                 pl_size,
                 tot_size,
                 scaling):

        self.dir_in = dir_in
        self.files = sorted(glob.glob(self.dir_in + '/*.fits'))

        # make the masks
        img = pyfits.getdata(self.files[0])

        # scaling
        img = rescale(image=np.asarray(img,
                                       dtype=np.float64),
                      scale=(scaling,
                             scaling),
                      order=3,
                      mode="reflect")

        self.scaling = scaling
        self.pl_pos = (pl_pos[0] * scaling, pl_pos[1] * scaling)
        self.pl_size = int(pl_size * scaling)
        self.tot_size = tot_size * scaling

        self.mk_noise_mask(img,
                           pl_pos=self.pl_pos,
                           pl_size=self.pl_size,
                           tot_size=self.tot_size)

        self.mk_signal_mask(img,
                            pl_pos=self.pl_pos,
                            pl_size=self.pl_size)

        snr_list = []

        # calculate signal to noise for every image separately
        for ind in xrange(np.size(self.files)):

            # read current image and determine signal and noise pixels
            img = pyfits.getdata(self.files[ind])
            img = rescale(image=np.asarray(img,
                                           dtype=np.float64),
                          scale=(scaling,
                                 scaling),
                          order=3,
                          mode="reflect")

            noise_pixels = img[np.where(self.noise_mask == 1)]
            signal_pixels = img[np.where(self.signal_mask == 1)]

            # calculate signal to noise for the current image
            noise = np.std(noise_pixels) * np.sqrt(1. + 1. / self.nr_of_noisesamples)
            signal = np.sum(signal_pixels) / np.size(signal_pixels) - np.sum(
                noise_pixels) / np.size(noise_pixels)
            snr = signal / noise

            print snr
            '''
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.imshow(img, origin = 'lower', interpolation = 'nearest', vmin = -4.e-6, vmax = 10.e-6)
            fig.add_subplot(1,3,2)
            plt.imshow(self.noise_mask + self.signal_mask, origin = 'lower', interpolation = 'nearest')
            fig.add_subplot(1,3,3)
            plt.imshow(img * (1. - (self.noise_mask + self.signal_mask)), origin = 'lower', interpolation = 'nearest', vmin = -4.e-6, vmax = 10.e-6)
            plt.savefig(dir_in + str(ind) + "_Signal_to_noise.pdf")
            plt.close()'''

            snr_list.append(snr)

        plt.plot(snr)
        plt.savefig(dir_in + "False-alarm-probability.svg")

    def mk_noise_mask(self,
                      img,
                      pl_pos,
                      pl_size,
                      tot_size):
        """
        Creates a mask with ones in a few circular masks around the star and
        zeros everywhere else. The masks have the same separation from the star
        and cover the same are as the signal mask around the planet position.
        This method creates the noise pixel mask and also returns the number
        of circular noise mask which were created for the following statistical
        analysis.
        """

        # determine the center of the image, create the temporary mask and
        # determine the radial distance between star and planet in pixels
        center = [img.shape[0] / 2, img.shape[1] / 2]
        temp_mask = np.zeros(img.shape)
        pl_rad = np.sqrt((pl_pos[0] - center[0]) ** 2 + (pl_pos[1] - center[1]) ** 2)
        # print pl_rad / (4.8e-6/8.2*180/np.pi*3600/0.027190*2)

        # find all pixels with equal radial distance as the planet
        for x_ind in xrange(img.shape[0]):
            for y_ind in xrange(img.shape[1]):
                if (x_ind - center[0]) ** 2 + (y_ind - center[1]) ** 2 >= (
                    pl_rad - 1 / 2.) ** 2 and (x_ind - center[0]) ** 2 + (
                    y_ind - center[1]) ** 2 <= (pl_rad + 1 / 2.) ** 2:
                    temp_mask[x_ind, y_ind] = 1

        # exclude the pixels which are too near to the position of the planet
        for x_ind in xrange(img.shape[0]):
            for y_ind in xrange(img.shape[1]):
                if (x_ind - pl_pos[0]) ** 2 + (y_ind - pl_pos[1]) ** 2 <= (
                        tot_size / 2. + pl_size / 2.) ** 2:
                    temp_mask[x_ind, y_ind] = 0

        # now take only the subset of pixels which are separated by the size
        # of circular mask. the result is a mask where only the central pixels
        # of the circular noise masks are non-zero
        comp_pos = cp.copy(pl_pos)
        while np.size(np.where(temp_mask == 1)[0]) != 0:
            temp_sample_pixels = np.where(temp_mask == 1)
            pixel_dist = np.zeros(np.size(temp_sample_pixels[0]))
            for ind in xrange(np.size(temp_sample_pixels[0])):
                pixel_dist[ind] = (comp_pos[0] - temp_sample_pixels[0][ind]) ** 2 + (comp_pos[1] -
                                                                                     temp_sample_pixels[
                                                                                         1][
                                                                                         ind]) ** 2
            x_ind = temp_sample_pixels[0][pixel_dist.argmin()]
            y_ind = temp_sample_pixels[1][pixel_dist.argmin()]
            temp_mask[x_ind - pl_size:x_ind + pl_size + 1, y_ind - pl_size:y_ind + pl_size + 1] = 0
            temp_mask[x_ind, y_ind] = 2
            comp_pos = [x_ind, y_ind]

        # create the small circular noise masks around every non-zero pixel
        samples = np.where(temp_mask == 2)
        for ind in xrange(np.size(samples[0])):
            for x_ind in xrange(img.shape[0]):
                for y_ind in xrange(img.shape[1]):
                    if (x_ind - samples[0][ind]) ** 2 + (y_ind - samples[1][ind]) ** 2 <= (
                        pl_size / 2.) ** 2:
                        temp_mask[x_ind, y_ind] = 1

        self.nr_of_noisesamples = np.size(samples[0])
        self.noise_mask = temp_mask

    def mk_signal_mask(self, img, pl_pos, pl_size):
        """
        Creates a mask with ones in a circular area around the position where
        the planet is supposed to be and zeros everywhere else.
        """

        temp_mask = np.zeros(img.shape)
        for x_ind in xrange(img.shape[0]):
            for y_ind in xrange(img.shape[1]):
                if (x_ind - pl_pos[0]) ** 2 + (y_ind - pl_pos[1]) ** 2 <= (pl_size / 2.) ** 2:
                    temp_mask[x_ind, y_ind] = 1

        self.signal_mask = temp_mask

