import warnings

from pynpoint.core.pypeline import Pypeline

from pynpoint.processing.background import SimpleBackgroundSubtractionModule, \
                                           MeanBackgroundSubtractionModule, \
                                           PCABackgroundPreparationModule, \
                                           PCABackgroundSubtractionModule, \
                                           DitheringBackgroundModule, \
                                           NoddingBackgroundModule

from pynpoint.processing.badpixel import BadPixelSigmaFilterModule, \
                                         BadPixelInterpolationModule, \
                                         BadPixelMapModule, \
                                         BadPixelTimeFilterModule, \
                                         ReplaceBadPixelsModule

from pynpoint.processing.basic import SubtractImagesModule, \
                                      AddImagesModule, \
                                      RotateImagesModule

from pynpoint.processing.centering import StarAlignmentModule, \
                                          StarExtractionModule, \
                                          StarCenteringModule, \
                                          ShiftImagesModule, \
                                          WaffleCenteringModule

from pynpoint.processing.darkflat import DarkCalibrationModule, \
                                         FlatCalibrationModule

from pynpoint.processing.frameselection import RemoveFramesModule, \
                                               FrameSelectionModule, \
                                               RemoveLastFrameModule, \
                                               RemoveStartFramesModule

from pynpoint.processing.fluxposition import FakePlanetModule, \
                                             SimplexMinimizationModule, \
                                             FalsePositiveModule, \
                                             MCMCsamplingModule, \
                                             AperturePhotometryModule

from pynpoint.processing.limits import ContrastCurveModule

from pynpoint.processing.psfpreparation import PSFpreparationModule, \
                                               AngleInterpolationModule, \
                                               AngleCalculationModule, \
                                               SortParangModule, \
                                               SDIpreparationModule

from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule

from pynpoint.processing.resizing import CropImagesModule, \
                                         ScaleImagesModule, \
                                         AddLinesModule, \
                                         RemoveLinesModule

from pynpoint.processing.stacksubset import StackAndSubsetModule, \
                                            MeanCubeModule, \
                                            DerotateAndStackModule, \
                                            CombineTagsModule

from pynpoint.processing.timedenoising import CwtWaveletConfiguration, \
                                              DwtWaveletConfiguration, \
                                              WaveletTimeDenoisingModule, \
                                              TimeNormalizationModule

from pynpoint.readwrite.fitsreading import FitsReadingModule

from pynpoint.readwrite.fitswriting import FitsWritingModule

from pynpoint.readwrite.hdf5reading import Hdf5ReadingModule

from pynpoint.readwrite.hdf5writing import Hdf5WritingModule

from pynpoint.readwrite.textwriting import AttributeWritingModule, \
                                           ParangWritingModule, \
                                           TextWritingModule

from pynpoint.readwrite.textreading import ParangReadingModule

warnings.simplefilter('always', DeprecationWarning)

__author__ = 'Tomas Stolker, Markus Bonse, Sascha Quanz, and Adam Amara'
__license__ = 'GPLv3'
__version__ = '0.6.0'
__maintainer__ = 'Tomas Stolker'
__email__ = 'tomas.stolker@phys.ethz.ch'
__status__ = 'Development'
