import warnings

from pynpoint.core.pypeline import Pypeline

from pynpoint.processing.background import SimpleBackgroundSubtractionModule, \
                                           MeanBackgroundSubtractionModule, \
                                           LineSubtractionModule, \
                                           NoddingBackgroundModule

from pynpoint.processing.badpixel import BadPixelSigmaFilterModule, \
                                         BadPixelInterpolationModule, \
                                         BadPixelMapModule, \
                                         BadPixelTimeFilterModule, \
                                         ReplaceBadPixelsModule

from pynpoint.processing.basic import SubtractImagesModule, \
                                      AddImagesModule, \
                                      RotateImagesModule, \
                                      RepeatImagesModule

from pynpoint.processing.centering import StarAlignmentModule, \
                                          FitCenterModule, \
                                          ShiftImagesModule, \
                                          WaffleCenteringModule

from pynpoint.processing.darkflat import DarkCalibrationModule, \
                                         FlatCalibrationModule

from pynpoint.processing.extract import StarExtractionModule, \
                                        ExtractBinaryModule

from pynpoint.processing.filter import GaussianFilterModule

from pynpoint.processing.fluxposition import FakePlanetModule, \
                                             SimplexMinimizationModule, \
                                             FalsePositiveModule, \
                                             MCMCsamplingModule, \
                                             AperturePhotometryModule, \
                                             SystematicErrorModule

from pynpoint.processing.frameselection import RemoveFramesModule, \
                                               FrameSelectionModule, \
                                               RemoveLastFrameModule, \
                                               RemoveStartFramesModule, \
                                               ImageStatisticsModule, \
                                               FrameSimilarityModule, \
                                               SelectByAttributeModule, \
                                               ResidualSelectionModule

from pynpoint.processing.limits import ContrastCurveModule, \
                                       MassLimitsModule

from pynpoint.processing.pcabackground import PCABackgroundPreparationModule, \
                                              PCABackgroundSubtractionModule, \
                                              DitheringBackgroundModule

from pynpoint.processing.psfpreparation import PSFpreparationModule, \
                                               AngleInterpolationModule, \
                                               AngleCalculationModule, \
                                               SortParangModule, \
                                               SDIpreparationModule

from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule, \
                                               ClassicalADIModule

from pynpoint.processing.resizing import CropImagesModule, \
                                         ScaleImagesModule, \
                                         AddLinesModule, \
                                         RemoveLinesModule

from pynpoint.processing.stacksubset import StackAndSubsetModule, \
                                            StackCubesModule, \
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

from pynpoint.readwrite.textreading import ParangReadingModule, \
                                           AttributeReadingModule

from pynpoint.readwrite.nearreading import NearReadingModule

warnings.simplefilter('always', DeprecationWarning)

__author__ = 'Tomas Stolker & Markus Bonse'
__license__ = 'GPLv3'
__version__ = '0.8.2'
__maintainer__ = 'Tomas Stolker'
__email__ = 'tomas.stolker@phys.ethz.ch'
__status__ = 'Development'
