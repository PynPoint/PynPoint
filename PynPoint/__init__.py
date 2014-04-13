__author__ = 'Adam Amara'
__email__ = 'adam.amara@phys.ethz.ch'
__version__ = '0.1.1'
__credits__ = 'ETH Zurich, Institute for Astronomy'



from Basis import basis
from Images import images
from Residuals import residuals
from Workflow import workflow

from _BasePynPoint import base_pynpoint
import _Ctx as pynpointctx
import _Cache as pynpointcache
import Plotter as plotter

from Workflow import run, restore

def get_data_dir():
    """
    Returns the path to the data directory containing the example data sets.
    
    :returns path: String with path to the directory
    """
    from pkg_resources import resource_filename
    import PynPoint
    return resource_filename(PynPoint.__name__, "data")