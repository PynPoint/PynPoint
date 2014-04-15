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
import PynPlot as pynplot

def get_data_dir():
    """
    Returns the path to the data directory containing the example data sets.
    
    :returns path: String with path to the directory
    """
    from pkg_resources import resource_filename
    import PynPoint
    return resource_filename(PynPoint.__name__, "data")

def run(config,force_replace=False):
    """
    Delegates the execution to :py:meth:`workflow.run`
    
    :param config: name of the config file with details of the run to be executed
    :param force_replace: If True then the workspace directory will be overwritten if it already exists
    
    :returns ws: the instance of the workflow
    """
    ws = workflow.run(config, force_replace)
    return ws
    
def restore(dirin):
    """
    Delegates the execution to :py:meth:`workflow.restore`
    
    :param dirin: Work directory created by by an earlier calculation (using run method). 

    :returns ws: the instance of the workflow
    """
    ws = workflow.restore(dirin)
    return ws