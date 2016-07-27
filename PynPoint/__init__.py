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

__author__ = 'Adam Amara'
__email__ = 'adam.amara@phys.ethz.ch'
__version__ = '0.2.0'
__credits__ = 'ETH Zurich, Institute for Astronomy'

# from PynPoint.Residuals import residuals
# from PynPoint.Basis import basis
# from PynPoint.Images import images

import PynPoint.old_version.PynPlot
import PynPoint.old_version._Cache as pynpointcache
import PynPoint.old_version._Ctx as pynpointctx
from PynPoint.wrapper.BasisWrapper import BasisWrapper as basis
from PynPoint.wrapper.ImageWrapper import ImageWrapper as images
from PynPoint.wrapper.ResidualsWrapper import ResidualsWrapper as residuals
from PynPoint.old_version.Workflow import workflow
from PynPoint.old_version._BasePynPoint import base_pynpoint


def get_data_dir():
    """
    Returns the path to the data directory containing the example data sets.
    
    :return: String with path to the directory
    """
    from pkg_resources import resource_filename
    import PynPoint
    return resource_filename(PynPoint.__name__, "data")

def run(config,force_replace=False):
    """
    Delegates the execution to :py:meth:`workflow.run`
    
    :param config: name of the config file with details of the run to be executed
    :param force_replace: If True then the workspace directory will be overwritten if it already exists
    
    :return: the instance of the workflow
    """
    ws = workflow.run(config, force_replace)
    return ws
    
def restore(dirin):
    """
    Delegates the execution to :py:meth:`workflow.restore`
    
    :param dirin: Work directory created by an earlier calculation (using run method).

    :return: the instance of the workflow
    """
    ws = workflow.restore(dirin)
    return ws