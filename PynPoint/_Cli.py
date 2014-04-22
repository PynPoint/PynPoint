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

'''
Created on Apr 7, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import PynPoint
from PynPoint import _Util

def run():
    """
    Called by the entry point script. Delegating call to main()
    """
    _main(*sys.argv[1:])

def _main(*argv):
    
    if(argv is None or len(argv)<1):
        _usage()
        return
    argv = list(argv)
    force_replace = False
    if len(argv) > 1:
        force_replace = _Util.str2bool(argv[1])
        
    PynPoint.run(argv[0], force_replace)

def _usage():
    usage = """
    **PynPoint**
    Copyright (c) 2014 ETH Zurich, Institute for Astronomy
    
    Usage:
    PynPoint <configuration> <force_replace>
    
    example:
    - PynPoint workflow.config True
    """
    print(usage)
    
if __name__ == "__main__":
    _main(*sys.argv[1:])
