# Copyright (C) 2013 ETH Zurich, Institute for Astronomy

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
