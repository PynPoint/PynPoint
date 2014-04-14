# Copyright (C) 2013 ETH Zurich, Institute for Astronomy

'''
Created on Apr 14, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import
from PynPoint import _Cli
import contextlib
import sys
from mock import patch

"""
Tests for `ivy.cli` module.

author: jakeret
"""
import pytest
import StringIO

@contextlib.contextmanager
def stdout_redirect(where):
    sys.stdout = where
    try:
        yield where
    finally:
        sys.stdout = sys.__stdout__

# @pytest.skip("temp")
class TestCli(object):


    def test_launch_empty(self):
        string_io = StringIO.StringIO()
        with stdout_redirect(string_io):
            _Cli._main(*[])
        output = string_io.getvalue()
        assert output is not None
        assert len(output) > 0
        string_io.close()
        
    def test_launch_workflow(self):
        with patch("PynPoint.Workflow.workflow") as wf_mock:
            _Cli._main(*["config"])
            _Cli._main(*["config", "True"])
            _Cli._main(*["config", "False"])
            
if __name__ == '__main__':
    pytest.main("-k TestCli")