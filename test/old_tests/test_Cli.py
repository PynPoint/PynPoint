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
Created on Apr 14, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import

import contextlib
import sys

from mock import patch

from PynPoint import _Cli

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