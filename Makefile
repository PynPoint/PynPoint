.PHONY: help clean clean-build clean-python clean-test test test-all coverage docs

help:
	@echo "clean - remove all artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-python - remove Python artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "test - run test cases"
	@echo "test-all - run tests with tox"
	@echo "coverage - check code coverage"
	@echo "docs - generate Sphinx documentation"

clean: clean-build clean-python clean-test

clean-build:
	rm -rf PynPoint_exoplanet.egg-info/
	rm -rf dist/
	rm -rf build/
	rm -rf htmlcov/
	rm -rf .eggs/
	rm -rf docs/_build

clean-python:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -f coverage.xml
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .tox/
	rm -rf PynPoint.egg-info/
	rm -f junit-docs-ci.xml
	rm -f junit-py27.xml
	rm -f junit-py36.xml
	rm -f junit-py37.xml
	rm -rf .pytest_cache/

test:
	pytest

test-all:
	tox

coverage:
	coverage run --rcfile .coveragerc -m py.test
	coverage combine
	coverage report -m
	coverage html

docs:
	rm -f docs/PynPoint.Core.rst
	rm -f docs/PynPoint.IOmodules.rst
	rm -f docs/PynPoint.ProcessingModules.rst
	rm -f docs/PynPoint.Util.rst
	sphinx-apidoc -o docs/ PynPoint
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
