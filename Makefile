.PHONY: help clean clean-build clean-pyc clean-test lint test test-all coverage coverage_file docs sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run test cases"
	@echo "test-all - run tests with tox"
	@echo "coverage - check code coverage"
	@echo "docs - generate Sphinx documentation"
	@echo "sdist - create a source distribution"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf PynPoint_exoplanet.egg-info/
	rm -rf dist/
	rm -rf htmlcov/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .tox/
	rm -rf PynPoint_exoplanet.egg-info/
	rm -f junit-docs-ci.xml
	rm -f junit-py27.xml

lint:
	flake8 PynPoint test

test:
	py.test

test-all:
	tox

coverage:
	coverage run --source PynPoint -m py.test
	coverage report -m --omit=PynPoint/OldVersion/*
	coverage html --omit=PynPoint/OldVersion/*
	open htmlcov/index.html

coverage_file:
	coverage run --source PynPoint -m py.test ${File}
	coverage report -m --omit=PynPoint/OldVersion/*
	coverage html --omit=PynPoint/OldVersion/*
	open htmlcov/index.html

docs:
	rm -f docs/modules.rst
	rm -f docs/PynPoint.IOmodules.rst
	rm -f docs/PynPoint.OldVersion.rst
	rm -f docs/PynPoint.ProcessingModules.rst
	rm -f docs/PynPoint.Util.rst
	rm -f docs/PynPoint.Wrapper.rst
	sphinx-apidoc -o docs/ PynPoint
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

sdist: clean
	python setup.py sdist