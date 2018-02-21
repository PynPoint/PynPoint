.PHONY: help clean clean-build clean-pyc clean-test lint test test-all coverage coverage_file docs sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "sdist - package"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -rf .tox/
	rm -rf htmlcov/

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
	rm -f docs/PynPoint.Wrapper.rst
	sphinx-apidoc -o docs/ PynPoint
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

sdist: clean
#	pip freeze > requirements.rst
	python setup.py sdist
	ls -l dist