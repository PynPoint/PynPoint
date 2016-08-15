.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "sdist - package"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

lint:
	flake8 PynPoint test

test:
	find . -name 'test/__pycache__' -exec rm -rf {} +
	py.test test/test_Images.py
	py.test test/test_basis.py
	py.test test/test_residuals.py
	py.test test/test_plotter.py
	py.test test/test_workflow.py
	py.test test/test_PynPoint_cache.py
	py.test test/test_PynPoint_ctx.py
	py.test test/test_Cli.py


test-all:
	tox

coverage:
	coverage run --source PynPoint setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/modules.rst
	rm -f docs/PynPoint.io_modules.rst
	rm -f docs/PynPoint.old_version.rst
	rm -f docs/PynPoint.processing_modules.rst
	rm -f docs/PynPoint.wrapper.rst
	sphinx-apidoc -o docs/ PynPoint
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

sdist: clean
#	pip freeze > requirements.rst
	python setup.py sdist
	ls -l dist