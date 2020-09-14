from setuptools import setup

with open('README.md') as f:
	readme = f.read()

with open('LICENSE') as f:
	license = f.read()

setup(
	name = "pyGEMS_1D",
	version = 0.1,
	author = "Christopher R. Wentland",
	author_email = "chriswen@umich.edu",
	url = "https://github.com/cwentland0/pyGEMS_1D", 
	description = "One-dimension reacting flow with ROMs",
	long_description = readme,
	license = license,
	install_requires = ['numpy', 'scipy', 'matplotlib'],
	python_requires = ">=3.5",
)