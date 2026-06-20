from setuptools import setup, find_packages

setup(
	name="pyfilters",
	version="1.0.0",
	description="Python bindings for C implementation of XOR filters",
	author="Vasileios Tsouvalas",
	author_email="v.tsouvalas@tue.nl",
	license="Apache 2.0",
	python_requires=">=3.0",
	packages=find_packages(),
	ext_package="pyfilters",
	install_requires=["cffi"],
	setup_requires=["cffi"],
	cffi_modules=["pyfilters/ffibuild.py:ffi"],
)
