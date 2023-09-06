#!/usr/bin/env python

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hippo_Spin_Testing",
    version="0.0.1",
    author="Bradley Karat",
    author_email="bradleykarat@gmail.com",
    description="A tool for spin permutation testing using HippUnfold",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Bradley-Karat/Hippo_Spin_Testing",
    packages=setuptools.find_packages(),
    license="BSD 3-Clause License",
    package_data={
        "resources": ["*"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "nibabel",
        "numpy>=1.16.5",
        "scipy>=1.3.3",
        "matplotlib>=2.0.0",
    ],
    #dependency_links = [
     #"git+git://github.com/jordandekraker/hippunfold_toolbox.git#egg=hippunfold_toolbox",
    #],
    include_package_data=True,
    zip_safe=False,
)
