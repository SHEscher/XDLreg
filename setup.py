"""Setup XDLreg package."""

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="XDLreg",
    version="1.0.0",
    description="Simulation study on explainable deep learning (XDL) for regression tasks.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/SHEscher/XDLreg",
    author="Simon M. Hofmann",
    author_email="simon.hofmann@pm.me",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        # "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy==1.18.0", "matplotlib==3.3.4", "seaborn==0.11.1", "scikit-image==0.16.2",
                      "h5py==2.10.0", "Keras==2.2.4", "tensorflow==1.14.0rc1", "innvestigate==1.0.8"],
    entry_points={
        "console_scripts": [
            "xdlreg=xdlreg.__main__:main",
        ]
    },
)
