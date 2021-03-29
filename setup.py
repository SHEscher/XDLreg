"""Setup XDLreg package."""

import pathlib
from setuptools import setup  # , find_packages

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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['PumpkinNet', 'LRP'],  # OR find_packages(exclude=("...",))
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "seaborn", "scikit-image", "Keras==2.2.4", "innvestigate"],
    entry_points={
        "console_scripts": [
            "xdlreg=XDLreg.__main__:main",
        ]
    },
)
