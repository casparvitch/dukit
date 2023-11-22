import setuptools
import pathlib
import site
import sys
import sysconfig

# odd bug with develop (editable) installs, see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

required = [
    "setuptools-git",  # see link at bottom: allows to specify package data to keep/upload/etc.
    "numpy",
    "matplotlib>=3.4.0",
    "scipy>=1.7",  # requires scipy.stats.qmc for magsim
    "numba",
    "psutil",
    "matplotlib-scalebar>=0.7.2",
    "tqdm",
    "simplejson",
    "pandas",  # could be replaced with lighter csv reader
    "rebin",
    "pyfftw",
    "PySimpleGUI",  # magsim
    "foronoi",  # magsim
    "python-polylabel",  # magism
    "dill",  # for magsim
    "astropy",  # remove if remove sigma_clip background sub
    "PyQt6", # Needed for gui widget stuff
    "scikit-image"
]

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")
arch = sysconfig.get_platform().replace("-", "_").replace(".", "_")

if __name__ == "__main__":
    setuptools.setup(
        name="dukit",
        version="0.0.1",
        author="Sam Scholten",
        author_email="samcaspar@gmail.com",
        description="Defect μscopy toolkit",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/casparvitch/dukit",
        keywords=[
            "NV",
            "QDM",
            "Diamond",
            "Quantum",
            "Quantum Sensing",
            "gpufit",
            "Widefield Defect Microscopy"
        ],
        license="MIT",
        package_dir={"": "dukit"},
        packages=setuptools.find_packages(
            where="src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
        ),
        install_requires=required,
        # python_requires=">=3.8",  # check pyfftw
        package_data={"": ["*.md", "*.json"]},
        setup_requires=["wheel"],
        extras_require={
            "cpufit": [f"pycpufit @ file://localhost/{here}/ext_wheels/pyCpufit-101.2.0-py2.py3-none-{arch}.whl"],
            "gpufit": [f"pygpufit @ file://localhost/{here}/ext_wheels/pyGpufit-101.2.0-py2.py3-none-{arch}.whl"],
        }
    )
# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html

# https://stackoverflow.com/questions/35668295/how-to-include-and-install-local-dependencies-in-setup-py-in-python
# https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies
# https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies 
