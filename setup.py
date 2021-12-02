from glob import glob
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_modules = [
    Pybind11Extension("_solvers",
                      sorted(glob("tsp/solvers/src/*.cpp")),
    ),
]
    
setuptools.setup(
    name="cse6140-tsp",
    version="0.0.1",
    author="..",
    author_email="..",
    description="CSE 6140 final project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cse6140-project/cse-6140-project",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
