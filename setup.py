import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    package_dir={"": "tsp"},
    packages=setuptools.find_packages(where="tsp"),
    python_requires=">=3.6",
)