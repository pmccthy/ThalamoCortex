from setuptools import setup, find_packages

setup(
    name="thalamocortex",  # Change this to your package name
    version="0.1.0",
    author="Patrick McCarthy",
    author_email="patricj.mccarthy@dtc.ox.ac.uk",
    description="Deep learning models of thalamocortical circuits.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_project",  # Change to your repo
    packages=find_packages(),  # Automatically finds all packages in the directory
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.16",
)