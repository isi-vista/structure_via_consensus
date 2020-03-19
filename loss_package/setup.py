import os
import setuptools

requirements = [
    'torch>=1.1.0',
    'numpy>=1.15.4',
    ]
setuptools.setup(
    name="structure_via_consenus",
    version="1.0.0",
    author="Iacopo Masi",	
    author_email="masi@isi.edu",
    description="Loss function to induce smoothness in the training of a DCNN for semantic segmentation",
    long_description='#Loss function to induce smoothness in the training of a DCNN for semantic segmentation',
    long_description_content_type="text/markdown",
    url="https://github.com/isi-vista/structure_via_consensus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: USC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
