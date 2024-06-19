from setuptools import find_packages, setup

setup(
    name='corner_training',
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        'torch',
        'numpy',
        'h5py',
        'opencv-python',
        'openpyxl',
        'click',
        'scikit-learn',
        'scikit-image',
    ]
)
