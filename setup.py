"""Setup for the python package.""" 
from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh: 
     long_description = fh.read() 
setup(    
    author="Herearii Metuarea",    
    author_email="herearii.metuarea@gmail.com",
    name='napari-aphid', 
    description='A plugin to classify aphids by stage of development.',  
    version="1.1.3",    
    long_description=long_description, 
    long_description_content_type="text/markdown",  
    url='https://github.com/hereariim/napari-aphid', 
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=['napari',
    'numpy',
    'magicgui',
    'qtpy',
    'opencv-python-headless',
    'scikit-learn',
    'scikit-image',
    'h5py',
    'matplotlib',
    'pandas',
    'scipy'], 
    classifiers=[ 'Development Status :: 2 - Pre-Alpha',
    'Framework :: napari',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Image Processing',
                ],
     )