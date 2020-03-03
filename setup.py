from setuptools import find_packages
from distutils.core import setup

version = "0.0.1"

setup(
    name="intellino",
    version=version,
    license='GPLv3',
    keywords=["intellino"],
    author="SoC Platform Lab",
    author_email="hwangdonghyun@seoultech.ac.kr",
    description=("The Intellino core logic wrapper for ML simulation with intellino"),

    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=True,

    url="https://github.com/Intellino/intellino",
    packages=find_packages('intellino'),
    package_dir={'': 'intellino'},
    classifiers=[
        'Natural Language :: English',

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Development Status :: 3 - Alpha'
    ],
    
)
