# python geopack  setup.py
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ngeopack',
    version='1.0.0',
    author='Zach Beever',
    author_email='zbeever@bu.edu',
    description='Numba version of geopack and Tsyganenko models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url= 'https://github.com/zbeever/geopack',
    requires= ['numpy', 'scipy', 'numba'],
    license= 'MIT',
    keywords= ['geopack','space physics','Tsyganenko model'],
    packages= setuptools.find_packages(),
    package_data={'':['*.txt','*.md']},
    classifiers= [
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    python_requires='>=3.6',
)
