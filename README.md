# The geopack and Tsyganenko models in Numba

This is a modification of Sheng Tian's Python port of the geopack and Tsyganenko models, located at https://github.com/tsssss/geopack. The major differences are

1. Global variables have been removed in preference of passing and returning parameters.

2. The Bessel functions of the first and second kind have been recoded in Python and decorated with `@njit`.

You will need Numba to use this package. You do not need Sheng Tian's original port, nor do you need the original geopack and Tsyganenko Fortran files.

## Known issues
This fork was made primarily with the Tsyganenko models in mind, so most of the geopack functions are not yet Numba compatible.

## Installation
Using the terminal, type `pip install {path-to-directory-containing-setup.py}` This package can be removed using `pip uninstall ngeopack`

## Usage

For usage and notes, please see Sheng Tian's excellent introduction: https://github.com/tsssss/geopack

The major differences are 

1. The package is now named `ngeopack`

2. `igrf_gsm()` accepts arguments returned from `recalc()`, while these parameters were previously set and read globally.
