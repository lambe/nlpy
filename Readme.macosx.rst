=====================================
Comments pertaining to Mac OS/X users
=====================================

Installing NLPy will be much easier if you use Homebrew
(http://mxcl.github.com/homebrew). Follow the instructions to install Homebrew.
Then, the following dependencies can be installed automatically in /usr/local::

    brew install adol-c --enable-sparse     # will also install Colpack
    brew install cppad
    brew install boost --build-from-source  # to use pycppad
    brew install asl                        # instead of libampl
    brew install metis
    brew install gfortran


Troubleshooting
===============

1) If compiling on Leopard (10.5.*), you may receive the error message::

     undefined symbol: _strtod_ASL

   The problem is coming from the flags used to build the LibAmpl libraries.
   The solution is to add the flag::

     -mmacosx-version-min=10.4

   to the compiler options and to rebuild LibAmpl, i.e., switch to $LIBAMPL and
   issue the command 'make mrclean ; make' at the command line.

2) If compiling on Snow Leopard (10.6.*), you may receive error messages
   related to the architecture being linked (and the build might fail).  One
   solution seems to be to set the environment variables CFLAGS, FFLAGS and
   LDFLAGS appropriately before building. For example::

     CFLAGS='-arch x86_64 -mmacosx-version-min=10.5' \
     FFLAGS='-arch x86_64 -mmacosx-version-min=10.5' \
     LDFLAGS='-arch x86_64 -Wall -undefined dynamic_lookup -bundle -mmacosx-version-min=10.5' \
     python setup.py build config_fc --fcompiler=gfortran

   (all on one line). See, e.g., http://goo.gl/LNaA

   Note that LibAmpl should have been built using the same flags.
