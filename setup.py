#!/usr/bin/python
"""   
desc:  Setup script for 'fitting' package.
auth:  Craig Wm. Versek (cversek@physics.umass.edu)
date:  12/02/2009
notes: install with "sudo python setup.py install"
"""

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages


INSTALL_REQUIRES = []

def ask_yesno(prompt, default='y'):
    while True:
        full_prompt = prompt + "([y]/n): "
        val = raw_input(full_prompt)
        if val == "":
            val = default
        if val in ['y','Y']:
            return True
        elif val in ['n','N']:
            return False

###############################################################################
# run the setup script

setup(
      #metadata
      name         = "mathstuff",
      version      = "0.1dev",
      author       = "Craig Versek",
      author_email = "cversek@physics.umass.edu",

      #packages to install
      package_dir  = {'':'src'},
      packages     = find_packages('src'),
      
      #non-code files
      package_data     =   {'': ['*.yaml','*.yml']},

      #dependencies
      install_requires = INSTALL_REQUIRES,
      extras_require = {
                        'Plotting': ['matplotlib >= 0.98'],
                       },
      dependency_links = [
                          #'http://sourceforge.net/project/showfiles.php?group_id=80706', #matplotlib
                         ],
      #scripts and plugins
      entry_points = {
                      #'console_scripts': ['automat_decode_nispy = automat.scripts.decode_nispy:main']
                     }      
)
