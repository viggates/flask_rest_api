# ***** BEGIN LICENSE BLOCK *****
#
# For copyright and licensing please refer to COPYING.
#
# ***** END LICENSE BLOCK *****

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Conditionally include additional modules for docs

long_description = ('simple centrify api')

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__),os.pardir)))


EXCLUDE_FROM_PACKAGES = ['examples.*',
                         '*.logs.*',
                         '*.extra.*',
                         '*.tests.*',
                        ]

NAME = 'centrify-api'
entrypoints = {}


def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements('requirements.txt')
console_scripts = entrypoints['console_scripts'] = [
#    'run.py = jupiter_api.run:main',
]


setup(name=NAME,
      version='1.0',
      description='centrify-api',
      long_description=long_description,
      maintainer='centrify team',
      maintainer_email='centrify@firstdata.com',
      url='https://centrify.readthedocs.org ',
      packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
      #packages=['dolphind'],
      #scripts = [ 'extra/runinenv' ],
      entry_points=entrypoints,
      license='FDC ',
      include_package_data=True,
#      package_data={'centrify_api.centrify_api.config': ['LICENSE', 'README.md', '*.ini', 'requirements.txt']},
#      entry_points=entrypoints,
#      extras_require={'SQLAlchemy': ['SQLAlchemy'],
#                      'psycopg2': ['psycopg2'],
#                      'requests':['requests'],
#                      'MySQL-python': ['MySQL-python'],
#                      'cassandra-driver': ['cassandra-driver']},
      data_files = [('/etc/centrify/', ['wsgi_centrify_api.ini'])],
      install_requires=reqs,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Topic :: Communications',
          'Topic :: System :: Networking'],
      zip_safe=False)

