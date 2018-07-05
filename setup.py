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

long_description = ('simple  api')

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__),os.pardir)))


EXCLUDE_FROM_PACKAGES = ['examples.*',
                         '*.logs.*',
                         '*.extra.*',
                         '*.tests.*',
                        ]

NAME = 'simple-api'
entrypoints = {}


def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements('requirements.txt')
console_scripts = entrypoints['console_scripts'] = [
#    'run.py = simple_api.run:main',
]


setup(name=NAME,
      version='1.0',
      description='simple-api',
      long_description=long_description,
      maintainer='Vigneshvar A S',
      maintainer_email='vigneshvar.a.s@gmail.com',
      url='https://key-value.blogspot.com',
      packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
      #packages=['dolphind'],
      #scripts = [ 'extra/runinenv' ],
      entry_points=entrypoints,
      license='No one',
      include_package_data=True,
#      package_data={'simple_api.simple_api.config': ['LICENSE', 'README.md', '*.ini', 'requirements.txt']},
#      entry_points=entrypoints,
#      extras_require={'SQLAlchemy': ['SQLAlchemy'],
#                      'psycopg2': ['psycopg2'],
#                      'requests':['requests'],
#                      'MySQL-python': ['MySQL-python'],
#                      'cassandra-driver': ['cassandra-driver']},
      data_files = [('/etc/simple_api/', ['wsgi_simple_api.ini','supervisord_simple_api.conf'])],
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

