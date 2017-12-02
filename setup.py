from setuptools import setup, find_packages
import os
import pip
import pkg_resources
import re
pip.main(['install', 'git+https://github.com/davidfischer/requirements-parser.git'])
import requirements

# get long description
if os.path.isfile('README.rst'):
    with open('README.rst', 'r') as file:
        long_description = file.read()
else:
    long_description = ''

# get version
with open('conv_opt/VERSION', 'r') as file:
    version = file.read().strip()

# parse dependencies and their links from requirements.txt files
def parse_requirements(filename, install_requires, extras_require, dependency_links):
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            for req in requirements.parse(file):
                line = req.line
                if '#egg=' in line:
                    if line.find('#') < line.find('#egg='):
                        line = line[0:line.find('#')]
                    else:
                        line = line[0:line.find('#', line.find('#egg=')+5)]
                else:
                    if '#' in line:
                        line = line[0:line.find('#')]
                if ';' in line:
                    marker = line[line.find(';')+1:].strip()
                    marker_match = pkg_resources.Requirement.parse(req.name + '; ' + marker).marker.evaluate()

                else:
                    marker = ''
                    marker_match = True

                req_setup = req.name + ','.join([''.join(spec) for spec in req.specs]) + ('; ' if marker else '') + marker

                if req.extras:
                    for option in req.extras:
                        if option not in extras_require:
                            extras_require[option] = set()
                        extras_require[option].add(req_setup)
                else:
                    install_requires.add(req_setup)

                if req.uri:
                    if req.revision:
                        dependency_links[marker_match].add(req.uri + '@' + req.revision)
                    else:
                        dependency_links[marker_match].add(req.uri)

install_requires = set()
tests_require = set()
docs_require = set()
extras_require = {}
dependency_links = {True: set(), False: set()}

parse_requirements('requirements.txt', install_requires, extras_require, dependency_links)
parse_requirements('tests/requirements.txt', tests_require, extras_require, dependency_links)
parse_requirements('docs/requirements.txt', docs_require, extras_require, dependency_links)

tests_require = tests_require.difference(install_requires)
docs_require = docs_require.difference(install_requires)

extras_require['tests'] = tests_require
extras_require['docs'] = docs_require

install_requires = list(install_requires)
tests_require = list(tests_require)
docs_require = list(docs_require)
for option, reqs in extras_require.items():
    extras_require[option] = list(reqs)
for marker_match, reqs in dependency_links.items():
    dependency_links[marker_match] = list(reqs)

# install non-PyPI dependencies because setup doesn't do this correctly
for dependency_link in dependency_links[True]:
    pip.main(['install', dependency_link])

# install package
setup(
    name="conv_opt",
    version=version,
    description="conv_opt",
    long_description=long_description,
    url="https://github.com/KarrLab/conv_opt",
    download_url='https://github.com/KarrLab/conv_opt',
    author="Karr Lab",
    author_email="karr@mssm.com",
    license="MIT",
    keywords='convex optimization, linear programming, quadratic programming',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        'conv_opt': [
            'VERSION',
        ],
    },
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require,
    dependency_links=dependency_links[True] + dependency_links[False],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
