from setuptools import setup, find_packages
import conv_opt
import pip
import re

# parse dependencies and their links from requirements.txt files
install_requires = []
tests_require = []
dependency_links = []

for line in open('requirements.txt'):
    pkg_src = line.rstrip()
    match = re.match('^.+#egg=(.*?)$', pkg_src)
    if match:
        pkg_id = match.group(1)
        dependency_links.append(pkg_src)
    else:
        pkg_id = pkg_src
    install_requires.append(pkg_id)

for line in open('tests/requirements.txt'):
    pkg_src = line.rstrip()
    match = re.match('^.+#egg=(.*?)$', pkg_src)
    if match:
        pkg_id = match.group(1)
        dependency_links.append(pkg_src)
    else:
        pkg_id = pkg_src
    tests_require.append(pkg_id)
dependency_links = list(set(dependency_links))

# install non-PyPI dependencies because setup doesn't do this correctly
for dependency_link in dependency_links:
    pip.main(['install', dependency_link])

# install package
setup(
    name="conv_opt",
    version=conv_opt.__version__,
    description="conv_opt",
    url="https://github.com/KarrLab/conv_opt",
    download_url='https://github.com/KarrLab/conv_opt',
    author="Karr Lab",
    author_email="karr@mssm.com",
    license="MIT",
    keywords='',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        'conv_opt': [],
    },
    install_requires=install_requires,
    extras_require={
        'solver': ['cplex', 'cylp', 'gurobipy', 'mosek', 'xpress'],
    },
    tests_require=tests_require,
    dependency_links=dependency_links,
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
