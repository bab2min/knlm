from setuptools import setup, Extension
from codecs import open
import os, os.path
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README'), encoding='utf-8') as f:
    long_description = f.read()

sources = []
for f in os.listdir(os.path.join(here, 'src')):
    if f.endswith('.cpp'): sources.append('src/' + f)

if os.name == 'nt': cargs = ['/O2', '/MT', '/Gy']
else: cargs = ['-std=c++1y', '-O3', '-fpermissive']
modules = [Extension('knlm_c',
                    libraries = [],
                    sources = sources,
                    extra_compile_args=cargs)]

setup(
    name='knlm',

    version='0.1.0',

    description='Modified Kneser-ney Smoothing Language Model',
    long_description=long_description,

    url='https://github.com/bab2min/knlm',

    author='bab2min',
    author_email='bab2min@gmail.com',

    license='LGPL v3 License',

    classifiers=[
        'Development Status :: 3 - Alpha',

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",

        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",

        'Programming Language :: Python :: 3',
        'Programming Language :: C++'
    ],

    keywords=['nlp', 'language model', 'kneser-ney smoothing'],

    packages = ['knlm'],
    include_package_data=True,
    ext_modules = modules
)
