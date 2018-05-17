import sys
sys.setrecursionlimit(5000)
from distutils.core import setup
import py2exe
#import sys, os
#from cx_Freeze import setup, Executable

#__version__ = "1.1.0"

setup(
    name='Diabetes',
    version='0.0.1',
    description="Diabetes Model And Prediction Tool",
    author='Radhika Tayal',
    author_email='radhikatayal@gmail.com',
    #executables = [Executable("Diabetes.py",base="Win32GUI")])
    console=['Diabetes.py'])
