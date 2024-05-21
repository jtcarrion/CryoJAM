from setuptools import setup

setup(
   name='cryojam',
   version='1.0',
   description='Automated Homolog Fitting into cryo-EM Density Maps',
   author='Jackson Carrion, Mrunali Manjrekar, Anna Mikulevica',
   author_email='jcarrion@mit.edu',
   packages=['cryojam'],  #same as name
   # install_requires=['torch', 'numpy', 'h5py', 'mrcfile', 'Bio', 'scipy', 'matplotlib'], #external packages as dependencies
)