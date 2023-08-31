"""
Build the distribution
"""
import shutil
import os


# Clean dist folder
shutil.rmtree('./build/', ignore_errors=True)
shutil.rmtree('./dist/', ignore_errors=True)
shutil.rmtree('./pyosv.egg-info/', ignore_errors=True)

# Build distribuition
os.system("python build_doc.py")
os.system("python setup.py sdist bdist_wheel")
os.system("python3 -m twine upload dist/* --verbose")