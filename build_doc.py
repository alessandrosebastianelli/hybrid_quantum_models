"""
Build the documentation using pdoc (not pdoc3)
"""
import shutil
import glob
import os

# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

os.mkdir("./docs/")
os.mkdir("./docs/hqm/")

# Build documentation
os.system("pdoc ./hqm -o ./docs --docformat numpy --logo logo.png -t ./docs_assets/") 
shutil.copy2("./docs_assets/logo.png", "./docs/logo.png")
shutil.copy2("./docs_assets/logo.png", "./docs/hqm/logo.png")
shutil.copy2("./docs_assets/main-hqm.png", "./docs/main-hqm.png")

for p in glob.glob("./docs/hqm/*/"):
    shutil.copy2("./docs_assets/logo.png", p+"/logo.png")

for fld in glob.glob("./docs_assets/*/"):
    os.system(f"cp -r {fld} {fld.replace('docs_assets', 'docs')}")