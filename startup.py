print('Running .startup')
import os
import sys

home = os.path.expanduser("~")
local_repo = os.path.join("Documents", "Code", "qfl")
os.chdir(os.path.join(home, local_repo))  # Activate .env

home = os.path.expanduser("~")
local_repo = os.path.join("Documents", "Code", "qfl")
sys.path.append(os.path.join(home, local_repo))  # Activate .env

modules = ["qfl"]
sub_modules =['qfl', 'data', 'airflow']
for sm in sub_modules:
    modules.append(os.path.join(modules[0], sm))
modules.append(os.path.join("qfl", "etl"))
modules.append(os.path.join("qfl", "core"))
modules.append(os.path.join("qfl", "macro"))
modules.append(os.path.join("qfl", "deploy"))
modules.append(os.path.join("airflow", "dags"))

modules.append(os.path.join('data_extraction', 'database'))
sys.path.extend([os.path.join(home, local_repo, p) for p in modules])

from qfl.etl.data_ingest import *

print('startup complete!')
