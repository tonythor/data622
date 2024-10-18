from lussi.stackoverflow import *
from lussi.ziprecruiter import *

"""
Run like this: 
(.venv) hurricane:krijudato afraser$ python ./src/lussi/run.py
"""

# run this to build out your caches.
nogit_data_dir = "622data_nogit"
build_stack(data_dir=nogit_data_dir)
build_zip(data_dir=nogit_data_dir)

# this is how you load.
raw_stack  = load_stack(data_dir = nogit_data_dir, stack_type=StackType.RAW)
wide_stack = load_stack(data_dir = nogit_data_dir, stack_type=StackType.WIDE)
ziprecruiter = load_zip(data_dir = nogit_data_dir)

print(raw_stack.head())
print(wide_stack.head())
print(ziprecruiter.head())