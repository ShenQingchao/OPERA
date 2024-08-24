# Instrumentation for PyTorch operator instance extraction 

This folder contains code to perform instrumentation on PyTorch in order to collect operator call information.

We hook the invocation of `224` PyTorch operators, and the operator API names are listed in `operator_list.txt`.

The key function is `def hijack(output_file)` in `hijack.py`, where `output_file` represents the path that all the traced information will be saved to.

## Usage:

(1) Copy the folder `migrate` to the root directory where PyTorch is installed, 
such as `site-packages/torch/migrate/`.

(2) Append these lines to the `site-packages/torch/__init__.py`.

```
from torch.migrate.hijack import hijack
hijack("torch_migrated_tc.py")
```

Then we execute the all collected test cases to trace the dynamic execution information for each operator call. 
The outputs will be stored in the directory.
