# Instrumentation for ONNX operator instance extraction
This folder contains code to perform instrumentation on ONNX in order to collect operator call information.

## Usage:
(1) Insert the code in `instrument.txt` before the `return graph` of `make_graph` function in `onnx/helper.py`, 
which should be the file of your installed onnx, such as `site-packages/onnx/helper.py`.

(2) Change the `file_path` in `instrument.txt` to set the file path which the traced information will be saved to.
The default value of `file_path` is: `onnx_migrated_tc.py`
