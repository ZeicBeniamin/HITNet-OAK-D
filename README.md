## Conversion of HITNet to ONNX

-----------------

This repository contains the work of team Pixel Match aimed towards improving depth estimation on OAK-D Pro by using a HITNet model.

This repository is, broadly speaking, a copy of https://github.com/zjjMaiMai/TinyHITNet, but it contains changes and scripts used for converting the pytorch model to ONNX and then to BLOB.

There is still a lot of work to be done in order to reach a final result with a decent runtime.

## Running the 100x180 model
------------------

To run the already compiled 100x180 blob, you must:

``` cd HITNet-OAK-D && python3 depth_ai/run.py ```