# HIP-Examples

## Deprecation Notice
Please note that AMD will deprecate and archive the `hip-examples` repository. Please visit [rocm-examples](https://github.com/ROCm/rocm-examples), the new home for ROCm examples.

## Examples for HIP.
This depot should be extracted into the root directory of an existing HIP depot.

We managed to push the following benchmarks with HIP upstreamed on github:

* mixbench: <https://github.com/ekondis/mixbench>
* GPU-Stream: <https://github.com/UoB-HPC/GPU-STREAM>

mixbench and GPU-Stream have been added as submodules for this repository, to fetch data for submodules:

```bash
    git submodule init
    git submodule update
```
