The benchmark is modified from STREAM benchmark implementation with the following kernels:
    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)

To compile HIP version:
    make
To execute:
    ./stream

To compile on NV node, use Makefile.titan.
