
typedef float       NumericT;

int kernel_func(NumericT *x, NumericT const *y, NumericT const *z, int stride, int N)
{
  if (stride == 1)
  {
    #pragma omp parallel for
    for (int i=0; i<N; ++i)
      x[i] = y[i] + z[i];
  }
  else
  {
    #pragma omp parallel for
    for (int i=0; i<N; ++i)
      x[i*stride] = y[i*stride] + z[i*stride];
  }
}
