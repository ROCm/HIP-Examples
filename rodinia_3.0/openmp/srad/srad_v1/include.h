#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#include <mex.h>

#include <define.h>

#include <print.c>

#include <prepare_kernel.cu>
#include <extract_kernel.cu>
#include <reduce_kernel.cu>
#include <srad_kernel.cu>
#include <srad2_kernel.cu>
#include <compress_kernel.cu>