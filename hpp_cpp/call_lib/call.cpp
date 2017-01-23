/*
 *  README:  perform reducton sum in hip kernel using device function
 *
 */


#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include "function.h"


//using namespace std;


#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}




int main(int argc, char *argv[])
{
    return foo();
}



