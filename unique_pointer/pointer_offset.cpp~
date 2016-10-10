/*
 *  README:  perform reducton sum in hip kernel using device function
 *
 */


#include <stdio.h>
#include <hip_runtime.h>
#include <utility>

using namespace std;

#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}


struct host_mem_deleter
{
    template <class T> void operator()(T* MemObj)
    {
        if( MemObj != NULL )
            delete MemObj;
    };
};

struct host_mem_array_deleter
{
    template <class T> void operator()(T* MemObj)
    {
        if( MemObj != NULL )
            delete[] MemObj;
    };
};



struct hipStream_deleter
{
    void operator()(hipStream_t* stream)
    {
        if( stream != NULL )
            hipStreamDestroy(*stream);
    };
};


struct device_mem_deleter
{
    template <class T> void operator()(T* MemObj)
    {
        if( MemObj != NULL )
            hipFree(MemObj);
    };
};


int main(int argc, char *argv[])
{

	hipDeviceProp_t props;
	CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
	printf ("info: running on device %s\n", props.name);

    /*===========================================
             C++ host pointer experiment 
    =============================================*/

    // using new in unique_ptr 
    // create a object owing a pointer of type (float)
    unique_ptr< float, host_mem_deleter > float_object (new float);
    *float_object = 4;

    //using new in unique_ptr, which stores a array 
    unique_ptr< float[], host_mem_array_deleter > float_array_object (new float[3] );
    float_array_object[0] = float_array_object[1] = float_array_object[2] = 5;


    float *pd = new float;
    unique_ptr<float> upptr(pd);
    *upptr = 6;

  
    unique_ptr<float> inptr( (float*)malloc(sizeof(float)) ) ;
    *inptr = 5;


    /* Does not compile 

    float* p = (float*)malloc(sizeof(float));
    unique_ptr<p, free> inptr;
    *inptr = 5;

    */
    

 

    /*===========================================
             hipStream  experiment : success 
    =============================================*/

    // using hipStream in unique_ptr
    unique_ptr< hipStream_t, hipStream_deleter > stream_object (new hipStream_t);
    hipStreamCreate(stream_object.get());

    //preferred 
    hipStream_t stream;
    hipStreamCreate(&stream);
    unique_ptr< hipStream_t, hipStream_deleter > stream_object_2 (&stream);


    /*===========================================
             hipMalloc  experiment: mixed  
    =============================================*/

    size_t Nbyte = 1024 * sizeof(float);

    unique_ptr< float, device_mem_deleter > hip_object(new float);
    //Does not compile :           hipMalloc(&(hip_object.get()), Nbyte);
    //Does not compile :           hipMalloc(hip_object, Nbyte);


    //preferred
    float *device_pointer; 
    hipMalloc(&device_pointer, Nbyte);
    unique_ptr<float, device_mem_deleter>  device_object (device_pointer);

    //Does not compile :           unique_ptr< device_pointer, device_mem_deleter > hip_object_2;

    //printf("A_d=%f\n", A_d);

    
    return 0;
}



