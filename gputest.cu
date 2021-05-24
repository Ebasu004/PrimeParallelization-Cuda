#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>

const long N = 1000000; // Change array size (may need a long) 

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER CODE TO INITIALIZE, PRINT AND TIME
struct timeval start, end;


void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

void init(const char* c) {
  printf("***************** %s **********************\n", c);
  // TMC Commenting Out for Class  
  printf("Running %s...\n", c);
  starttime();
}

void finish(int a, long N, const char* c) {
	endtime(c);
	printf("Done.\n");
	printf("\nThere are %ld Prime numbers between 1 and %ld.", a, N);
	printf("***************************************************\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 __global__ void prime(long* a, long high)  {
	// Prime algorithm
	bool check = false;
	for(int i = 2; i <= high/2; ++i) {
		if(high % i == 0) {
			check = true;
			break;
		}
	}
	if(check)
		++a;
}
*/

// Normal C function to square root values
int normal(int a, long N)
{
    long low = 2, high = N, i, check;
	// printf("Prime numbers between 1 and %d are: ",high);
	while (low < high)
	{
		check = 0;
		for(i = 2; i <= low/2; ++i)
		{
		if(low % i == 0)
		{
			check = 1;
			break;
		}
		}
		if (check == 0)
			++a;
		//printf("%d ", low);
		++low;
   }
   return a;
}                                                                                                                                                                                                       

// GPU function to square root values
// Every thread on every core runs this function
__global__ void gpu_prime(int* a, long N) {
   // One element per thread on each core
   // blockIdx.x = Core #
   // blockDim.x = Threads per core
   // threadIdx.x = Thread #
   // The formula below makes sure the value of element 
   // is different on every thread on every core
   long element = blockIdx.x*blockDim.x + threadIdx.x;
   // If there is not an event split, some threads will be 
   // out of bounds
   // We just let those do nothing
   // The rest square root their elements 
	if (element <= N && element >= 2) {
		/*
		if (element % 2 != 0)
			element = N - element;
		//printf("%d\n", element);
		*/
		//printf("%d\n", element);
		int check = 0;	
		for(int i = 2; i <= element/2; ++i) {
        		if(element  % i == 0) {
        		check = 1;
        		break;
        		}
        	}
		if (check == 0){
	                atomicAdd(a,1);	
		}
	}
}

void gpu(int* a, long N) {
   int threadsPerCore = 512; // This can vary, up to 1024
   long numCores = N / threadsPerCore + 1; // This division will work.  If the split is uneven, we overshoot

   // Budget memory for counter
   // Memory must be on the graphics card (use cudaMalloc for this)
   int* gpuA;
   cudaMalloc(&gpuA, sizeof(int)); // Allocate enough memory on the GPU
   
   // Copy array of floats a from CPU memory to gpuA on the graphics card
   // Note: This operation is SLOW.  You will have to offset this cost with the parallelism below
   cudaMemcpy(gpuA, a, sizeof(int), cudaMemcpyHostToDevice); 
   //printf("%ld\n", *gpuA);	
   // Call parallel function with specified number of cores and threads per core
   gpu_prime<<<numCores, threadsPerCore>>>(gpuA, N);

   // Copy square rooted array of floats gpuA from graphics card to a in CPU memory
   // Again, this operation is SLOW.  
   cudaMemcpy(a, gpuA, sizeof(int), cudaMemcpyDeviceToHost); 
   
   // Release the memory for gpuA
   cudaFree(&gpuA); // Free the memory on the GPU
}
                                                                                                                                                                                               
 

int main()                                                                                                                                                                                  
{
	/////////////////////////////////////////////////////////////////////////
	// GPUs will likely have large N
	// Budget memory on the heap, prevent a stack overflow  
	int a = 1;
	/////////////////////////////////////////////////////////////////////////
	
	// Test 1: Sequential For Loop
	init ("Normal");
	a = normal(a, N); 
	finish(a, N, "Normal"); 
	// Test 2: GPU
	a = 1;
	init("GPU");
	gpu(&a, N);  
	finish(a, N, "GPU");

	// Memory on the heap must be freed manually
	//free(&a);
	return 0;
}

