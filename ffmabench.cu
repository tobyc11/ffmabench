#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <Windows.h>

#define CHECK_STATUS() do { if (status != cudaSuccess) \
	{\
	printf("CUDA call failed with %d at line %d\n", status, __LINE__);\
	return status;\
	} } while(0)

__device__ float4 operator*(float4 lhs, float4 rhs)
{
	float4 result = { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w };
	return result;
}

__device__ float4 operator+(float4 lhs, float4 rhs)
{
	float4 result = { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w };
	return result;
}

__device__ float4 operator-(float4 lhs, float4 rhs)
{
	float4 result = { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w };
	return result;
}

#define LOOP_LEN 1000

// Each FFMA is 2 float operations
// Times 4 lanes times LOOP_LEN iterations times 2
__global__ void bench_kernel(cudaTextureObject_t tex, float4 *out)
{
	float4 c0 = { 1.0f, 2.0f, 3.0f, 4.0f };
	float4 c1 = { 1.0f, 2.1f, 3.2f, 4.3f };
	float4 v0 = tex2D<float4>(tex, threadIdx.x, threadIdx.y);
	float4 v1 = v0;
	float4 v2;
	for (int i = 0; i < LOOP_LEN; i++)
	{
		v1 = v1 * v0 + c1;
	}
	v2 = v1;
	for (int i = 0; i < LOOP_LEN; i++)
	{
		v2 = v2 * v0 - c0;
	}
	*out = v2;
}

void* generate_arr_data()
{
	float *buffer = (float *)malloc(4 * 4 * 1024);
	for (int i = 0; i < 1024; i++)
	{
		buffer[i * 4 + 0] = 1.0f;
		buffer[i * 4 + 1] = 2.0f;
		buffer[i * 4 + 2] = 3.01f;
		buffer[i * 4 + 3] = 4.01f;
	}
	return buffer;
}

static inline int64_t get_ticks()
{
	LARGE_INTEGER ticks;
	if (!QueryPerformanceCounter(&ticks))
	{
		return 0;
	}
	return ticks.QuadPart;
}

static inline int64_t get_ticks_per_second()
{
	LARGE_INTEGER ticks;
	if (!QueryPerformanceFrequency(&ticks))
	{
		return 0;
	}
	return ticks.QuadPart;
}

int main()
{
	cudaError_t status;
	cudaArray_t arr;
	void* arr_data;
	cudaChannelFormatDesc format_desc;
	cudaTextureObject_t tex;
	cudaResourceDesc res_desc;
	cudaTextureDesc tex_desc;
	cudaResourceViewDesc view_desc;
	void* out;
	double ticks_per_second = (double)get_ticks_per_second();
	double begin, end;
	const unsigned int blockSz = 1024;
	const unsigned int gridSz = 1048576;
	const unsigned int flopUnit = 16 * LOOP_LEN;

	format_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
	status = cudaMallocArray(&arr, &format_desc, 1024, 1);
	CHECK_STATUS();
	arr_data = generate_arr_data();
	status = cudaMemcpyToArray(arr, 0, 0, arr_data, 4 * 4 * 1024, cudaMemcpyDefault);
	CHECK_STATUS();
	free(arr_data);
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = arr;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeBorder;
	tex_desc.addressMode[1] = cudaAddressModeBorder;
	tex_desc.addressMode[2] = cudaAddressModeBorder;
	tex_desc.readMode = cudaReadModeElementType;
	view_desc.format = cudaResViewFormatUnsignedInt4;
	view_desc.width = 1024;
	view_desc.height = 1;
	view_desc.depth = 0;
	view_desc.firstMipmapLevel = 0;
	view_desc.lastMipmapLevel = 0;
	view_desc.firstLayer = 0;
	view_desc.lastLayer = 0;
	status = cudaCreateTextureObject(&tex, &res_desc, &tex_desc, &view_desc);
	CHECK_STATUS();
	cudaMalloc(&out, 0x1000);
	printf("Running %u ops/thread * %u threads/block * %u blocks\n", flopUnit, blockSz, gridSz);
	begin = (double)get_ticks();
	bench_kernel<<<gridSz, blockSz>>>(tex, (float4 *)out);
	end = (double)get_ticks();
	printf("Launch latency: %f s\n", (end - begin) / ticks_per_second);
	status = cudaDeviceSynchronize();
	CHECK_STATUS();
	end = (double)get_ticks();
	double duration = (end - begin) / ticks_per_second;
	printf("Kernel duration: %f s\n", duration);
	double tflops = (double)blockSz * gridSz * flopUnit / 1e12 / duration;
	printf("Your GPU's TFLOPS is %f\n", tflops);
	cudaFree(out);
	status = cudaDestroyTextureObject(tex);
	CHECK_STATUS();
	status = cudaFreeArray(arr);
	CHECK_STATUS();
	return 0;
}
