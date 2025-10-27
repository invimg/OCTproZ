#include "gpuinfo.h"

GpuInfo::GpuInfo(QObject* parent) : QObject(parent)
{
}

bool GpuInfo::isCudaAvailable() {
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	if(err != cudaSuccess){
		emit error(QString("CUDA error: %1").arg(cudaGetErrorString(err)));
		return false;
	}

	return deviceCount > 0;
}

QVector<GpuDeviceInfo> GpuInfo::getAllDevices() {
	QVector<GpuDeviceInfo> devices;
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	if(err != cudaSuccess){
		emit error(QString("CUDA error: %1").arg(cudaGetErrorString(err)));
		return devices;
	}

	for(int i = 0; i < deviceCount; ++i){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		size_t freeMem = 0;
		size_t totalMem = 0;
		cudaSetDevice(i);
		cudaMemGetInfo(&freeMem, &totalMem);

		GpuDeviceInfo info;
		info.deviceId = i;
		info.name = QString(prop.name);
		info.major = prop.major;
		info.minor = prop.minor;
		info.totalGlobalMem = prop.totalGlobalMem;
		info.freeGlobalMem = freeMem;
		info.totalConstMem = prop.totalConstMem;
		info.sharedMemPerBlock = prop.sharedMemPerBlock;
		info.sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
		info.regsPerBlock = prop.regsPerBlock;
		info.regsPerMultiprocessor = prop.regsPerMultiprocessor;
		info.l2CacheSize = prop.l2CacheSize;
		info.multiProcessorCount = prop.multiProcessorCount;
		info.memoryBusWidth = prop.memoryBusWidth;
		info.warpSize = prop.warpSize;
		info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
		info.maxThreadsPerMultiprocessor = prop.maxThreadsPerMultiProcessor;
		for(int d = 0; d < 3; ++d){
			info.maxThreadsDim[d] = prop.maxThreadsDim[d];
			info.maxGridSize[d] = prop.maxGridSize[d];
		}
		info.integrated = prop.integrated;
		info.managedMemory = prop.managedMemory;
		info.concurrentKernels = prop.concurrentKernels;
		info.canMapHostMemory = prop.canMapHostMemory;
		info.asyncEngineCount = prop.asyncEngineCount;

//some fields are deprecated/removed in cuda 13 --> fetch via attributes when cuda 13+ is used.
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
		info.clockRate = -1;
		info.memoryClockRate = -1;
		int value = 0;
		if(getAttr(i, cudaDevAttrClockRate, value)) {
			info.clockRate = value;
		}
		if(getAttr(i, cudaDevAttrMemoryClockRate, value)) {
			info.memoryClockRate = value;
		}
#else
		info.clockRate = prop.clockRate;
		info.memoryClockRate = prop.memoryClockRate;
#endif

		devices.append(info);
	}

	return devices;
}

bool GpuInfo::getAttr(int device, cudaDeviceAttr attr, int &out) {
	cudaError_t err = cudaDeviceGetAttribute(&out, attr, device);
	if (err != cudaSuccess) {
		emit error(QString("CUDA attribute query failed (%1) on device %2: %3")
			.arg(static_cast<int>(attr))
			.arg(device)
			.arg(cudaGetErrorString(err)));
		return false;
	}
	return true;
}
