using System;
using System.Linq;
using Alea;
using Alea.CudaDnn;
using Alea.CudaToolkit;
using dim3 = Alea.dim3;

namespace Yolo_V2.Data
{
    public static class CudaUtils
    {
        public static bool UseGpu = false;
        private static Gpu gpu;
        public static int BlockSize = 512;
        private static bool cudnnInit;
        private static unsafe cudnnContext* cudnnHandle;
        private static bool cublasInit;
        private static unsafe cublasContext* cublasHandle;
        private static bool curandInit;
        private static unsafe curandGenerator_st* gen;

        public static bool HaveGpu()
        {
            try
            {
                if (Device.Devices.Count() > 0)
                {
                    gpu = Gpu.Default;
                    return true;
                }
            }
            catch 
            {
                return false;
            }

            return false;
        }

        public static dim3 cuda_gridsize(int n)
        {
            int k = (n - 1) / BlockSize + 1;
            int x = k;
            int y = 1;
            if (x > 65535)
            {
                x = (int)Math.Ceiling(Math.Sqrt(k));
                y = (n - 1) / (x * BlockSize) + 1;
            }
            return new dim3(x, y, 1);
        }

        public static unsafe cudnnContext* cudnn_handle()
        {
            if (!cudnnInit)
            {
                cudnnContext* handle;
                SafeCall(CuDnn.cudnnCreate(&handle));
                cudnnInit = true;
                cudnnHandle = handle;
            }
            return cudnnHandle;
        }

        public static unsafe cublasContext* blas_handle()
        {
            if (!cublasInit)
            {
                cublasContext* handle;
                SafeCall(CuBlas.cublasCreate_v2(&handle));
                cublasInit = true;
                cublasHandle = handle;
            }
            return cublasHandle;
        }
        
        public static unsafe void cuda_random(float[] x, ulong n)
        {
            if (!curandInit)
            {
                curandGenerator_st* handle;
                SafeCall(CuRand.curandCreateGenerator(&handle, curandRngType.CURAND_RNG_PSEUDO_DEFAULT));
                SafeCall(CuRand.curandSetPseudoRandomGeneratorSeed(handle, (ulong)DateTime.Now.Millisecond));
                curandInit = true;
                gen = handle;
            }

            using (var gpuX = Gpu.Default.AllocateDevice(x.ToArray()))
            {
                SafeCall(CuRand.curandGenerateUniform(gen, (float*)gpuX.Handle, n));
                CopyValues(x, gpuX);
            }
        }

        private static void CopyValues(float[] x, DeviceMemory<float> gpuX)
        {
            x = Gpu.CopyToHost(gpuX);
        }
        
        public static void SafeCall(cublasStatus_t status)
        {
            if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new InvalidOperationException(status.ToString());
            }
        }

        private static void SafeCall(curandStatus status)
        {
            if (status != curandStatus.CURAND_STATUS_SUCCESS)
            {
                throw new InvalidOperationException(status.ToString());
            }
        }

        private static void SafeCall(cudnnStatus_t status)
        {
            if (status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            {
                throw new InvalidOperationException(status.ToString());
            }
        }
    }
}
