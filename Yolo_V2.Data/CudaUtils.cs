using System;
using System.Linq;
using Alea;
using Alea.cuBLAS;
using Alea.CudaToolkit;

namespace Yolo_V2.Data
{
    public static class CudaUtils
    {
        public static bool UseGpu = false;
        private static Gpu gpu;
        public static int BlockSize = 512;

        private static bool cublasInit;
        private static Alea.cuBLAS.Blas blas;
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

        public static LaunchParam cuda_gridsize(int n)
        {
            if (n > 65535)
                n = 65535;
            return new LaunchParam(32, 512);
        }


        public static IntPtr blas_handle()
        {
            if (!cublasInit)
            {
                blas = Alea.cuBLAS.Blas.Get(Gpu.Default);
                cublasInit = true;
            }
            return blas.Handle;
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

        public static void SafeCall(Alea.cuBLAS.Status status)
        {
            if (status != Status.SUCCESS)
            {
                throw new InvalidOperationException(status.ToString());
            }
        }
        
        public static void SafeCall(curandStatus status)
        {
            if (status != curandStatus.CURAND_STATUS_SUCCESS)
            {
                throw new InvalidOperationException(status.ToString());
            }
        }

    }
}
