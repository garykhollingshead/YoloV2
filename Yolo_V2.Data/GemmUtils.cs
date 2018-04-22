using System;
using System.Runtime.InteropServices;
using Alea;
using Alea.cuBLAS;
using Alea.CudaToolkit;

namespace Yolo_V2.Data
{
    public static class GemmUtils
    {
        public static void Gemm(int ta, int tb, int m, int n, int k, float alpha,
                float[] a, int lda,
                float[] b, int ldb,
                float beta,
                float[] c, int ldc)
        {
            gemm_cpu(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        private static void gemm_nn(int m, int n, int k, float alpha,
                float[] a, int lda,
                float[] b, int ldb,
                float[] c, int ldc)
        {
            Console.WriteLine($"starting...\n{c[0]}");
            for (var i = 0; i < m; ++i)
            {
                var iapart = i * lda;
                var icpart = i * ldc;
                for (var ik = 0; ik < k; ++ik)
                {
                    float aPart = alpha * a[iapart + ik];
                    var kbpart = ik * ldb;

                    for (int ij = 0; ij < n; ++ij)
                    {
                        if (icpart + ij == 0)
                        {
                            Console.WriteLine($"+ {aPart} * {b[kbpart + ij]}");
                        }
                        c[icpart + ij] += aPart * b[kbpart + ij];
                    }
                }
            }
            Console.WriteLine($"{c[0]}\ndone");
        }

        private static void gemm_nt(int m, int n, int ok, float alpha,
                float[] a, int lda,
                float[] b, int ldb,
                float[] c, int ldc)
        {
            for (var i = 0; i < m; ++i)
            {
                for (var j = 0; j < n; ++j)
                {
                    float sum = 0;
                    for (var k = 0; k < ok; ++k)
                    {
                        sum += alpha * a[i * lda + k] * b[j * ldb + k];
                    }
                    c[i * ldc + j] += sum;
                }
            }
        }

        private static void gemm_tn(int m, int n, int ok, float alpha,
                float[] a, int lda,
                float[] b, int ldb,
                float[] c, int ldc)
        {
            for (var i = 0; i < m; ++i)
            {
                for (var k = 0; k < ok; ++k)
                {
                    float aPart = alpha * a[k * lda + i];
                    for (var j = 0; j < n; ++j)
                    {
                        c[i * ldc + j] += aPart * b[k * ldb + j];
                    }
                }
            }
        }

        private static void gemm_tt(int m, int n, int ok, float alpha,
                float[] a, int lda,
                float[] b, int ldb,
                float[] c, int ldc)
        {
            for (var i = 0; i < m; ++i)
            {
                for (var j = 0; j < n; ++j)
                {
                    float sum = 0;
                    for (var k = 0; k < ok; ++k)
                    {
                        sum += alpha * a[i + k * lda] * b[k + j * ldb];
                    }
                    c[i * ldc + j] += sum;
                }
            }
        }

        private static void gemm_cpu(int ta, int tb, int m, int n, int k, float alpha,
                float[] a, int lda,
                float[] b, int ldb,
                float beta,
                float[] c, int ldc)
        {
            int i, j;
            for (i = 0; i < m; ++i)
            {
                for (j = 0; j < n; ++j)
                {
                    c[i * ldc + j] *= beta;
                }
            }
            if (ta == 0 && tb == 0)
                gemm_nn(m, n, k, alpha, a, lda, b, ldb, c, ldc);
            else if (ta != 0 && tb == 0)
                gemm_tn(m, n, k, alpha, a, lda, b, ldb, c, ldc);
            else if (ta == 0 && tb != 0)
                gemm_nt(m, n, k, alpha, a, lda, b, ldb, c, ldc);
            else
                gemm_tt(m, n, k, alpha, a, lda, b, ldb, c, ldc);
        }

        public static void gemm_ongpu(int ta, int tb, int m, int n, int k, float alpha,
            float[] a, int lda,
            float[] b, int ldb,
            float beta,
            ref float[] c, int ldc)
        {
            unsafe
            {
                Gpu.FreeAllImplicitMemory(true);
                ulong free = 0;
                ulong total = 0;
                Cuda.cudaMemGetInfo(&free, &total);

                using (var gpuA = Gpu.Default.AllocateDevice(a))
                using (var gpuB = Gpu.Default.AllocateDevice(b))
                using (var gpuC = Gpu.Default.AllocateDevice(c))
                {
                    var handle = CudaUtils.blas_handle();
                    CudaUtils.SafeCall(Alea.cuBLAS.Interop.cublasSgemm(handle,
                        (tb != 0 ? Operation.T : Operation.N),
                        (ta != 0 ? Operation.T : Operation.N), n, m, k, &alpha,
                        (float*)gpuB.Handle, ldb,
                        (float*)gpuA.Handle, lda, &beta, (float*)gpuC.Handle, ldc));
                    //a = Gpu.CopyToHost(gpuA);
                    //b = Gpu.CopyToHost(gpuB);
                    c = Gpu.CopyToHost(gpuC);
                }
                Gpu.FreeAllImplicitMemory(true);
                ulong freeA = 0;
                ulong totalA = 0;
                Cuda.cudaMemGetInfo(&freeA, &totalA);
            }

        }
    }
}