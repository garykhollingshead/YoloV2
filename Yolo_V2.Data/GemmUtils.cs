﻿using System.Linq;
using Alea;
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
            for (var i = 0; i < m; ++i)
            {
                for (var ik = 0; ik < k; ++ik)
                {
                    float aPart = alpha * a[i * lda + ik];
                    for (var ij = 0; ij < n; ++ij)
                    {
                        c[i * ldc + ij] += aPart * b[ik * ldb + ij];
                    }
                }
            }
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

        public static unsafe void gemm_ongpu(int ta, int tb, int m, int n, int k, float alpha,
            float[] a, int lda,
            float[] b, int ldb,
            float beta,
            float[] c, int ldc)
        {
            using (var gpuA = Gpu.Default.AllocateDevice(a.ToArray()))
            using (var gpuB = Gpu.Default.AllocateDevice(b.ToArray()))
            using (var gpuC = Gpu.Default.AllocateDevice(c.ToArray()))
            {
                var handle = CudaUtils.blas_handle();
                CudaUtils.SafeCall(CuBlas.cublasSgemm_v2(handle,
                    (tb != 0 ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N),
                    (ta != 0 ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N), n, m, k, &alpha, (float*)gpuB.Handle, ldb,
                    (float*)gpuA.Handle, lda, &beta, (float*)gpuC.Handle, ldc));
                a = Gpu.CopyToHost(gpuA);
                b = Gpu.CopyToHost(gpuB);
                c = Gpu.CopyToHost(gpuC);
            }
            
        }
    }
}