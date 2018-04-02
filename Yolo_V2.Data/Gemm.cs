using System;
using System.Diagnostics;
using System.Linq;
using Alea;
using Alea.CudaToolkit;

namespace Yolo_V2.Data
{
    public static class Gemm
    {
        public static void gemm_bin(int M, int N, int K, float ALPHA,
                string A, int lda,
                float[] B, int ldb,
                float[] C, int ldc)
        {
            int i, j, k;
            for (i = 0; i < M; ++i)
            {
                for (k = 0; k < K; ++k)
                {
                    char A_PART = A[i * lda + k];
                    if (A_PART != '\0')
                    {
                        for (j = 0; j < N; ++j)
                        {
                            C[i * ldc + j] += B[k * ldb + j];
                        }
                    }
                    else
                    {
                        for (j = 0; j < N; ++j)
                        {
                            C[i * ldc + j] -= B[k * ldb + j];
                        }
                    }
                }
            }
        }

        public static float[] random_matrix(int rows, int cols)
        {
            int i;
            float[] m = new float[rows * cols];
            for (i = 0; i < rows * cols; ++i)
            {
                m[i] = (float)(Utils.Rand.NextDouble() / double.MaxValue);
            }
            return m;
        }

        public static void time_random_matrix(int TA, int TB, int m, int k, int n)
        {
            float[] a;
            if (TA == 0) a = random_matrix(m, k);
            else a = random_matrix(k, m);
            int lda = (TA == 0) ? k : m;
            float[] b;
            if (TB == 0) b = random_matrix(k, n);
            else b = random_matrix(n, k);
            int ldb = (TB == 0) ? n : k;

            float[] c = random_matrix(m, n);
            int i;
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (i = 0; i < 10; ++i)
            {
                gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
            }
            sw.Stop();
            Console.WriteLine($"Matrix Multiplication {m}x{k} * {k}x{n}, TA={TA}, TB={TB}: {sw.ElapsedMilliseconds:4} ms");
        }

        public static void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float BETA,
                float[] C, int ldc)
        {
            gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
        }

        public static void gemm_nn(int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float[] C, int ldc)
        {
            int i, j, k;
            for (i = 0; i < M; ++i)
            {
                for (k = 0; k < K; ++k)
                {
                    float A_PART = ALPHA * A[i * lda + k];
                    for (j = 0; j < N; ++j)
                    {
                        C[i * ldc + j] += A_PART * B[k * ldb + j];
                    }
                }
            }
        }

        public static void gemm_nt(int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float[] C, int ldc)
        {
            int i, j, k;
            for (i = 0; i < M; ++i)
            {
                for (j = 0; j < N; ++j)
                {
                    float sum = 0;
                    for (k = 0; k < K; ++k)
                    {
                        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
                    }
                    C[i * ldc + j] += sum;
                }
            }
        }

        public static void gemm_tn(int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float[] C, int ldc)
        {
            int i, j, k;
            for (i = 0; i < M; ++i)
            {
                for (k = 0; k < K; ++k)
                {
                    float A_PART = ALPHA * A[k * lda + i];
                    for (j = 0; j < N; ++j)
                    {
                        C[i * ldc + j] += A_PART * B[k * ldb + j];
                    }
                }
            }
        }

        public static void gemm_tt(int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float[] C, int ldc)
        {
            int i, j, k;
            for (i = 0; i < M; ++i)
            {
                for (j = 0; j < N; ++j)
                {
                    float sum = 0;
                    for (k = 0; k < K; ++k)
                    {
                        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
                    }
                    C[i * ldc + j] += sum;
                }
            }
        }

        public static void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float BETA,
                float[] C, int ldc)
        {
            int i, j;
            for (i = 0; i < M; ++i)
            {
                for (j = 0; j < N; ++j)
                {
                    C[i * ldc + j] *= BETA;
                }
            }
            if (TA == 0 && TB == 0)
                gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
            else if (TA != 0 && TB == 0)
                gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
            else if (TA == 0 && TB != 0)
                gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
            else
                gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
        }

        public static unsafe void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA,
            float[] A, int lda,
            float[] B, int ldb,
            float BETA,
            float[] C, int ldc)
        {
            using (var gpuA = Gpu.Default.AllocateDevice(A.ToArray()))
            using (var gpuB = Gpu.Default.AllocateDevice(B.ToArray()))
            using (var gpuC = Gpu.Default.AllocateDevice(C.ToArray()))
            {
                var handle = CudaUtils.blas_handle();
                CudaUtils.SafeCall(CuBlas.cublasSgemm_v2(handle,
                    (TB != 0 ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N),
                    (TA != 0 ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N), N, M, K, &ALPHA, (float*)gpuB.Handle, ldb,
                    (float*)gpuA.Handle, lda, &BETA, (float*)gpuC.Handle, ldc));
                A = Gpu.CopyToHost(gpuA);
                B = Gpu.CopyToHost(gpuB);
                C = Gpu.CopyToHost(gpuC);
            }
            
        }

        public static void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
                float[] A, int lda,
                float[] B, int ldb,
                float BETA,
                float[] C, int ldc)
        {
            gemm_ongpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
        }


        public static void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
        {
            var a = TA == 0
                ? random_matrix(m, k)
                : random_matrix(k, m);
            var lda = (TA == 0) ? k : m;
            var b = TB == 0
                ? random_matrix(k, n)
                : random_matrix(n, k);
            var ldb = (TB == 0) ? n : k;

            var c = random_matrix(m, n);
            int i;
            var sw = new Stopwatch();
            sw.Start();
            for (i = 0; i < 32; ++i)
            {
                gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
            }
            sw.Stop();
            Console.Error.WriteLine($"Matrix Multiplication {m}x{k} * {k}x{n}, TA={TA}, TB={TB}: {sw.ElapsedMilliseconds:1} ms");
        }

        public static void time_ongpu(int TA, int TB, int m, int k, int n)
        {
            int iter = 10;
            float[] a = random_matrix(m, k);
            float[] b = random_matrix(k, n);

            int lda = (TA == 0) ? k : m;
            int ldb = (TB == 0) ? n : k;

            float[] c = random_matrix(m, n);

            int i;
            var sw = new Stopwatch();
            sw.Start();
            for (i = 0; i < iter; ++i)
            {
                gemm_ongpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
            }
            double flop = ((double)m) * n * (2.0 * k + 2.0) * iter;
            double gflop = flop / Math.Pow(10.0, 9);
            sw.Stop();
            Console.Error.WriteLine(
                $"Matrix Multiplication {m}x{k} * {k}x{n}, TA={TA}, TB={TB}: {sw.ElapsedMilliseconds} ms, {gflop / TimeSpan.FromMilliseconds(sw.ElapsedMilliseconds).TotalSeconds} GFLOPS");
        }

        public static void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
        {
            var a = TA == 0 ? random_matrix(m, k) : random_matrix(k, m);
            int lda = (TA == 0) ? k : m;
            var b = TB == 0 ? random_matrix(k, n) : random_matrix(n, k);
            int ldb = (TB == 0) ? n : k;

            float[] c = random_matrix(m, n);
            float[] c_gpu = random_matrix(m, n);

            int i;
            gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c_gpu, n);

            gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
            double sse = 0;
            for (i = 0; i < m * n; ++i)
            {
                sse += Math.Pow(c[i] - c_gpu[i], 2);
            }
            Console.Error.WriteLine($"Matrix Multiplication {m}x{k} * {k}x{n}, TA={TA}, TB={TB}: {sse / (m * n)} SSE");
        }

        public static int test_gpu_blas()
        {
            time_ongpu(0, 0, 64, 75, 12544);
            time_ongpu(0, 0, 64, 75, 12544);
            time_ongpu(0, 0, 64, 75, 12544);
            time_ongpu(0, 0, 64, 576, 12544);
            time_ongpu(0, 0, 256, 2304, 784);
            time_ongpu(1, 1, 2304, 256, 784);
            time_ongpu(0, 0, 512, 4608, 196);
            time_ongpu(1, 1, 4608, 512, 196);

            return 0;
        }

    }
}