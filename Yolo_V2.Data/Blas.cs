using System;
using System.Collections.Generic;
using System.Linq;
using Alea;
using Alea.CSharp;
using Alea.FSharp;
using dim3 = Alea.CudaToolkit.dim3;

namespace Yolo_V2.Data
{
    public static class Blas
    {
        public static void Reorg_cpu(float[] x, int w, int h, int c, int batch, int stride, int forward, float[] output)
        {
            int b, i, j, k;
            int outC = c / (stride * stride);

            for (b = 0; b < batch; ++b)
            {
                for (k = 0; k < c; ++k)
                {
                    for (j = 0; j < h; ++j)
                    {
                        for (i = 0; i < w; ++i)
                        {
                            int inIndex = i + w * (j + h * (k + c * b));
                            int c2 = k % outC;
                            int offset = k / outC;
                            int w2 = i * stride + offset % stride;
                            int h2 = j * stride + offset / stride;
                            int outIndex = w2 + w * stride * (h2 + h * stride * (c2 + outC * b));
                            if (forward != 0) output[outIndex] = x[inIndex];
                            else output[inIndex] = x[outIndex];
                        }
                    }
                }
            }
        }

        public static void Flatten(ref float[] x, int size, int layers, int batch, int forward)
        {
            float[] swap = new float[size * layers * batch];
            int i, c, b;
            for (b = 0; b < batch; ++b)
            {
                for (c = 0; c < layers; ++c)
                {
                    for (i = 0; i < size; ++i)
                    {
                        int i1 = b * layers * size + c * size + i;
                        int i2 = b * layers * size + i * layers + c;
                        if (forward != 0) swap[i2] = x[i1];
                        else swap[i1] = x[i2];
                    }
                }
            }

            x = swap;
        }

        public static void Weighted_sum_cpu(float[] a, float[] b, float[] s, int n, float[] c)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                c[i] = s[i] * a[i] + (1 - s[i]) * (i < b.Length ? b[i] : 0);
            }
        }

        public static void Shortcut_cpu(int batch, int w1, int h1, int c1, float[] add, int w2, int h2, int c2, float[] output)
        {
            int stride = w1 / w2;
            int sample = w2 / w1;
            if (stride < 1) stride = 1;
            if (sample < 1) sample = 1;
            int minw = (w1 < w2) ? w1 : w2;
            int minh = (h1 < h2) ? h1 : h2;
            int minc = (c1 < c2) ? c1 : c2;

            int i, j, k, b;
            for (b = 0; b < batch; ++b)
            {
                for (k = 0; k < minc; ++k)
                {
                    for (j = 0; j < minh; ++j)
                    {
                        for (i = 0; i < minw; ++i)
                        {
                            int outIndex = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                            int addIndex = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                            output[outIndex] += add[addIndex];
                        }
                    }
                }
            }
        }

        public static void Mean_cpu(float[] x, int batch, int filters, int spatial, float[] mean)
        {
            float scale = 1.0f / (batch * spatial);
            int i, j, k;
            for (i = 0; i < filters; ++i)
            {
                mean[i] = 0;
                for (j = 0; j < batch; ++j)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + i * spatial + k;
                        mean[i] += x[index];
                    }
                }
                mean[i] *= scale;
            }
        }

        public static void Scale_bias(float[] output, float[] scales, int batch, int n, int size)
        {
            int i, j, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    for (j = 0; j < size; ++j)
                    {
                        output[(b * n + i) * size + j] *= scales[i];
                    }
                }
            }
        }

        public static void Variance_cpu(float[] x, float[] mean, int batch, int filters, int spatial, float[] variance)
        {
            float scale = 1.0f / (batch * spatial - 1);
            int i, j, k;
            for (i = 0; i < filters; ++i)
            {
                variance[i] = 0;
                for (j = 0; j < batch; ++j)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + i * spatial + k;
                        variance[i] += (float)Math.Pow((x[index] - mean[i]), 2);
                    }
                }
                variance[i] *= scale;
            }
        }

        public static void Normalize_cpu(float[] x, float[] mean, float[] variance, int batch, int filters, int spatial)
        {
            int b, f, i;
            for (b = 0; b < batch; ++b)
            {
                for (f = 0; f < filters; ++f)
                {
                    for (i = 0; i < spatial; ++i)
                    {
                        int index = b * filters * spatial + f * spatial + i;
                        x[index] = (x[index] - mean[f]) / ((float)Math.Sqrt(variance[f]) + .000001f);
                    }
                }
            }
        }

        public static void Const_cpu(int n, float alpha, float[] x, int incx)
        {
            int i;
            for (i = 0; i < n; ++i) x[i * incx] = alpha;
        }

        public static void Mul_cpu(int n, float[] x, int incx, float[] y, int incy)
        {
            int i;
            for (i = 0; i < n; ++i) y[i * incy] *= x[i * incx];
        }

        public static void Pow_cpu(int n, float alpha, float[] x, int incx, float[] y, int incy)
        {
            int i;
            for (i = 0; i < n; ++i) y[i * incy] = (float)Math.Pow(x[i * incx], alpha);
        }

        public static void Axpy_cpu(int n, float alpha, float[] x, int incx, float[] y, int incy)
        {
            int i;
            for (i = 0; i < n; ++i) y[i * incy] += alpha * x[i * incx];
        }

        public static void Scal_cpu(int n, float alpha, float[] x, int incx)
        {
            int i;
            for (i = 0; i < n; ++i) x[i * incx] *= alpha;
        }

        public static void Fill_cpu(int n, float alpha, float[] x, int incx)
        {
            int i;
            for (i = 0; i < n; ++i) x[i * incx] = alpha;
        }

        public static void Copy_cpu(int n, float[] x, int incx, float[] y, int incy)
        {
            int i;
            for (i = 0; i < n; ++i)
                y[i * incy] = x[i * incx];
        }

        public static void Smooth_l1_cpu(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                float diff = truth[i] - pred[i];
                float absVal = Math.Abs(diff);
                if (absVal < 1)
                {
                    error[i] = diff * diff;
                    delta[i] = diff;
                }
                else
                {
                    error[i] = 2 * absVal - 1;
                    delta[i] = (diff < 0) ? -1 : 1;
                }
            }
        }

        public static void L2_cpu(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                float diff = truth[i] - pred[i];
                error[i] = diff * diff;
                delta[i] = diff;
            }
        }

        public static float Dot_cpu(int n, float[] x, int incx, float[] y, int incy)
        {
            int i;
            float dot = 0;
            for (i = 0; i < n; ++i) dot += x[i * incx] * y[i * incy];
            return dot;
        }

        public static void Softmax(float[] input, int n, float temp, float[] output)
        {
            int i;
            float sum = 0;
            float largest = float.MinValue;
            for (i = 0; i < n; ++i)
            {
                if (input[i] > largest) largest = input[i];
            }
            for (i = 0; i < n; ++i)
            {
                float e = (float)Math.Exp(input[i] / temp - largest / temp);
                sum += e;
                output[i] = e;
            }
            for (i = 0; i < n; ++i)
            {
                output[i] /= sum;
            }
        }

        private static void scale_bias_kernel(float[] output, float[] biases, int n, int size)
        {
            int offset = blockIdx.x * blockDim.x + threadIdx.x;
            int filter = blockIdx.y;
            int batch = blockIdx.z;

            if (offset < size) output[(batch * n + filter) * size + offset] *= biases[filter];
        }

        [GpuManaged]
        public static void scale_bias_gpu(float[] output, float[] biases, int batch, int n, int size)
        {
            var dimGrid = new Alea.dim3((size - 1) / CudaUtils.BlockSize + 1, n, batch);
            var dimBlock = new Alea.dim3(CudaUtils.BlockSize, 1, 1);
            var lp = new Alea.LaunchParam(dimGrid, dimBlock);
            Gpu.Default.Launch(scale_bias_kernel, lp, output, biases, n, size);
        }

        private static void backward_scale_kernel(float[] x_norm, float[] delta, int batch, int n, int size, float[] scale_updates)
        {
            var part = __shared__.Array<float>(CudaUtils.BlockSize);
            int i, b;
            int filter = blockIdx.x;
            int p = threadIdx.x;
            float sum = 0;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < size; i += CudaUtils.BlockSize)
                {
                    int index = p + i + size * (filter + n * b);
                    sum += (p + i < size) ? delta[index] * x_norm[index] : 0;
                }
            }
            part[p] = sum;
            DeviceFunction.SyncThreads();
            if (p == 0)
            {
                for (i = 0; i < CudaUtils.BlockSize; ++i) scale_updates[filter] += part[i];
            }
        }

        [GpuManaged]
        public static void backward_scale_gpu(float[] x_norm, float[] delta, int batch, int n, int size, float[] scale_updates)
        {
            var lp = new LaunchParam(n, CudaUtils.BlockSize);
            Gpu.Default.Launch(backward_scale_kernel, lp, x_norm, delta, batch, n, size, scale_updates);
        }

        private static void add_bias_kernel(float[] output, float[] biases, int n, int size)
        {
            int offset = blockIdx.x * blockDim.x + threadIdx.x;
            int filter = blockIdx.y;
            int batch = blockIdx.z;

            if (offset < size) output[(batch * n + filter) * size + offset] += biases[filter];
        }

        [GpuManaged]
        public static void add_bias_gpu(float[] output, float[] biases, int batch, int n, int size)
        {
            var lp = new LaunchParam(new Alea.dim3((size - 1) / CudaUtils.BlockSize + 1, n, batch), new Alea.dim3(CudaUtils.BlockSize, 1, 1));
            Gpu.Default.Launch(add_bias_kernel, lp, output, biases, n, size);
        }

        private static void backward_bias_kernel(float[] bias_updates, float[] delta, int batch, int n, int size)
        {
            var part = __shared__.Array<float>(CudaUtils.BlockSize);
            int i, b;
            int filter = blockIdx.x;
            int p = threadIdx.x;
            float sum = 0;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < size; i += CudaUtils.BlockSize)
                {
                    int index = p + i + size * (filter + n * b);
                    sum += (p + i < size) ? delta[index] : 0;
                }
            }
            part[p] = sum;
            DeviceFunction.SyncThreads();
            if (p == 0)
            {
                for (i = 0; i < CudaUtils.BlockSize; ++i) bias_updates[filter] += part[i];
            }
        }

        [GpuManaged]
        public static void backward_bias_gpu(float[] bias_updates, float[] delta, int batch, int n, int size)
        {
            var lp = new LaunchParam(n, CudaUtils.BlockSize);
            Gpu.Default.Launch(backward_bias_kernel, lp, bias_updates, delta, batch, n, size);
        }
        
        private static void adam_kernel(int N, float[] x, float[] m, float[] v, float B1, float B2, float rate, float eps, int t)
        {
            int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (index >= N) return;

            x[index] = x[index] - (float)(rate * Math.Sqrt(1.0 - Math.Pow(B2, t)) / (1.0 - Math.Pow(B1, t)) * m[index] / (Math.Sqrt(v[index]) + eps));
        }

        [GpuManaged]
        public static void adam_gpu(int n, float[] x, float[] m, float[] v, float B1, float B2, float rate, float eps, int t)
        {
            var lp = new LaunchParam(n, CudaUtils.BlockSize);
            Gpu.Default.Launch(adam_kernel, lp, n, x, m, v, B1, B2, rate, eps, t);
        }

        private static void normalize_kernel(int N, float[] x, float[] mean, float[] variance, int batch, int filters, int spatial)
        {
            int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (index >= N) return;
            int f = (index / spatial) % filters;

            x[index] = (x[index] - mean[f]) / ((float)Math.Sqrt(variance[f]) + .000001f);
        }

        private static void normalize_delta_kernel(int N, float[] x, float[] mean, float[] variance, float[] mean_delta, float[] variance_delta, int batch, int filters, int spatial, float[] delta)
        {
            int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (index >= N) return;
            int f = (index / spatial) % filters;

            delta[index] = delta[index] * 1.0f / (float)(Math.Sqrt(variance[f]) + .000001f) + variance_delta[f] * 2.0f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f] / (spatial * batch);
        }

        [GpuManaged]
        public static void normalize_delta_gpu(float[] x, float[] mean, float[] variance, float[] mean_delta, float[] variance_delta, int batch, int filters, int spatial, float[] delta)
        {
            int N = batch * filters * spatial;
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(normalize_delta_kernel, lp, N, x, mean, variance, mean_delta, variance_delta, batch,
                filters, spatial, delta);
        }

        private static void variance_delta_kernel(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, float[] variance_delta)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= filters) return;
            int j, k;
            variance_delta[i] = 0;
            for (j = 0; j < batch; ++j)
            {
                for (k = 0; k < spatial; ++k)
                {
                    int index = j * filters * spatial + i * spatial + k;
                    variance_delta[i] += delta[index] * (x[index] - mean[i]);
                }
            }
            variance_delta[i] *= -.5f * (float)Math.Pow(variance[i] + .000001f, (-3.0f/ 2.0f));
        }

        private static void accumulate_kernel(float[] x, int n, int groups, float[] sum)
        {
            int k;
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= groups) return;
            sum[i] = 0;
            for (k = 0; k < n; ++k)
            {
                sum[i] += x[k * groups + i];
            }
        }

        private static void fast_mean_delta_kernel(float[] delta, float[] variance, int batch, int filters, int spatial, float[] mean_delta)
        {
            int threads = CudaUtils.BlockSize;
            var local = __shared__.Array<float>(threads);

            int id = threadIdx.x;
            local[id] = 0;

            int filter = blockIdx.x;

            int i, j;
            for (j = 0; j < batch; ++j)
            {
                for (i = 0; i < spatial; i += threads)
                {
                    int index = j * spatial * filters + filter * spatial + i + id;
                    local[id] += (i + id < spatial) ? delta[index] : 0;
                }
            }

            if (id == 0)
            {
                mean_delta[filter] = 0;
                for (i = 0; i < threads; ++i)
                {
                    mean_delta[filter] += local[i];
                }
                mean_delta[filter] *= (-1.0f / (float)Math.Sqrt(variance[filter] + .000001f));
            }
        }

        private static void fast_variance_delta_kernel(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, float[] variance_delta)
        {
            int threads = CudaUtils.BlockSize;
            var local = __shared__.Array<float>(threads);

            int id = threadIdx.x;
            local[id] = 0;

            int filter = blockIdx.x;

            int i, j;
            for (j = 0; j < batch; ++j)
            {
                for (i = 0; i < spatial; i += threads)
                {
                    int index = j * spatial * filters + filter * spatial + i + id;

                    local[id] += (i + id < spatial) ? delta[index] * (x[index] - mean[filter]) : 0;
                }
            }

            if (id == 0)
            {
                variance_delta[filter] = 0;
                for (i = 0; i < threads; ++i)
                {
                    variance_delta[filter] += local[i];
                }
                variance_delta[filter] *= -0.5f * (float)Math.Pow(variance[filter] + .000001f, -3.0f / 2.0f);
            }
        }

        private static void mean_delta_kernel(float[] delta, float[] variance, int batch, int filters, int spatial, float[] mean_delta)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= filters) return;
            int j, k;
            mean_delta[i] = 0;
            for (j = 0; j < batch; ++j)
            {
                for (k = 0; k < spatial; ++k)
                {
                    int index = j * filters * spatial + i * spatial + k;
                    mean_delta[i] += delta[index];
                }
            }
            mean_delta[i] *= (-1.0f / (float)Math.Sqrt(variance[i] + 0.000001f));
        }

        [GpuManaged]
        public static void mean_delta_gpu(float[] delta, float[] variance, int batch, int filters, int spatial, float[] mean_delta)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(filters), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(mean_delta_kernel, lp, delta, variance, batch, filters, spatial, mean_delta);
        }

        [GpuManaged]
        public static void fast_mean_delta_gpu(float[] delta, float[] variance, int batch, int filters, int spatial, float[] mean_delta)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(filters), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(fast_mean_delta_kernel, lp, delta, variance, batch, filters, spatial, mean_delta);
        }

        [GpuManaged]
        public static void fast_variance_delta_gpu(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, float[] variance_delta)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(filters), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(fast_variance_delta_kernel, lp, x, delta, mean, variance, batch, filters, spatial,
                variance_delta);
        }

        private static void mean_kernel(float[] x, int batch, int filters, int spatial, float[] mean)
        {
            float scale = 1.0f/ (batch * spatial);
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= filters) return;
            int j, k;
            mean[i] = 0;
            for (j = 0; j < batch; ++j)
            {
                for (k = 0; k < spatial; ++k)
                {
                    int index = j * filters * spatial + i * spatial + k;
                    mean[i] += x[index];
                }
            }
            mean[i] *= scale;
        }

        private static void variance_kernel(float[] x, float[] mean, int batch, int filters, int spatial, float[] variance)
        {
            float scale = 1.0f/ (batch * spatial - 1);
            int j, k;
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= filters) return;
            variance[i] = 0;
            for (j = 0; j < batch; ++j)
            {
                for (k = 0; k < spatial; ++k)
                {
                    int index = j * filters * spatial + i * spatial + k;
                    variance[i] += (float)Math.Pow((x[index] - mean[i]), 2);
                }
            }
            variance[i] *= scale;
        }

        private static void reorg_kernel(int N, float[] x, int w, int h, int c, int batch, int stride, int forward, float[] output)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= N) return;
            int in_index = i;
            int in_w = i % w;
            i = i / w;
            int in_h = i % h;
            i = i / h;
            int in_c = i % c;
            i = i / c;
            int b = i % batch;

            int out_c = c / (stride * stride);

            int c2 = in_c % out_c;
            int offset = in_c / out_c;
            int w2 = in_w * stride + offset % stride;
            int h2 = in_h * stride + offset / stride;
            int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));

            if (forward != 0) output[out_index] = x[in_index];
            else output[in_index] = x[out_index];
        }

        private static void axpy_kernel(int N, float ALPHA, float[] X, int OFFX, int INCX, float[] Y, int OFFY, int INCY)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
        }

        private static void pow_kernel(int N, float ALPHA, float[] X, int INCX, float[] Y, int INCY)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) Y[i * INCY] = (float)Math.Pow(X[i * INCX], ALPHA);
        }

        private static void const_kernel(int N, float ALPHA, float[] X, int INCX)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) X[i * INCX] = ALPHA;
        }

        private static void constrain_kernel(int N, float ALPHA, float[] X, int INCX)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) X[i * INCX] = Math.Min(ALPHA, Math.Max(-ALPHA, X[i * INCX]));
        }

        private static void supp_kernel(int N, float ALPHA, float[] X, int INCX)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N)
            {
                if ((X[i * INCX] * X[i * INCX]) < (ALPHA * ALPHA)) X[i * INCX] = 0;
            }
        }

        private static void scal_kernel(int N, float ALPHA, float[] X, int INCX)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) X[i * INCX] *= ALPHA;
        }

        private static void fill_kernel(int N, float ALPHA, float[] X, int INCX)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) X[i * INCX] = ALPHA;
        }

        private static void mask_kernel(int n, float[] x, float mask_num, float[] mask)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n && mask[i] == mask_num) x[i] = mask_num;
        }

        private static void copy_kernel(int N, float[] X, int OFFX, int INCX, float[] Y, int OFFY, int INCY)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) Y[i * INCY + OFFY] = X[i * INCX + OFFX];
        }

        private static void mul_kernel(int N, float[] X, int INCX, float[] Y, int INCY)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < N) Y[i * INCY] *= X[i * INCX];
        }
        
        [GpuManaged]
        public static void normalize_gpu(float[] x, float[] mean, float[] variance, int batch, int filters, int spatial)
        {
            var N = batch * filters * spatial;
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(normalize_kernel, lp, N, x, mean, variance, batch, filters, spatial);
        }

        private static void fast_mean_kernel(float[] x, int batch, int filters, int spatial, float[] mean)
        {
            int threads = CudaUtils.BlockSize;
            var local = __shared__.Array<float>(threads);

            int id = threadIdx.x;
            local[id] = 0;

            int filter = blockIdx.x;

            int i, j;
            for (j = 0; j < batch; ++j)
            {
                for (i = 0; i < spatial; i += threads)
                {
                    int index = j * spatial * filters + filter * spatial + i + id;
                    local[id] += (i + id < spatial) ? x[index] : 0;
                }
            }

            if (id == 0)
            {
                mean[filter] = 0;
                for (i = 0; i < threads; ++i)
                {
                    mean[filter] += local[i];
                }
                mean[filter] /= spatial * batch;
            }
        }

        private static void fast_variance_kernel(float[] x, float[] mean, int batch, int filters, int spatial, float[] variance)
        {
            int threads = CudaUtils.BlockSize;
            var local = __shared__.Array<float>(threads);

            int id = threadIdx.x;
            local[id] = 0;

            int filter = blockIdx.x;

            int i, j;
            for (j = 0; j < batch; ++j)
            {
                for (i = 0; i < spatial; i += threads)
                {
                    int index = j * spatial * filters + filter * spatial + i + id;

                    local[id] += (i + id < spatial) ? (float)Math.Pow((x[index] - mean[filter]), 2) : 0;
                }
            }

            if (id == 0)
            {
                variance[filter] = 0;
                for (i = 0; i < threads; ++i)
                {
                    variance[filter] += local[i];
                }
                variance[filter] /= (spatial * batch - 1);
            }
        }

        [GpuManaged]
        public static void fast_mean_gpu(float[] x, int batch, int filters, int spatial, float[] mean)
        {
            var lp = new LaunchParam(filters, CudaUtils.BlockSize);
            Gpu.Default.Launch(fast_mean_kernel, lp, x, batch, filters, spatial, mean);
        }

        [GpuManaged]
        public static void fast_variance_gpu(float[] x, float[] mean, int batch, int filters, int spatial, float[] variance)
        {
            var lp = new LaunchParam(filters, CudaUtils.BlockSize);
            Gpu.Default.Launch(fast_variance_kernel, lp, x, mean, batch, filters, spatial, variance);
        }


        [GpuManaged]
        public static void mean_gpu(float[] x, int batch, int filters, int spatial, float[] mean)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(filters), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(mean_kernel, lp, x, batch, filters, spatial, mean);
        }

        [GpuManaged]
        public static void variance_gpu(float[] x, float[] mean, int batch, int filters, int spatial, float[] variance)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(filters), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(variance_kernel, lp, x, mean, batch, filters, spatial, variance);
        }

        [GpuManaged]
        public static void axpy_ongpu(int N, float ALPHA, float[] X, int INCX, float[] Y, int INCY)
        {
            axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
        }

        [GpuManaged]
        public static void pow_ongpu(int N, float ALPHA, float[] X, int INCX, float[] Y, int INCY)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(pow_kernel, lp, N, ALPHA, X, INCX, Y, INCY);
        }

        [GpuManaged]
        public static void axpy_ongpu_offset(int N, float ALPHA, float[] X, int OFFX, int INCX, float[] Y, int OFFY, int INCY)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(axpy_kernel, lp, N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
        }

        [GpuManaged]
        public static void copy_ongpu(int N, float[] X, int INCX, float[] Y, int INCY)
        {
            copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
        }

        [GpuManaged]
        public static void mul_ongpu(int N, float[] X, int INCX, float[] Y, int INCY)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(mul_kernel, lp, N, X, INCX, Y, INCY);
        }

        [GpuManaged]
        public static void copy_ongpu_offset(int N, float[] X, int OFFX, int INCX, float[] Y, int OFFY, int INCY)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(copy_kernel, lp, N, X, OFFX, INCX, Y, OFFY, INCY);
        }

        private static void flatten_kernel(int N, float[] x, int spatial, int layers, int batch, int forward, float[] output)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= N) return;
            int in_s = i % spatial;
            i = i / spatial;
            int in_c = i % layers;
            i = i / layers;
            int b = i;

            int i1 = b * layers * spatial + in_c * spatial + in_s;
            int i2 = b * layers * spatial + in_s * layers + in_c;

            if (forward != 0) output[i2] = x[i1];
            else output[i1] = x[i2];
        }

        [GpuManaged]
        public static void flatten_ongpu(float[] x, int spatial, int layers, int batch, int forward, float[] output)
        {
            int size = spatial * batch * layers;
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(flatten_kernel, lp, size, x, spatial, layers, batch, forward, output);
        }

        [GpuManaged]
        public static void reorg_ongpu(float[] x, int w, int h, int c, int batch, int stride, int forward, float[] output)
        {
            int size = w * h * c * batch;
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(reorg_kernel, lp, size, x, w, h, c, batch, stride, forward, output);
        }

        [GpuManaged]
        public static void mask_ongpu(int N, float[] X, float mask_num, float[] mask)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(mask_kernel, lp, N, X, mask_num, mask);
        }

        [GpuManaged]
        public static void const_ongpu(int N, float ALPHA, float[] X, int INCX)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(const_kernel, lp, N, ALPHA, X, INCX);
        }

        [GpuManaged]
        public static void constrain_ongpu(int N, float ALPHA, float[] X, int INCX)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(constrain_kernel, lp, N, ALPHA, X, INCX);
        }
        
        [GpuManaged]
        public static void scal_ongpu(int N, float ALPHA, float[] X, int INCX)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(scal_kernel, lp, N, ALPHA, X, INCX);
        }

        [GpuManaged]
        public static void supp_ongpu(int N, float ALPHA, float[] X, int INCX)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(supp_kernel, lp, N, ALPHA, X, INCX);
        }

        [GpuManaged]
        public static void fill_ongpu(int N, float ALPHA, float[] X, int INCX)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(N), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(fill_kernel, lp, N, ALPHA, X, INCX);
        }

        private static void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float[] add, int w2, int h2, int c2, float[] output)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= size) return;
            int i = id % minw;
            id /= minw;
            int j = id % minh;
            id /= minh;
            int k = id % minc;
            id /= minc;
            int b = id % batch;

            int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
            int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
            output[out_index] += add[add_index];
        }

        [GpuManaged]
        public static void shortcut_gpu(int batch, int w1, int h1, int c1, float[] add, int w2, int h2, int c2, float[] output)
        {
            int minw = (w1 < w2) ? w1 : w2;
            int minh = (h1 < h2) ? h1 : h2;
            int minc = (c1 < c2) ? c1 : c2;

            int stride = w1 / w2;
            int sample = w2 / w1;
            if (stride < 1) stride = 1;
            if (sample < 1) sample = 1;

            int size = batch * minw * minh * minc;
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(shortcut_kernel, lp, size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2,
                h2, c2, output);
        }

        private static void smooth_l1_kernel(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                float diff = truth[i] - pred[i];
                float abs_val = Math.Abs(diff);
                if (abs_val < 1)
                {
                    error[i] = diff * diff;
                    delta[i] = diff;
                }
                else
                {
                    error[i] = 2 * abs_val - 1;
                    delta[i] = (diff < 0) ? -1 : 1;
                }
            }
        }

        [GpuManaged]
        public static void smooth_l1_gpu(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(smooth_l1_kernel, lp, n, pred, truth, delta, error);
        }

        private static void l2_kernel(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                float diff = truth[i] - pred[i];
                error[i] = diff * diff; //I know this is technically wrong, deal with it.
                delta[i] = diff;
            }
        }

        [GpuManaged]
        public static void l2_gpu(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(l2_kernel, lp, n, pred, truth, delta, error);
        }
        
        private static void weighted_sum_kernel(int n, float[] a, float[] b, float[] s, float[] c)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                c[i] = s[i] * a[i] + (1 - s[i]) * (i < b.Length ? b[i] : 0);
            }
        }

        [GpuManaged]
        public static void weighted_sum_gpu(float[] a, float[] b, float[] s, int num, float[] c)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(num), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(weighted_sum_kernel, lp, num, a, b, s, c);
        }

        private static void weighted_delta_kernel(int n, float[] a, float[] b, float[] s, float[] da, float[] db, float[] ds, float[] dc)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                if (i < da.Length) da[i] += dc[i] * s[i];
                db[i] += dc[i] * (1 - s[i]);
                ds[i] += dc[i] * a[i] + dc[i] * -b[i];
            }
        }

        [GpuManaged]
        public static void weighted_delta_gpu(float[] a, float[] b, float[] s, float[] da, float[] db, float[] ds, int num, float[] dc)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(num), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(weighted_delta_kernel, lp, num, a, b, s, da, db, ds, dc);
        }

        private static void mult_add_into_kernel(int n, float[] a, float[] b, float[] c)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                c[i] += a[i] * b[i];
            }
        }

        [GpuManaged]
        public static void mult_add_into_gpu(int num, float[] a, float[] b, float[] c)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(num), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(mult_add_into_kernel, lp, num, a, b, c);
        }
        
        private static void softmax_device(int n, float[] input, float temp, float[] output)
        {
            int i;
            float sum = 0;
            float largest = float.NegativeInfinity;
            for (i = 0; i < n; ++i)
            {
                var val = input[i];
                largest = (val > largest) ? val : largest;
            }
            for (i = 0; i < n; ++i)
            {
                float e = (float)Math.Exp(input[i] / temp - largest / temp);
                sum += e;
                output[i] = e;
            }
            for (i = 0; i < n; ++i)
            {
                output[i] /= sum;
            }
        }

        private static void softmax_kernel(int n, int offset, int batch, float[] input, float temp, float[] output)
        {
            int b = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (b >= batch) return;
            var tempOut = output.Skip(b * offset).ToArray();
            var tempIn = input.Skip(b * offset).ToArray();
            softmax_device(n, tempIn, temp, tempOut);
            Layer.CombineLists(output, b * offset, tempOut);
            Layer.CombineLists(input, b * offset, tempIn);
        }

        [GpuManaged]
        public static void softmax_gpu(float[] input, int n, int offset, int groups, float temp, float[] output)
        {
            int inputs = n;
            int batch = groups;
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(batch), new Alea.dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(softmax_kernel, lp, inputs, offset, batch, input, temp, output);
        }
    }
}