using System;
using Alea;
using Alea.CSharp;

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

        public static void Pow_cpu(int n, float alpha, float[] x, float[] y)
        {
            int i;
            for (i = 0; i < n; ++i) y[i] = (float)Math.Pow(x[i], alpha);
        }

        public static void Axpy_cpu(int n, float alpha, float[] from, float[] to, int fromStart = 0, int toStart = 0)
        {
            int i;
            for (i = 0; i < n; ++i) to[i + toStart] += alpha * from[i + fromStart];
        }

        public static void Scal_cpu(int n, float alpha, float[] x, int incx, int xStart = 0)
        {
            int i;
            for (i = 0; i < n; ++i) x[xStart + i * incx] *= alpha;
        }

        public static void Fill_cpu(int n, float alpha, float[] x, int incx)
        {
            int i;
            for (i = 0; i < n; ++i) x[i * incx] = alpha;
        }

        public static void Copy_cpu(int n, float[] x, float[] y, int xStart = 0, int yStart = 0)
        {
            Array.Copy(x, xStart, y, yStart, n);
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

        public static void Softmax(float[] input, int n, float temp, float[] output, int inputStart = 0, int outputStart = 0)
        {
            int i;
            float sum = 0;
            float largest = float.MinValue;
            for (i = 0; i < n; ++i)
            {
                if (input[i + inputStart] > largest) largest = input[i + inputStart];
            }
            for (i = 0; i < n; ++i)
            {
                float e = (float)Math.Exp(input[i + inputStart] / temp - largest / temp);
                sum += e;
                output[i + outputStart] = e;
            }
            for (i = 0; i < n; ++i)
            {
                output[i + outputStart] /= sum;
            }
        }

        private static void scale_bias_kernel(float[] output, float[] biases, int n, int size)
        {
            int offset = blockIdx.x * blockDim.x + threadIdx.x;
            int filter = blockIdx.y;
            int batch = blockIdx.z;

            if (offset < size)
            {
                var orig = output[(batch * n + filter) * size + offset];
                output[(batch * n + filter) * size + offset] *= biases[filter];
            }
        }

        [GpuManaged]
        public static void scale_bias_gpu(ref float[] output, float[] biases, int batch, int n, int size)
        {
            var dimGrid = new dim3((size - 1) / CudaUtils.BlockSize + 1, n, batch);
            var dimBlock = new dim3(CudaUtils.BlockSize, 1, 1);
            var lp = new LaunchParam(dimGrid, dimBlock);
            Gpu.Default.Launch(scale_bias_kernel, lp, output, biases, n, size);
        }

        private static void backward_scale_kernel(float[] xNorm, float[] delta, int batch, int n, int size, float[] scaleUpdates)
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
                    sum += (p + i < size) ? delta[index] * xNorm[index] : 0;
                }
            }
            part[p] = sum;
            DeviceFunction.SyncThreads();
            if (p == 0)
            {
                for (i = 0; i < CudaUtils.BlockSize; ++i) scaleUpdates[filter] += part[i];
            }
        }

        public static void backward_scale_gpu(float[] xNorm, float[] delta, int batch, int n, int size, ref float[] scaleUpdates)
        {
            var lp = new LaunchParam(n, CudaUtils.BlockSize);
            var tempScaleUpdates = Gpu.Default.Allocate(scaleUpdates);
            Gpu.Default.Launch(backward_scale_kernel, lp, xNorm, delta, batch, n, size, scaleUpdates);
            scaleUpdates = Gpu.CopyToHost(tempScaleUpdates);
            Gpu.Free(tempScaleUpdates);
        }

        private static void add_bias_kernel(float[] output, float[] biases, int n, int size)
        {
            int offset = blockIdx.x * blockDim.x + threadIdx.x;
            int filter = blockIdx.y;
            int batch = blockIdx.z;

            if (offset < size) output[(batch * n + filter) * size + offset] += biases[filter];
        }

        public static void add_bias_gpu(ref float[] output, float[] biases, int batch, int n, int size)
        {
            var lp = new LaunchParam(new dim3((size - 1) / CudaUtils.BlockSize + 1, n, batch), new dim3(CudaUtils.BlockSize, 1, 1));
            var temp = Gpu.Default.Allocate(output);
            Gpu.Default.Launch(add_bias_kernel, lp, temp, biases, n, size);
            output = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void backward_bias_kernel(float[] biasUpdates, float[] delta, int batch, int n, int size)
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
                for (i = 0; i < CudaUtils.BlockSize; ++i) biasUpdates[filter] += part[i];
            }
        }

        public static void backward_bias_gpu(ref float[] biasUpdates, float[] delta, int batch, int n, int size)
        {
            var lp = new LaunchParam(n, CudaUtils.BlockSize);
            var temp = Gpu.Default.Allocate(biasUpdates);
            Gpu.Default.Launch(backward_bias_kernel, lp, temp, delta, batch, n, size);
            biasUpdates = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }
        
        private static void adam_kernel(int n, float[] x, float[] m, float[] v, float b1, float b2, float rate, float eps, int t)
        {
            int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (index >= n) return;

            x[index] = x[index] - (float)(rate * Math.Sqrt(1.0 - Math.Pow(b2, t)) / (1.0 - Math.Pow(b1, t)) * m[index] / (Math.Sqrt(v[index]) + eps));
        }

        public static void adam_gpu(int n, ref float[] x, float[] m, float[] v, float b1, float b2, float rate, float eps, int t)
        {
            var lp = new LaunchParam(n, CudaUtils.BlockSize);
            var temp = Gpu.Default.Allocate(x);
            Gpu.Default.Launch(adam_kernel, lp, n, temp, m, v, b1, b2, rate, eps, t);
            x = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void normalize_kernel(int n, float[] x, float[] mean, float[] variance, int batch, int filters, int spatial)
        {
            int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (index >= n) return;
            int f = (index / spatial) % filters;

            x[index] = (x[index] - mean[f]) / ((float)Math.Sqrt(variance[f]) + .000001f);
        }

        private static void normalize_delta_kernel(int n, float[] x, float[] mean, float[] variance, float[] meanDelta, float[] varianceDelta, int batch, int filters, int spatial, float[] delta)
        {
            int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (index >= n) return;
            int f = (index / spatial) % filters;

            delta[index] = delta[index] * 1.0f / (float)(Math.Sqrt(variance[f]) + .000001f) + varianceDelta[f] * 2.0f * (x[index] - mean[f]) / (spatial * batch) + meanDelta[f] / (spatial * batch);
        }

        public static void normalize_delta_gpu(float[] x, float[] mean, float[] variance, float[] meanDelta, float[] varianceDelta, int batch, int filters, int spatial,ref float[] delta)
        {
            int n = batch * filters * spatial;
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(delta);
            Gpu.Default.Launch(normalize_delta_kernel, lp, n, x, mean, variance, meanDelta, varianceDelta, batch,
                filters, spatial, temp);
            delta = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void fast_mean_delta_kernel(float[] delta, float[] variance, int batch, int filters, int spatial, float[] meanDelta)
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
                meanDelta[filter] = 0;
                for (i = 0; i < threads; ++i)
                {
                    meanDelta[filter] += local[i];
                }
                meanDelta[filter] *= (-1.0f / (float)Math.Sqrt(variance[filter] + .000001f));
            }
        }

        private static void fast_variance_delta_kernel(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, float[] varianceDelta)
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
                varianceDelta[filter] = 0;
                for (i = 0; i < threads; ++i)
                {
                    varianceDelta[filter] += local[i];
                }
                varianceDelta[filter] *= -0.5f * (float)Math.Pow(variance[filter] + .000001f, -3.0f / 2.0f);
            }
        }

        public static void fast_mean_delta_gpu(float[] delta, float[] variance, int batch, int filters, int spatial, ref float[] meanDelta)
        {
            var lp = CudaUtils.cuda_gridsize(filters);
            var temp = Gpu.Default.Allocate(meanDelta);
            Gpu.Default.Launch(fast_mean_delta_kernel, lp, delta, variance, batch, filters, spatial, temp);
            meanDelta = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void fast_variance_delta_gpu(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, ref float[] varianceDelta)
        {
            var lp = CudaUtils.cuda_gridsize(filters);
            var temp = Gpu.Default.Allocate(varianceDelta);
            Gpu.Default.Launch(fast_variance_delta_kernel, lp, x, delta, mean, variance, batch, filters, spatial,
                temp);
            varianceDelta = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void reorg_kernel(int n, float[] x, int w, int h, int c, int batch, int stride, int forward, float[] output)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            int inIndex = i;
            int inW = i % w;
            i = i / w;
            int inH = i % h;
            i = i / h;
            int inC = i % c;
            i = i / c;
            int b = i % batch;

            int outC = c / (stride * stride);

            int c2 = inC % outC;
            int offset = inC / outC;
            int w2 = inW * stride + offset % stride;
            int h2 = inH * stride + offset / stride;
            int outIndex = w2 + w * stride * (h2 + h * stride * (c2 + outC * b));

            if (forward != 0) output[outIndex] = x[inIndex];
            else output[inIndex] = x[outIndex];
        }

        private static void axpy_kernel(int n, float alpha, float[] x,  float[] y, int xStart = 0, int yStart = 0)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) y[ i + yStart ] += alpha * x[i + xStart];
        }

        private static void pow_kernel(int n, float alpha, float[] x, float[] y)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i] = (float)Math.Pow(x[i], alpha);
        }

        private static void const_kernel(int n, float alpha, float[] x, int incx)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) x[i * incx] = alpha;
        }

        private static void constrain_kernel(int n, float alpha, float[] x, int incx)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) x[i * incx] = Math.Min(alpha, Math.Max(-alpha, x[i * incx]));
        }

        private static void supp_kernel(int n, float alpha, float[] x, int incx)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                if ((x[i * incx] * x[i * incx]) < (alpha * alpha)) x[i * incx] = 0;
            }
        }

        private static void scal_kernel(int n, float alpha, float[] x, int incx)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) x[i * incx] *= alpha;
        }

        private static void fill_kernel(int n, float alpha, float[] x, int incx, int startx = 0)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) x[startx + i * incx] = alpha;
        }

        private static void mask_kernel(int n, float[] x, float maskNum, float[] mask)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n && mask[i] == maskNum) x[i] = maskNum;
        }

        private static void copy_kernel(int n, float[] from, float[] to, int xStart = 0, int yStart = 0)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) to[i + yStart] = from[i * + xStart];
        }

        private static void mul_kernel(int n, float[] x, int incx, float[] y, int incy)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i * incy] *= x[i * incx];
        }
        
        [GpuManaged]
        public static void normalize_gpu(ref float[] x, float[] mean, float[] variance, int batch, int filters, int spatial)
        {
            var n = batch * filters * spatial;
            
            var lp = CudaUtils.cuda_gridsize(n);
            Gpu.Default.Launch(normalize_kernel, lp, n, x, mean, variance, batch, filters, spatial);

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

        public static void fast_mean_gpu(float[] x, int batch, int filters, int spatial, ref float[] mean)
        {
            var lp = new LaunchParam(filters, CudaUtils.BlockSize);
            var temp = Gpu.Default.Allocate(mean);
            Gpu.Default.Launch(fast_mean_kernel, lp, x, batch, filters, spatial, temp);
            mean = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void fast_variance_gpu(float[] x, float[] mean, int batch, int filters, int spatial, ref float[] variance)
        {
            var lp = new LaunchParam(filters, CudaUtils.BlockSize);
            var temp = Gpu.Default.Allocate(variance);
            Gpu.Default.Launch(fast_variance_kernel, lp, x, mean, batch, filters, spatial, temp);
            variance = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void axpy_ongpu(int n, float alpha, float[] x, float[] y, int xStart = 0, int yStart = 0)
        {
            axpy_ongpu_offset(n, alpha, x, ref y,  xStart, yStart);
        }

        public static void pow_ongpu(int n, float alpha, float[] x, ref float[] y)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(y);
            Gpu.Default.Launch(pow_kernel, lp, n, alpha, x, temp);
            y = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void axpy_ongpu_offset(int n, float alpha, float[] x, ref float[] y, int xStart = 0, int yStart = 0)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(y);
            Gpu.Default.Launch(axpy_kernel, lp, n, alpha, x, temp, xStart, yStart);
            y = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void copy_ongpu(int n, float[] from, ref float[] to, int xStart = 0, int yStart = 0)
        {
            copy_ongpu_offset(n, from, ref to, xStart, yStart);
        }

        public static void mul_ongpu(int n, float[] x, int incx,ref float[] y, int incy)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(y);
            Gpu.Default.Launch(mul_kernel, lp, n, x, incx, temp, incy);
            y = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void copy_ongpu_offset(int n, float[] from,ref float[] to, int xStart = 0, int yStart = 0)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(to);
            Gpu.Default.Launch(copy_kernel, lp, n, from, temp, xStart, yStart);
            to = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void flatten_kernel(int n, float[] x, int spatial, int layers, int batch, int forward, float[] output)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            int inS = i % spatial;
            i = i / spatial;
            int inC = i % layers;
            i = i / layers;
            int b = i;

            int i1 = b * layers * spatial + inC * spatial + inS;
            int i2 = b * layers * spatial + inS * layers + inC;

            if (forward != 0) output[i2] = x[i1];
            else output[i1] = x[i2];
        }

        public static void flatten_ongpu(float[] x, int spatial, int layers, int batch, int forward,ref float[] output)
        {
            int size = spatial * batch * layers;
            var lp = CudaUtils.cuda_gridsize(size);
            var temp = Gpu.Default.Allocate(output);
            Gpu.Default.Launch(flatten_kernel, lp, size, x, spatial, layers, batch, forward, temp);
            output = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void reorg_ongpu(float[] x, int w, int h, int c, int batch, int stride, int forward,ref float[] output)
        {
            int size = w * h * c * batch;
            var lp = CudaUtils.cuda_gridsize(size);
            var temp = Gpu.Default.Allocate(output);
            Gpu.Default.Launch(reorg_kernel, lp, size, x, w, h, c, batch, stride, forward, output);
            output = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void mask_ongpu(int n,ref float[] x, float maskNum, float[] mask)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(x);
            Gpu.Default.Launch(mask_kernel, lp, n, temp, maskNum, mask);
            x = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void const_ongpu(int n, float alpha,ref float[] x, int incx)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(x);
            Gpu.Default.Launch(const_kernel, lp, n, alpha, temp, incx);
            x = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void constrain_ongpu(int n, float alpha,ref float[] x, int incx)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(x);
            Gpu.Default.Launch(constrain_kernel, lp, n, alpha, temp, incx);
            x = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }
        
        [GpuManaged]
        public static void scal_ongpu(int n, float alpha,ref float[] x, int incx)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            Gpu.Default.Launch(scal_kernel, lp, n, alpha, x, incx);
        }

        public static void supp_ongpu(int n, float alpha,ref float[] x, int incx)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(x);
            Gpu.Default.Launch(supp_kernel, lp, n, alpha, temp, incx);
            x = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        public static void fill_ongpu(int n, float alpha,ref float[] x, int incx, int startx = 0)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var temp = Gpu.Default.Allocate(x);
            Gpu.Default.Launch(fill_kernel, lp, n, alpha, temp, incx, startx);
            x = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
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

            int outIndex = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
            int addIndex = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
            output[outIndex] += add[addIndex];
        }

        public static void shortcut_gpu(int batch, int w1, int h1, int c1, float[] add, int w2, int h2, int c2,ref float[] output)
        {
            int minw = (w1 < w2) ? w1 : w2;
            int minh = (h1 < h2) ? h1 : h2;
            int minc = (c1 < c2) ? c1 : c2;

            int stride = w1 / w2;
            int sample = w2 / w1;
            if (stride < 1) stride = 1;
            if (sample < 1) sample = 1;

            int size = batch * minw * minh * minc;
            var lp = CudaUtils.cuda_gridsize(size);
            var temp = Gpu.Default.Allocate(output);
            Gpu.Default.Launch(shortcut_kernel, lp, size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2,
                h2, c2, temp);
            output = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }

        private static void smooth_l1_kernel(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
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

        public static void smooth_l1_gpu(int n, float[] pred, float[] truth, ref float[] delta,ref float[] error)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var tempDelta = Gpu.Default.Allocate(delta);
            var tempError = Gpu.Default.Allocate(error);
            Gpu.Default.Launch(smooth_l1_kernel, lp, n, pred, truth, tempDelta, tempError);
            delta = Gpu.CopyToHost(tempDelta);
            Gpu.Free(tempDelta);
            error = Gpu.CopyToHost(tempError);
            Gpu.Free(tempError);
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

        public static void l2_gpu(int n, float[] pred, float[] truth, float[] delta, float[] error)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var tempDelta = Gpu.Default.Allocate(delta);
            var tempError = Gpu.Default.Allocate(error);
            Gpu.Default.Launch(l2_kernel, lp, n, pred, truth, tempDelta, tempError);
            delta = Gpu.CopyToHost(tempDelta);
            Gpu.Free(tempDelta);
            error = Gpu.CopyToHost(tempError);
            Gpu.Free(tempError);
        }
        
        private static void weighted_sum_kernel(int n, float[] a, float[] b, float[] s, float[] c)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                c[i] = s[i] * a[i] + (1 - s[i]) * (i < b.Length ? b[i] : 0);
            }
        }

        public static void weighted_sum_gpu(float[] a, float[] b, float[] s, int num,ref float[] c)
        {
            var lp = CudaUtils.cuda_gridsize(num);
            var temp = Gpu.Default.Allocate(c);
            Gpu.Default.Launch(weighted_sum_kernel, lp, num, a, b, s, temp);
            c = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
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

        public static void weighted_delta_gpu(float[] a, float[] b, float[] s,ref float[] da,ref float[] db,ref float[] ds, int num, float[] dc)
        {
            var lp = CudaUtils.cuda_gridsize(num);
            var tempDa = Gpu.Default.Allocate(da);
            var tempDb = Gpu.Default.Allocate(db);
            var tempDs = Gpu.Default.Allocate(ds);
            Gpu.Default.Launch(weighted_delta_kernel, lp, num, a, b, s, tempDa, tempDb, tempDs, dc);
            da = Gpu.CopyToHost(tempDa);
            Gpu.Free(tempDa);
            db = Gpu.CopyToHost(tempDb);
            Gpu.Free(tempDb);
            ds = Gpu.CopyToHost(tempDs);
            Gpu.Free(tempDs);
        }

        private static void mult_add_into_kernel(int n, float[] a, float[] b, float[] c)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n)
            {
                c[i] += a[i] * b[i];
            }
        }

        public static void mult_add_into_gpu(int num, float[] a, float[] b,ref float[] c)
        {
            var lp = CudaUtils.cuda_gridsize(num);
            var temp = Gpu.Default.Allocate(c);
            Gpu.Default.Launch(mult_add_into_kernel, lp, num, a, b, temp);
            c = Gpu.CopyToHost(temp);
            Gpu.Free(temp);
        }
        

        private static void softmax_kernel(int n, int offset, int batch, float[] input, float temp, float[] output, int inputStart = 0, int outputStart = 0)
        {
            int b = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (b >= batch) return;
            inputStart += offset;
            outputStart += offset;
            int i;
            float sum = 0;
            float largest = float.NegativeInfinity;
            for (i = 0; i < n; ++i)
            {
                var val = input[i + inputStart];
                largest = (val > largest) ? val : largest;
            }
            for (i = 0; i < n; ++i)
            {
                float e = (float)Math.Exp(input[i + inputStart] / temp - largest / temp);
                sum += e;
                output[i + outputStart] = e;
            }
            for (i = 0; i < n; ++i)
            {
                output[i + outputStart] /= sum;
            }
        }

        public static void softmax_gpu(float[] input, int n, int offset, int groups, float temp,ref  float[] output, int inputStart = 0, int outputStart = 0)
        {
            int inputs = n;
            int batch = groups;
            var lp = CudaUtils.cuda_gridsize(batch);
            var tempOutput = Gpu.Default.Allocate(output);
            Gpu.Default.Launch(softmax_kernel, lp, inputs, offset, batch, input, temp, tempOutput, inputStart, outputStart);
            output = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }
    }
}