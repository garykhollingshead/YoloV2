using Alea;
using Alea.CSharp;
using System;

namespace Yolo_V2.Data
{
    public static class Im2Col
    {
        public static float im2col_get_pixel(float[] im, int height, int width, int channels,
            int row, int col, int channel, int pad)
        {
            row -= pad;
            col -= pad;

            if (row < 0 || col < 0 ||
                row >= height || col >= width) return 0;
            return im[col + width * (row + height * channel)];
        }

        public static void im2col_cpu(float[] data_im,
            int channels, int height, int width,
            int ksize, int stride, int pad, float[] data_col)
        {
            int c, h, w;
            int height_col = (height + 2 * pad - ksize) / stride + 1;
            int width_col = (width + 2 * pad - ksize) / stride + 1;

            int channels_col = channels * ksize * ksize;
            for (c = 0; c < channels_col; ++c)
            {
                int w_offset = c % ksize;
                int h_offset = (c / ksize) % ksize;
                int c_im = c / ksize / ksize;
                for (h = 0; h < height_col; ++h)
                {
                    for (w = 0; w < width_col; ++w)
                    {
                        int im_row = h_offset + h * stride;
                        int im_col = w_offset + w * stride;
                        int col_index = (c * height_col + h) * width_col + w;
                        data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                            im_row, im_col, c_im, pad);
                    }
                }
            }
        }

        public static void col2im_add_pixel(float[] im, int height, int width, int channels,
            int row, int col, int channel, int pad, float val)
        {
            row -= pad;
            col -= pad;

            if (row < 0 || col < 0 ||
                row >= height || col >= width) return;
            im[col + width * (row + height * channel)] += val;
        }

        public static void col2im_cpu(float[] data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, float[] data_im)
        {
            int c, h, w;
            int height_col = (height + 2 * pad - ksize) / stride + 1;
            int width_col = (width + 2 * pad - ksize) / stride + 1;

            int channels_col = channels * ksize * ksize;
            for (c = 0; c < channels_col; ++c)
            {
                int w_offset = c % ksize;
                int h_offset = (c / ksize) % ksize;
                int c_im = c / ksize / ksize;
                for (h = 0; h < height_col; ++h)
                {
                    for (w = 0; w < width_col; ++w)
                    {
                        int im_row = h_offset + h * stride;
                        int im_col = w_offset + w * stride;
                        int col_index = (c * height_col + h) * width_col + w;
                        float val = data_col[col_index];
                        col2im_add_pixel(data_im, height, width, channels,
                            im_row, im_col, c_im, pad, val);
                    }
                }
            }
        }

        private static void im2col_gpu_kernel(int n, float[] data_im,
            int height, int width, int ksize,
            int pad,
            int stride,
            int height_col, int width_col,
            float[] data_col)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            for (; index < n; index += blockDim.x * gridDim.x)
            {
                int w_out = index % width_col;
                int h_index = index / width_col;
                int h_out = h_index % height_col;
                int channel_in = h_index / height_col;
                int channel_out = channel_in * ksize * ksize;
                int h_in = h_out * stride - pad;
                int w_in = w_out * stride - pad;

                int data_col_ptr = (channel_out * height_col + h_out) * width_col + w_out;
                int data_im_ptr = (channel_in * height + h_in) * width + w_in;

                for (int i = 0; i < ksize; ++i)
                {
                    for (int j = 0; j < ksize; ++j)
                    {
                        int h = h_in + i;
                        int w = w_in + j;
                        data_col[data_col_ptr] = (h >= 0 && w >= 0 && h < height && w < width)
                            ? data_im[data_im_ptr + i * width + j]
                            : 0;

                        data_col_ptr += height_col * width_col;
                    }
                }
            }
        }

        [GpuManaged]
        public static void im2col_ongpu(float[] im,
                 int channels, int height, int width,
                 int ksize, int stride, int pad, float[] data_col)
        {
            // We are going to launch channels * height_col * width_col kernels, each
            // kernel responsible for copying a single-channel grid.
            int height_col = (height + 2 * pad - ksize) / stride + 1;
            int width_col = (width + 2 * pad - ksize) / stride + 1;
            int num_kernels = channels * height_col * width_col;
            var lp = new LaunchParam((num_kernels + CudaUtils.BlockSize - 1) / CudaUtils.BlockSize, CudaUtils.BlockSize);
            Gpu.Default.Launch(im2col_gpu_kernel, lp, num_kernels, im, height, width, ksize, pad,
                stride, height_col, width_col, data_col);
        }

        private static void col2im_gpu_kernel(int n, float[] data_col,
        int height, int width, int ksize,
        int pad,
        int stride,
        int height_col, int width_col,
        float[] data_im)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            for (; index < n; index += blockDim.x * gridDim.x)
            {
                float val = 0;
                int w = index % width + pad;
                int h = (index / width) % height + pad;
                int c = index / (width * height);
                // compute the start and end of the output
                int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
                int w_col_end = Math.Min(w / stride + 1, width_col);
                int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
                int h_col_end = Math.Min(h / stride + 1, height_col);
                // equivalent implementation
                int offset =
                    (c * ksize * ksize + h * ksize + w) * height_col * width_col;
                int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
                int coeff_w_col = (1 - stride * height_col * width_col);
                for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
                {
                    for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
                    {
                        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
                    }
                }
                data_im[index] += val;
            }
        }

        [GpuManaged]
        public static void col2im_ongpu(float[] data_col,
            int channels, int height, int width,
            int ksize, int stride, int pad, float[] data_im)
        {
            // We are going to launch channels * height_col * width_col kernels, each
            // kernel responsible for copying a single-channel grid.
            int height_col = (height + 2 * pad - ksize) / stride + 1;
            int width_col = (width + 2 * pad - ksize) / stride + 1;
            int num_kernels = channels * height * width;
            var lp = new LaunchParam((num_kernels + CudaUtils.BlockSize - 1) / CudaUtils.BlockSize, CudaUtils.BlockSize);
            Gpu.Default.Launch(col2im_gpu_kernel, lp, num_kernels, data_col, height, width, ksize, pad, stride,
                height_col, width_col, data_im);
        }
    }
}