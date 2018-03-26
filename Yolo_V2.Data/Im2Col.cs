using System.Collections.Generic;
using Alea;
using Alea.CSharp;

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
            var lp = new LaunchParam((num_kernels + Cuda.BlockSize - 1) / Cuda.BlockSize, Cuda.BlockSize);
            Gpu.Default.Launch(im2col_gpu_kernel, lp, num_kernels, im, height, width, ksize, pad,
                stride, height_col, width_col, data_col);
        }

    }
}