using System;
using Alea;
using Alea.CSharp;

namespace Yolo_V2.Data
{
    public static class Im2Col
    {
        public static float im2col_get_pixel(float[] im, int height, int width, int channels,
            int row, int col, int channel, int pad, int imStart = 0)
        {
            row -= pad;
            col -= pad;

            if (row < 0 || col < 0 ||
                row >= height || col >= width) return 0;
            return im[imStart + col + width * (row + height * channel)];
        }

        public static void im2col_cpu(float[] dataIm,
            int channels, int height, int width,
            int ksize, int stride, int pad, float[] dataCol, int dataImStart = 0, int dataColStart = 0)
        {
            int c, h, w;
            int heightCol = (height + 2 * pad - ksize) / stride + 1;
            int widthCol = (width + 2 * pad - ksize) / stride + 1;

            int channelsCol = channels * ksize * ksize;
            for (c = 0; c < channelsCol; ++c)
            {
                int wOffset = c % ksize;
                int hOffset = (c / ksize) % ksize;
                int cIm = c / ksize / ksize;
                for (h = 0; h < heightCol; ++h)
                {
                    for (w = 0; w < widthCol; ++w)
                    {
                        int imRow = hOffset + h * stride;
                        int imCol = wOffset + w * stride;
                        int colIndex = (c * heightCol + h) * widthCol + w;
                        dataCol[dataColStart + colIndex] = im2col_get_pixel(dataIm, height, width, channels,
                            imRow, imCol, cIm, pad, dataImStart);
                    }
                }
            }
        }

        public static void col2im_add_pixel(float[] im, int height, int width, int channels,
            int row, int col, int channel, int pad, float val, int imStart = 0)
        {
            row -= pad;
            col -= pad;

            if (row < 0 || col < 0 ||
                row >= height || col >= width) return;
            im[imStart + col + width * (row + height * channel)] += val;
        }

        public static void col2im_cpu(float[] dataCol,
            int channels, int height, int width,
            int ksize, int stride, int pad, float[] dataIm, int dataImStart = 0)
        {
            int c, h, w;
            int heightCol = (height + 2 * pad - ksize) / stride + 1;
            int widthCol = (width + 2 * pad - ksize) / stride + 1;

            int channelsCol = channels * ksize * ksize;
            for (c = 0; c < channelsCol; ++c)
            {
                int wOffset = c % ksize;
                int hOffset = (c / ksize) % ksize;
                int cIm = c / ksize / ksize;
                for (h = 0; h < heightCol; ++h)
                {
                    for (w = 0; w < widthCol; ++w)
                    {
                        int imRow = hOffset + h * stride;
                        int imCol = wOffset + w * stride;
                        int colIndex = (c * heightCol + h) * widthCol + w;
                        float val = dataCol[colIndex];
                        col2im_add_pixel(dataIm, height, width, channels,
                            imRow, imCol, cIm, pad, val, dataImStart);
                    }
                }
            }
        }

        private static void im2col_gpu_kernel(int n, float[] dataIm,
            int height, int width, int ksize,
            int pad,
            int stride,
            int heightCol, int widthCol,
            float[] dataCol)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            for (; index < n; index += blockDim.x * gridDim.x)
            {
                int wOut = index % widthCol;
                int hIndex = index / widthCol;
                int hOut = hIndex % heightCol;
                int channelIn = hIndex / heightCol;
                int channelOut = channelIn * ksize * ksize;
                int hIn = hOut * stride - pad;
                int wIn = wOut * stride - pad;

                int dataColPtr = (channelOut * heightCol + hOut) * widthCol + wOut;
                int dataImPtr = (channelIn * height + hIn) * width + wIn;

                for (int i = 0; i < ksize; ++i)
                {
                    for (int j = 0; j < ksize; ++j)
                    {
                        int h = hIn + i;
                        int w = wIn + j;
                        dataCol[dataColPtr] = (h >= 0 && w >= 0 && h < height && w < width)
                            ? dataIm[dataImPtr + i * width + j]
                            : 0;

                        dataColPtr += heightCol * widthCol;
                    }
                }
            }
        }

        [GpuManaged]
        public static void im2col_ongpu(float[] im,
                 int channels, int height, int width,
                 int ksize, int stride, int pad, float[] dataCol)
        {
            // We are going to launch channels * height_col * width_col kernels, each
            // kernel responsible for copying a single-channel grid.
            int heightCol = (height + 2 * pad - ksize) / stride + 1;
            int widthCol = (width + 2 * pad - ksize) / stride + 1;
            int numKernels = channels * heightCol * widthCol;
            var lp = new LaunchParam((numKernels + CudaUtils.BlockSize - 1) / CudaUtils.BlockSize, CudaUtils.BlockSize);
            Gpu.Default.Launch(im2col_gpu_kernel, lp, numKernels, im, height, width, ksize, pad,
                stride, heightCol, widthCol, dataCol);
        }

        private static void col2im_gpu_kernel(int n, float[] dataCol,
        int height, int width, int ksize,
        int pad,
        int stride,
        int heightCol, int widthCol,
        float[] dataIm, int imStart)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            for (; index + imStart < n; index += blockDim.x * gridDim.x)
            {
                float val = 0;
                int w = index % width + pad;
                int h = (index / width) % height + pad;
                int c = index / (width * height);
                // compute the start and end of the output
                int wColStart = (w < ksize) ? 0 : (w - ksize) / stride + 1;
                int wColEnd = Math.Min(w / stride + 1, widthCol);
                int hColStart = (h < ksize) ? 0 : (h - ksize) / stride + 1;
                int hColEnd = Math.Min(h / stride + 1, heightCol);
                // equivalent implementation
                int offset =
                    (c * ksize * ksize + h * ksize + w) * heightCol * widthCol;
                int coeffHCol = (1 - stride * ksize * heightCol) * widthCol;
                int coeffWCol = (1 - stride * heightCol * widthCol);
                for (int hCol = hColStart; hCol < hColEnd; ++hCol)
                {
                    for (int wCol = wColStart; wCol < wColEnd; ++wCol)
                    {
                        val += dataCol[offset + hCol * coeffHCol + wCol * coeffWCol];
                    }
                }
                dataIm[index + imStart] += val;
            }
        }

        [GpuManaged]
        public static void col2im_ongpu(float[] dataCol,
            int channels, int height, int width,
            int ksize, int stride, int pad, float[] dataIm, int imStart = 0)
        {
            // We are going to launch channels * height_col * width_col kernels, each
            // kernel responsible for copying a single-channel grid.
            int heightCol = (height + 2 * pad - ksize) / stride + 1;
            int widthCol = (width + 2 * pad - ksize) / stride + 1;
            int numKernels = channels * height * width;
            var lp = new LaunchParam((numKernels + CudaUtils.BlockSize - 1) / CudaUtils.BlockSize, CudaUtils.BlockSize);
            Gpu.Default.Launch(col2im_gpu_kernel, lp, numKernels, dataCol, height, width, ksize, pad, stride,
                heightCol, widthCol, dataIm, imStart);
        }
    }
}