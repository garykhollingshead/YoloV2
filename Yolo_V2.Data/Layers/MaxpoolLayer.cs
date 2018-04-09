using System;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class MaxpoolLayer:Layer
    {
        public MaxpoolLayer(int batch, int height, int width, int numberOfChannels, int size, int stride, int padding)
        {
            LayerType = Layers.Maxpool;
            Batch = batch;
            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            Pad = padding;
            OutW = (width + 2 * padding) / stride;
            OutH = (height + 2 * padding) / stride;
            OutC = numberOfChannels;
            Outputs = OutH * OutW * OutC;
            Inputs = height * width * numberOfChannels;
            Size = size;
            Stride = stride;
            int outputSize = OutH * OutW * OutC * batch;
            Indexes = new int[outputSize];
            Output = new float[outputSize];
            Delta = new float[outputSize];

            IndexesGpu = new int[outputSize];
            OutputGpu = new float[outputSize];
            DeltaGpu = new float[outputSize];

            Console.Error.Write($"max\t\t{size} x {size} / {stride}\t{width} x{height} x{numberOfChannels}\t.\t{OutW} x{OutH} x{OutC}\n");
        }

        public void resize_maxpool_layer(int w, int h)
        {
            Height = h;
            Width = w;
            Inputs = h * w * NumberOfChannels;

            OutW = (w + 2 * Pad) / Stride;
            OutH = (h + 2 * Pad) / Stride;
            Outputs = OutW * OutH * NumberOfChannels;
            int outputSize = Outputs * Batch;

            Array.Resize(ref Indexes, outputSize);
            Array.Resize(ref Output, outputSize);
            Array.Resize(ref Delta, outputSize);

            IndexesGpu = new int[outputSize];
            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();

        }

        public override void Forward(ref NetworkState state)
        {
            int b, i, j, k, m, n;
            int wOffset = -Pad;
            int hOffset = -Pad;

            int h = OutH;
            int w = OutW;
            int c = NumberOfChannels;

            for (b = 0; b < Batch; ++b)
            {
                for (k = 0; k < c; ++k)
                {
                    for (i = 0; i < h; ++i)
                    {
                        for (j = 0; j < w; ++j)
                        {
                            int outIndex = j + w * (i + h * (k + c * b));
                            float max = float.MinValue;
                            int maxI = -1;
                            for (n = 0; n < Size; ++n)
                            {
                                for (m = 0; m < Size; ++m)
                                {
                                    int curH = hOffset + i * Stride + n;
                                    int curW = wOffset + j * Stride + m;
                                    int index = curW + Width * (curH + Height * (k + b * NumberOfChannels));
                                    bool valid = (curH >= 0 && curH < Height &&
                                                 curW >= 0 && curW < Width);
                                    float val = (valid) ? state.Input[index] : float.MinValue;
                                    maxI = (val > max) ? index : maxI;
                                    max = (val > max) ? val : max;
                                }
                            }
                            Output[outIndex] = max;
                            Indexes[outIndex] = maxI;
                        }
                    }
                }
            }
        }

        public override void Backward(ref NetworkState state)
        {
            int i;
            int h = OutH;
            int w = OutW;
            int c = NumberOfChannels;
            for (i = 0; i < h * w * c * Batch; ++i)
            {
                int index = Indexes[i];
                state.Delta[index] += Delta[i];
            }
        }

        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public  void forward_maxpool_layer_kernel(int n, int inH, int inW, int inC, int stride, int size, int pad, float[] input, float[] output, int[] indexes)
        {
            int h = (inH + 2 * pad) / stride;
            int w = (inW + 2 * pad) / stride;
            int c = inC;

            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int j = id % w;
            id /= w;
            int i = id % h;
            id /= h;
            int k = id % c;
            id /= c;
            int b = id;

            int wOffset = -pad;
            int hOffset = -pad;

            int outIndex = j + w * (i + h * (k + c * b));
            float max = float.NegativeInfinity;
            int maxI = -1;
            int l, m;
            for (l = 0; l < size; ++l)
            {
                for (m = 0; m < size; ++m)
                {
                    int curH = hOffset + i * stride + l;
                    int curW = wOffset + j * stride + m;
                    int index = curW + inW * (curH + inH * (k + b * inC));
                    bool valid = (curH >= 0 && curH < inH &&
                            curW >= 0 && curW < inW);
                    float val = (valid) ? input[index] : float.NegativeInfinity;
                    maxI = (val > max) ? index : maxI;
                    max = (val > max) ? val : max;
                }
            }
            output[outIndex] = max;
            indexes[outIndex] = maxI;
        }

        public  void backward_maxpool_layer_kernel(int n, int inH, int inW, int inC, int stride, int size, int pad, float[] delta, float[] prevDelta, int[] indexes)
        {
            int h = (inH + 2 * pad) / stride;
            int w = (inW + 2 * pad) / stride;
            int c = inC;
            int area = (size - 1) / stride;

            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int index = id;
            int j = id % inW;
            id /= inW;
            int i = id % inH;
            id /= inH;
            int k = id % inC;
            id /= inC;
            int b = id;

            int wOffset = -pad;
            int hOffset = -pad;

            float d = 0;
            int l, m;
            for (l = -area; l < area + 1; ++l)
            {
                for (m = -area; m < area + 1; ++m)
                {
                    int outW = (j - wOffset) / stride + m;
                    int outH = (i - hOffset) / stride + l;
                    int outIndex = outW + w * (outH + h * (k + c * b));
                    bool valid = (outW >= 0 && outW < w &&
                             outH >= 0 && outH < h);
                    d += (valid && indexes[outIndex] == index) ? delta[outIndex] : 0;
                }
            }
            prevDelta[index] += d;
        }

        [GpuManaged]
        public override void ForwardGpu(ref NetworkState state)
        {
            int h = OutH;
            int w = OutH;
            int c = NumberOfChannels;

            var n = h * w * c * Batch;

            var lp = CudaUtils.cuda_gridsize(n);
            var tempOutput = Gpu.Default.Allocate(OutputGpu);
            var tempIndexes = Gpu.Default.Allocate(IndexesGpu);
            Gpu.Default.Launch(forward_maxpool_layer_kernel, lp, n, Height, Width, NumberOfChannels, Stride, Size, Pad, state.Input, tempOutput, tempIndexes);
            OutputGpu = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
            IndexesGpu = Gpu.CopyToHost(tempIndexes);
            Gpu.Free(tempIndexes);
        }

        [GpuManaged]
        public override void BackwardGpu(ref NetworkState state)
        {
            var n = Height * Width * NumberOfChannels * Batch;

            var lp = CudaUtils.cuda_gridsize(n);
            var tempOutput = Gpu.Default.Allocate(state.Delta);
            Gpu.Default.Launch(backward_maxpool_layer_kernel, lp, n, Height, Width, NumberOfChannels, Stride, Size, Pad, DeltaGpu, tempOutput, IndexesGpu);
            state.Delta = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

    }
}