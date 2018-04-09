using System;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class AvgpoolLayer: Layer
    {
        public AvgpoolLayer(int batch, int width, int height, int numberOfChannels)
        {
            Console.Error.Write($"avg                     {width} x{height} x{numberOfChannels}   .  {numberOfChannels}\n");
            LayerType = Layers.Avgpool;
            Batch = batch;
            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            OutW = 1;
            OutH = 1;
            OutC = numberOfChannels;
            Outputs = OutC;
            Inputs = height * width * numberOfChannels;
            int outputSize = Outputs * batch;
            Output = new float[outputSize];
            Delta = new float[outputSize];
            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
        }

        public void resize_avgpool_layer(int w, int h)
        {
            Width = w;
            Height = h;
            Inputs = h * w * NumberOfChannels;
        }

        public override void Forward(ref NetworkState state)
        {
            int b, i, k;

            for (b = 0; b < Batch; ++b)
            {
                for (k = 0; k < NumberOfChannels; ++k)
                {
                    int outIndex = k + b * NumberOfChannels;
                    Output[outIndex] = 0;
                    for (i = 0; i < Height * Width; ++i)
                    {
                        int inIndex = i + Height * Width * (k + b * NumberOfChannels);
                        Output[outIndex] += state.Input[inIndex];
                    }
                    Output[outIndex] /= Height * Width;
                }
            }
        }

        public override void Backward(ref NetworkState state)
        {
            int b, i, k;

            for (b = 0; b < Batch; ++b)
            {
                for (k = 0; k < NumberOfChannels; ++k)
                {
                    int outIndex = k + b * NumberOfChannels;
                    for (i = 0; i < Height * Width; ++i)
                    {
                        int inIndex = i + Height * Width * (k + b * NumberOfChannels);
                        state.Delta[inIndex] += Delta[outIndex] / (Height * Width);
                    }
                }
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

        public void forward_avgpool_layer_kernel(int n, int w, int h, int c, float[] input, float[] output)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int k = id % c;
            id /= c;
            int b = id;

            int i;
            int outIndex = (k + c * b);
            output[outIndex] = 0;
            for (i = 0; i < w * h; ++i)
            {
                int inIndex = i + h * w * (k + b * c);
                output[outIndex] += input[inIndex];
            }
            output[outIndex] /= w * h;
        }

        public void backward_avgpool_layer_kernel(int n, int w, int h, int c, float[] inDelta, float[] outDelta)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int k = id % c;
            id /= c;
            int b = id;

            int i;
            int outIndex = (k + c * b);
            for (i = 0; i < w * h; ++i)
            {
                int inIndex = i + h * w * (k + b * c);
                inDelta[inIndex] += outDelta[outIndex] / (w * h);
            }
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            int n = NumberOfChannels * Batch;

            var lp = CudaUtils.cuda_gridsize(n);
            var tempOutput = Gpu.Default.Allocate(OutputGpu);
            Gpu.Default.Launch(forward_avgpool_layer_kernel, lp, n, Width, Height, NumberOfChannels, state.Input, tempOutput);
            OutputGpu = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            int n = NumberOfChannels * Batch;

            var lp = CudaUtils.cuda_gridsize(n);
            var tempOutput = Gpu.Default.Allocate(state.Delta);
            Gpu.Default.Launch(backward_avgpool_layer_kernel, lp, n, Width, Height, NumberOfChannels, tempOutput, DeltaGpu);
            state.Delta = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

    }
}