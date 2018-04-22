using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class ReorgLayer:Layer
    {

        public ReorgLayer(int batch, int width, int height, int numberOfChannels, int stride, bool reverse)
        {
            LayerType = Layers.Reorg;
            Batch = batch;
            Stride = stride;
            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            if (reverse)
            {
                OutW = width * stride;
                OutH = height * stride;
                OutC = numberOfChannels / (stride * stride);
            }
            else
            {
                OutW = width / stride;
                OutH = height / stride;
                OutC = numberOfChannels * (stride * stride);
            }
            Reverse = reverse;
            Console.Error.Write($"reorg              /{stride}  {width} x{height} x{numberOfChannels}   .  {OutW} x{OutH} x{OutC}\n");
            Outputs = OutH * OutW * OutC;
            Inputs = height * width * numberOfChannels;
            int outputSize = OutH * OutW * OutC * batch;
            Output = new float[outputSize];
            Delta = new float[outputSize];

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
        }

        public void resize_reorg_layer( int w, int h)
        {
            int stride = Stride;
            int c = NumberOfChannels;

            Height = h;
            Width = w;

            if (Reverse)
            {
                OutW = w * stride;
                OutH = h * stride;
                OutC = c / (stride * stride);
            }
            else
            {
                OutW = w / stride;
                OutH = h / stride;
                OutC = c * (stride * stride);
            }

            Outputs = OutH * OutW * OutC;
            Inputs = Outputs;
            int outputSize = Outputs * Batch;

            Array.Resize(ref Output, outputSize);
            Array.Resize(ref Delta, outputSize);

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
        }

        public override void Forward(ref NetworkState state)
        {
            if (Reverse)
            {
                Blas.Reorg_cpu(state.Input, Width, Height, NumberOfChannels, Batch, Stride, 1, Output);
            }
            else
            {
                Blas.Reorg_cpu(state.Input, Width, Height, NumberOfChannels, Batch, Stride, 0, Output);
            }
        }

        public override void Backward(ref NetworkState state)
        {
            if (Reverse)
            {
                Blas.Reorg_cpu(Delta, Width, Height, NumberOfChannels, Batch, Stride, 0, state.Delta);
            }
            else
            {
                Blas.Reorg_cpu(Delta, Width, Height, NumberOfChannels, Batch, Stride, 1, state.Delta);
            }
        }

        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            return;
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            return;
            throw new NotImplementedException();
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            if (Reverse)
            {
                Blas.reorg_ongpu(state.Input, Width, Height, NumberOfChannels, Batch, Stride, 1, ref OutputGpu);
            }
            else
            {
                Blas.reorg_ongpu(state.Input, Width, Height, NumberOfChannels, Batch, Stride, 0, ref OutputGpu);
            }
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            if (Reverse)
            {
                Blas.reorg_ongpu(DeltaGpu, Width, Height, NumberOfChannels, Batch, Stride, 0, ref state.Delta);
            }
            else
            {
                Blas.reorg_ongpu(DeltaGpu, Width, Height, NumberOfChannels, Batch, Stride, 1, ref state.Delta);
            }
        }

    }
}