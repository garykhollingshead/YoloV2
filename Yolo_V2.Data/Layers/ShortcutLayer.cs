using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class ShortcutLayer : Layer
    {

        public ShortcutLayer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
        {
            Console.Error.Write($"Shortcut Layer: {index}\n");
            LayerType = Layers.Shortcut;
            Batch = batch;
            Width = w2;
            Height = h2;
            NumberOfChannels = c2;
            OutW = w;
            OutH = h;
            OutC = c;
            Outputs = w * h * c;
            Inputs = Outputs;

            Index = index;

            Delta = new float[Outputs * batch];
            Output = new float[Outputs * batch];

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();
        }

        public override void Forward(ref NetworkState state)
        {
            Blas.Copy_cpu(Outputs * Batch, state.Input, Output);
            Blas.Shortcut_cpu(Batch, Width, Height, NumberOfChannels, state.Net.Layers[Index].Output, OutW, OutH, OutC, Output);
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }

        public override void Backward(ref NetworkState state)
        {
            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);
            Blas.Axpy_cpu(Outputs * Batch, 1, Delta, state.Delta);
            Blas.Shortcut_cpu(Batch, OutW, OutH, OutC, Delta, Width, Height, NumberOfChannels, state.Net.Layers[Index].Delta);
        }

        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            Blas.copy_ongpu(Outputs * Batch, state.Input, ref OutputGpu);
            Blas.shortcut_gpu(Batch, Width, Height, NumberOfChannels, state.Net.Layers[Index].OutputGpu, OutW, OutH, OutC, ref OutputGpu);
            ActivationsHelper.activate_array_ongpu(ref OutputGpu, Outputs * Batch, Activation);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, ref DeltaGpu);
            Blas.axpy_ongpu(Outputs * Batch, 1, DeltaGpu, state.Delta);
            Blas.shortcut_gpu(Batch, OutW, OutH, OutC, DeltaGpu, Width, Height, NumberOfChannels, ref state.Net.Layers[Index].DeltaGpu);
        }

    }
}