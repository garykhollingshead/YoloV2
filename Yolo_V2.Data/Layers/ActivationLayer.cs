using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class ActivationLayer : Layer
    {
        public ActivationLayer(int batch, int inputs, Activation activation)
        {
            LayerType = Layers.Active;

            Inputs = inputs;
            Outputs = inputs;
            Batch = batch;

            Output = new float[batch];
            Delta = new float[batch];

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
            Activation = activation;
            Console.Error.Write($"Activation Layer: {inputs} inputs\n");
        }

        public override void Forward(ref NetworkState state)
        {
            Blas.Copy_cpu(Outputs * Batch, state.Input, Output);
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }

        public override void Backward(ref NetworkState state)
        {
            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);
            Blas.Copy_cpu(Outputs * Batch, Delta, state.Delta);
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
            ActivationsHelper.activate_array_ongpu(ref OutputGpu, Outputs * Batch, Activation);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, ref DeltaGpu);
            Blas.copy_ongpu(Outputs * Batch, DeltaGpu, ref state.Delta);
        }

    }
}