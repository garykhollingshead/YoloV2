using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class RouteLayer: Layer
    {
        public RouteLayer(int batch, int n, int[] inputLayers, int[] inputSizes)
        {
            Console.Error.Write($"route ");
            LayerType = Layers.Route;
            Batch = batch;
            N = n;
            InputLayers = inputLayers;
            InputSizes = inputSizes;
            int i;
            int outputs = 0;
            for (i = 0; i < n; ++i)
            {
                Console.Error.Write($" {inputLayers[i]}");
                outputs += inputSizes[i];
            }
            Console.Error.Write($"\n");
            Outputs = outputs;
            Inputs = outputs;
            Delta = new float[outputs * batch];
            Output = new float[outputs * batch];

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();
        }

        public void resize_route_layer(ref Network net)
        {
            int i;
            Layer first = net.Layers[InputLayers[0]];
            OutW = first.OutW;
            OutH = first.OutH;
            OutC = first.OutC;
            Outputs = first.Outputs;
            InputSizes[0] = first.Outputs;
            for (i = 1; i < N; ++i)
            {
                int index = InputLayers[i];
                Layer next = net.Layers[index];
                Outputs += next.Outputs;
                InputSizes[i] = next.Outputs;
                if (next.OutW == first.OutW && next.OutH == first.OutH)
                {
                    OutC += next.OutC;
                }
                else
                {
                    Console.Write($"{next.OutW} {next.OutH}, {first.OutW} {first.OutH}\n");
                    OutH = OutW = OutC = 0;
                }
            }
            Inputs = Outputs;
            Array.Resize(ref Delta, Outputs * Batch);
            Array.Resize(ref Output, Outputs * Batch);

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
        }

        public override void Forward(ref NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < N; ++i)
            {
                int index = InputLayers[i];
                float[] input = state.Net.Layers[index].Output;
                int inputSize = InputSizes[i];
                for (j = 0; j < Batch; ++j)
                {
                    Blas.Copy_cpu(inputSize, input, Output, j * inputSize, offset + j * Outputs);
                }
                offset += inputSize;
            }
        }

        public override void Backward(ref NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < N; ++i)
            {
                int index = InputLayers[i];
                float[] delta = state.Net.Layers[index].Delta;
                int inputSize = InputSizes[i];
                for (j = 0; j < Batch; ++j)
                {
                    Blas.Axpy_cpu(inputSize, 1, Delta, delta, offset + j * Outputs, j * inputSize);
                }
                offset += inputSize;
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

        public override void ForwardGpu(ref NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < N; ++i)
            {
                int index = InputLayers[i];
                float[] input = state.Net.Layers[index].OutputGpu;
                int inputSize = InputSizes[i];
                for (j = 0; j < Batch; ++j)
                {
                    Blas.copy_ongpu(inputSize, input, OutputGpu, j * inputSize, offset + j * Outputs);
                }
                offset += inputSize;
            }
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < N; ++i)
            {
                int index = InputLayers[i];
                float[] delta = state.Net.Layers[index].DeltaGpu;
                int inputSize = InputSizes[i];
                for (j = 0; j < Batch; ++j)
                {
                    Blas.axpy_ongpu(inputSize, 1, DeltaGpu, delta, offset + j * Outputs, j * inputSize);
                }
                offset += inputSize;
            }
        }

    }
}