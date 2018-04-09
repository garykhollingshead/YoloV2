using System;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class DropoutLayer:Layer
    {
        public DropoutLayer(int batch, int inputs, float probability)
        {
            LayerType = Layers.Dropout;
            Probability = probability;
            Inputs = inputs;
            Outputs = inputs;
            Batch = batch;
            Rand = new float[inputs * batch];
            Scale = 1.0f / (1.0f - probability);

            RandGpu = (float[])Rand.Clone();

            Console.Error.Write($"dropout       p = {probability:F2}               {inputs}  .  {inputs}\n");
     
        }

        public override void Forward(ref NetworkState state)
        {
            int i;
            if (!state.Train) return;
            for (i = 0; i < Batch * Inputs; ++i)
            {
                float r = Utils.rand_uniform(0, 1);
                Rand[i] = r;
                if (r < Probability) state.Input[i] = 0;
                else state.Input[i] *= Scale;
            }
        }

        public override void Backward(ref NetworkState state)
        {
            int i;
            if (state.Delta.Length == 0) return;
            for (i = 0; i < Batch * Inputs; ++i)
            {
                float r = Rand[i];
                if (r < Probability) state.Delta[i] = 0;
                else state.Delta[i] *= Scale;
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
            if (!state.Train) return;
            int size = Inputs * Batch;
            CudaUtils.cuda_random(RandGpu, (ulong)size);

            var lp = CudaUtils.cuda_gridsize(size);
            var tempOutput = Gpu.Default.Allocate(state.Input);
            Gpu.Default.Launch(Yoloswag420Blazeit360Noscope, lp, tempOutput, size, RandGpu, Probability, Scale);
            state.Input = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            if (state.Delta.Length == 0) return;
            int size = Inputs * Batch;

            var lp = CudaUtils.cuda_gridsize(size);
            var tempOutput = Gpu.Default.Allocate(state.Delta);
            Gpu.Default.Launch(Yoloswag420Blazeit360Noscope, lp, tempOutput, size, RandGpu, Probability, Scale);
            state.Delta = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

        public void Yoloswag420Blazeit360Noscope(float[] input, int size, float[] rand, float prob, float scale)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id < size) input[id] = (rand[id] < prob) ? 0 : input[id] * scale;
        }

    }
}