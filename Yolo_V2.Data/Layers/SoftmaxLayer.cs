using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class SoftmaxLayer:Layer
    {

        public SoftmaxLayer(int batch, int inputs, int groups)
        {
            Console.Error.Write($"softmax                                        {inputs}\n");
            LayerType = Layers.Softmax;
            Batch = batch;
            Groups = groups;
            Inputs = inputs;
            Outputs = inputs;
            Output = new float[inputs * batch];
            Delta = new float[inputs * batch];

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
        }

        public void softmax_tree(float[] input, int batch, int inputs, float temp, Tree hierarchy, float[] output)
        {
            int b;
            for (b = 0; b < batch; ++b)
            {
                int i;
                int count = 0;
                for (i = 0; i < hierarchy.Groups; ++i)
                {
                    int groupSize = hierarchy.GroupSize[i];
                    Blas.Softmax(input, groupSize, temp, output, b * inputs + count, b * inputs + count);
                    count += groupSize;
                }
            }
        }

        public override void Forward(ref NetworkState state)
        {
            int b;
            int inputs = Inputs / Groups;
            int batch = Batch * Groups;
            if (SoftmaxTree != null)
            {
                softmax_tree(state.Input, batch, inputs, Temperature, SoftmaxTree, Output);
            }
            else
            {
                for (b = 0; b < batch; ++b)
                {
                    Blas.Softmax(state.Input, inputs, Temperature, Output, b * inputs, b * inputs);
                }
            }
        }

        public override void Backward(ref NetworkState state)
        {
            int i;
            for (i = 0; i < Inputs * Batch; ++i)
            {
                state.Delta[i] += Delta[i];
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
            int inputs = Inputs / Groups;
            int batch = Batch * Groups;
            if (SoftmaxTree != null)
            {
                int i;
                int count = 0;
                for (i = 0; i < SoftmaxTree.Groups; ++i)
                {
                    int groupSize = SoftmaxTree.GroupSize[i];
                    Blas.softmax_gpu(state.Input, groupSize, inputs, batch, Temperature, ref OutputGpu, count, count);
                    count += groupSize;
                }
            }
            else
            {
                Blas.softmax_gpu(state.Input, inputs, inputs, batch, Temperature, ref OutputGpu);
            }
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            Blas.axpy_ongpu(Batch * Inputs, 1, DeltaGpu, state.Delta);
        }

    }
}