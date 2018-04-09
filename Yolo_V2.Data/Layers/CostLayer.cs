using System;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class CostLayer:Layer
    {
        public CostLayer(int batch, int inputs, CostType costType, float scale)
        {
            Console.Error.Write($"cost                                           {inputs}\n");
            LayerType = Layers.Cost;

            Scale = scale;
            Batch = batch;
            Inputs = inputs;
            Outputs = inputs;
            CostType = costType;
            Delta = new float[inputs * batch];
            Output = new float[inputs * batch];
            Cost = 0;

            DeltaGpu = (float[])Output.Clone();
            OutputGpu = (float[])Delta.Clone();
        }

        public void resize_cost_layer(int inputs)
        {
            Inputs = inputs;
            Outputs = inputs;
            Array.Resize(ref Delta, inputs * Batch);
            Array.Resize(ref Output, inputs * Batch);

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

        }

        public override void Forward(ref NetworkState state)
        {
            if (!state.Truth.Any()) return;
            if (CostType == CostType.Masked)
            {
                int i;
                for (i = 0; i < Batch * Inputs; ++i)
                {
                    if (state.Truth[i] == Utils.SecretNum) state.Input[i] = Utils.SecretNum;
                }
            }
            if (CostType == CostType.Smooth)
            {
                Blas.Smooth_l1_cpu(Batch * Inputs, state.Input, state.Truth, Delta, Output);
            }
            else
            {
                Blas.L2_cpu(Batch * Inputs, state.Input, state.Truth, Delta, Output);
            }
            Cost = Output.Sum();
        }

        public override void Backward(ref NetworkState state)
        {
            Blas.Axpy_cpu(Batch * Inputs, Scale, Delta, state.Delta);
        }

        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public int float_abs_compare(float a, float b)
        {
            a = Math.Abs(a);
            b = Math.Abs(b);
            int ag = a > b ? 1 : 0;
            int bg = a < b ? 1 : 0;
            return ag - bg;
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            if (!state.Truth.Any()) return;
            if (CostType == CostType.Masked)
            {
                Blas.mask_ongpu(Batch * Inputs, ref state.Input, Utils.SecretNum, state.Truth);
            }

            if (CostType == CostType.Smooth)
            {
                Blas.smooth_l1_gpu(Batch * Inputs, state.Input, state.Truth, ref DeltaGpu, ref OutputGpu);
            }
            else
            {
                Blas.l2_gpu(Batch * Inputs, state.Input, state.Truth, DeltaGpu, OutputGpu);
            }

            if (Ratio != 0)
            {
                Array.Copy(DeltaGpu, Delta, Batch * Inputs);
                Array.Sort(Delta, float_abs_compare);
                int n = (int)((1 - Ratio) * Batch * Inputs);
                //float thresh = Delta[n];
                float thresh = 0;
                Console.Write($"{thresh}\n");
                Blas.supp_ongpu(Batch * Inputs, thresh, ref DeltaGpu, 1);
            }

            Array.Copy(OutputGpu, Output, Batch * Inputs);
            Cost = Output.Sum();
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            Blas.axpy_ongpu(Batch * Inputs, Scale, DeltaGpu, state.Delta);
        }

    }
}