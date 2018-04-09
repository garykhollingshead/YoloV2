using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class NormalizationLayer:Layer
    {
        public NormalizationLayer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
        {
            Console.Error.Write($"Local Response Normalization Layer: %d x %d x %d Image, %d size\n", w, h, c, size);
            LayerType = Layers.Normalization;
            Batch = batch;
            Height = OutH = h;
            Width = OutW = w;
            NumberOfChannels = OutC = c;
            Kappa = kappa;
            Size = size;
            Alpha = alpha;
            Beta = beta;
            Output = new float[h * w * c * batch];
            Delta = new float[h * w * c * batch];
            Squared = new float[h * w * c * batch];
            Norms = new float[h * w * c * batch];
            Inputs = w * h * c;
            Outputs = Inputs;
            
            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
            SquaredGpu = (float[])Squared.Clone();
            NormsGpu = (float[])Norms.Clone();
        }

        public void resize_normalization_layer(int w, int h)
        {
            int c = NumberOfChannels;
            int batch = Batch;
            Height = h;
            Width = w;
            OutH = h;
            OutW = w;
            Inputs = w * h * c;
            Outputs = Inputs;
            Array.Resize(ref Output, h * w * c * batch);
            Array.Resize(ref Delta, h * w * c * batch);
            Array.Resize(ref Squared, h * w * c * batch);
            Array.Resize(ref Norms, h * w * c * batch);

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
            SquaredGpu = (float[])Squared.Clone();
            NormsGpu = (float[])Norms.Clone();

        }

        public override void Forward(ref NetworkState state)
        {
            int k, b;
            int w = Width;
            int h = Height;
            int c = NumberOfChannels;
            Blas.Scal_cpu(w * h * c * Batch, 0, Squared, 1);

            for (b = 0; b < Batch; ++b)
            {
                var index = w * h * c * b;
                float[] squared = new float[Squared.Length - index];
                float[] norms = new float[Norms.Length - index];
                float[] input = new float[state.Input.Length - index];
                Array.Copy(Squared, index, squared, 0, squared.Length);
                Array.Copy(Norms, index, norms, 0, norms.Length);
                Array.Copy(state.Input, index, input, 0, input.Length);
                Blas.Pow_cpu(w * h * c, 2, input, squared);

                Blas.Const_cpu(w * h, Kappa, norms, 1);
                for (k = 0; k < Size / 2; ++k)
                {
                    Blas.Axpy_cpu(w * h, Alpha, squared, norms, 0, w * h * k);
                }

                for (k = 1; k < NumberOfChannels; ++k)
                {
                    Blas.Copy_cpu(w * h, norms, norms, w * h * (k - 1), w * h * k);
                    int prev = k - ((Size - 1) / 2) - 1;
                    int next = k + (Size / 2);
                    if (prev >= 0) Blas.Axpy_cpu(w * h, -Alpha, squared, norms, w * h * prev, w * h * k);
                    if (next < NumberOfChannels) Blas.Axpy_cpu(w * h, Alpha, squared, norms, w * h * next, w * h * k);
                }
                Array.Copy(squared, 0, Squared, index, squared.Length);
                Array.Copy(norms, 0, Norms, index, norms.Length);
            }
            Blas.Pow_cpu(w * h * c * Batch, -Beta, Norms, Output);
            Blas.Mul_cpu(w * h * c * Batch, state.Input, 1, Output, 1);
        }

        public override void Backward(ref NetworkState state)
        {
            // TODO This is approximate ;-)
            // Also this should add in to delta instead of overwritting.

            int w = Width;
            int h = Height;
            int c = NumberOfChannels;
            Blas.Pow_cpu(w * h * c * Batch, -Beta, Norms, state.Delta);
            Blas.Mul_cpu(w * h * c * Batch, Delta, 1, state.Delta, 1);
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
            int k, b;
            int w = Width;
            int h = Height;
            int c = NumberOfChannels;
            Blas.scal_ongpu(w * h * c * Batch, 0, ref SquaredGpu, 1);

            for (b = 0; b < Batch; ++b)
            {
                var index = w * h * c * b;
                float[] squared = new float[SquaredGpu.Length - index];
                float[] norms = new float[NormsGpu.Length - index];
                float[] input = new float[state.Input.Length - index];
                Array.Copy(Squared, index, squared, 0, squared.Length);
                Array.Copy(Norms, index, norms, 0, norms.Length);
                Array.Copy(state.Input, index, input, 0, input.Length);

                Blas.pow_ongpu(w * h * c, 2, input, ref squared);

                Blas.const_ongpu(w * h, Kappa, ref norms, 1);
                for (k = 0; k < Size / 2; ++k)
                {
                    Blas.axpy_ongpu(w * h, Alpha, squared, norms, w * h * k);
                }

                for (k = 1; k < NumberOfChannels; ++k)
                {
                    Blas.copy_ongpu(w * h, norms, norms, w * h * (k - 1), w * h * k);
                    int prev = k - ((Size - 1) / 2) - 1;
                    int next = k + (Size / 2);
                    if (prev >= 0) Blas.axpy_ongpu(w * h, -Alpha, squared, norms, w * h * prev, w * h * k);
                    if (next < NumberOfChannels) Blas.axpy_ongpu(w * h, Alpha, squared, norms, w * h * next, w * h * k);
                }
                Array.Copy(squared, 0, Squared, index, squared.Length);
                Array.Copy(norms, 0, Norms, index, norms.Length);
            }
            Blas.pow_ongpu(w * h * c * Batch, -Beta, NormsGpu, ref OutputGpu);
            Blas.mul_ongpu(w * h * c * Batch, state.Input, 1, ref OutputGpu, 1);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            // TODO This is approximate ;-)

            int w = Width;
            int h = Height;
            int c = NumberOfChannels;
            Blas.pow_ongpu(w * h * c * Batch, -Beta, NormsGpu, ref state.Delta);
            Blas.mul_ongpu(w * h * c * Batch, DeltaGpu, 1, ref state.Delta, 1);
        }

    }
}