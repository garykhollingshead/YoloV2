using System;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class ConnectedLayer : Layer
    {
        public ConnectedLayer(int batch, int inputs, int outputs, Activation activation, bool batchNormalize)
        {
            int i;
            LayerType = Layers.Connected;

            Inputs = inputs;
            Outputs = outputs;
            Batch = batch;
            BatchNormalize = batchNormalize;
            Height = 1;
            Width = 1;
            NumberOfChannels = inputs;
            OutH = 1;
            OutW = 1;
            OutC = outputs;

            Output = new float[batch * outputs];
            Delta = new float[batch * outputs];

            WeightUpdates = new float[inputs * outputs];
            BiasUpdates = new float[outputs];

            WeightsComplete = new float[outputs * inputs];
            BiasesComplete = new float[outputs];
            WeightsIndex = BiasesIndex = 0;


            float scale = (float)Math.Sqrt(2.0 / inputs);
            for (i = 0; i < outputs * inputs; ++i)
            {
                WeightsComplete[i] = scale * Utils.rand_uniform(-1, 1);
            }

            if (batchNormalize)
            {
                Scales = new float[outputs];
                ScaleUpdates = new float[outputs];
                for (i = 0; i < outputs; ++i)
                {
                    Scales[i] = 1;
                }

                Mean = new float[outputs];
                MeanDelta = new float[outputs];
                Variance = new float[outputs];
                VarianceDelta = new float[outputs];

                RollingMean = new float[outputs];
                RollingVariance = new float[outputs];

                X = new float[batch * outputs];
                XNorm = new float[batch * outputs];
            }


            WeightsGpu = (float[])WeightsComplete.Clone();
            BiasesGpu = (float[])BiasesComplete.Clone();

            WeightUpdatesGpu = (float[])WeightUpdates.Clone();
            BiasUpdatesGpu = (float[])BiasUpdates.Clone();

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
            if (batchNormalize)
            {
                ScalesGpu = (float[])Scales.Clone();
                ScaleUpdatesGpu = (float[])ScaleUpdates.Clone();

                MeanGpu = (float[])Mean.Clone();
                VarianceGpu = (float[])Variance.Clone();

                RollingMeanGpu = (float[])Mean.Clone();
                RollingVarianceGpu = (float[])Variance.Clone();

                MeanDeltaGpu = (float[])Mean.Clone();
                VarianceDeltaGpu = (float[])Variance.Clone();

                XGpu = (float[])Output.Clone();
                XNormGpu = (float[])Output.Clone();
            }
            Activation = activation;
            Console.Error.Write($"connected                            {inputs}  .  {outputs}\n");

        }

        public override void Update(ref int batch, ref float learningRate, ref float momentum, ref float decay)
        {
            Blas.Axpy_cpu(Outputs, learningRate / batch, BiasUpdates, BiasesComplete, 0, BiasesIndex);
            Blas.Scal_cpu(Outputs, momentum, BiasUpdates, 1);

            if (BatchNormalize)
            {
                Blas.Axpy_cpu(Outputs, learningRate / batch, ScaleUpdates, Scales);
                Blas.Scal_cpu(Outputs, momentum, ScaleUpdates, 1);
            }

            Blas.Axpy_cpu(Inputs * Outputs, -decay * batch, WeightsComplete, WeightUpdates, WeightsIndex);
            Blas.Axpy_cpu(Inputs * Outputs, learningRate / batch, WeightUpdates, WeightsComplete, 0, WeightsIndex);
            Blas.Scal_cpu(Inputs * Outputs, momentum, WeightUpdates, 1);
        }

        public override void Forward(ref NetworkState state)
        {
            int i;
            Blas.Fill_cpu(Outputs * Batch, 0, Output, 1);
            int m = Batch;
            int k = Inputs;
            int n = Outputs;
            float[] a = state.Input;
            float[] b = new float[WeightsComplete.Length - WeightsIndex];
            Array.Copy(WeightsComplete, WeightsIndex, b, 0, b.Length);
            float[] c = Output;
            GemmUtils.Gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
            if (BatchNormalize)
            {
                if (state.Train)
                {
                    Blas.Mean_cpu(Output, Batch, Outputs, 1, Mean);
                    Blas.Variance_cpu(Output, Mean, Batch, Outputs, 1, Variance);

                    Blas.Scal_cpu(Outputs, .95f, RollingMean, 1);
                    Blas.Axpy_cpu(Outputs, .05f, Mean, RollingMean);
                    Blas.Scal_cpu(Outputs, .95f, RollingVariance, 1);
                    Blas.Axpy_cpu(Outputs, .05f, Variance, RollingVariance);

                    Blas.Copy_cpu(Outputs * Batch, Output, X);
                    Blas.Normalize_cpu(Output, Mean, Variance, Batch, Outputs, 1);
                    Blas.Copy_cpu(Outputs * Batch, Output, XNorm);
                }
                else
                {
                    Blas.Normalize_cpu(Output, RollingMean, RollingVariance, Batch, Outputs, 1);
                }
                scale_bias(Output, Scales, Batch, Outputs, 1);
            }
            for (i = 0; i < Batch; ++i)
            {
                Blas.Axpy_cpu(Outputs, 1, BiasesComplete, Output, BiasesIndex, i * Outputs);
            }
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }

        public override void Backward(ref NetworkState state)
        {
            int i;
            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);
            for (i = 0; i < Batch; ++i)
            {
                Blas.Axpy_cpu(Outputs, 1, Delta, BiasUpdates, i * Outputs);
            }
            if (BatchNormalize)
            {
                backward_scale_cpu(XNorm, Delta, Batch, Outputs, 1, ScaleUpdates);

                scale_bias(Delta, Scales, Batch, Outputs, 1);

                mean_delta_cpu(Delta, Variance, Batch, Outputs, 1, MeanDelta);
                variance_delta_cpu(X, Delta, Mean, Variance, Batch, Outputs, 1, VarianceDelta);
                normalize_delta_cpu(X, Mean, Variance, MeanDelta, VarianceDelta, Batch, Outputs, 1, Delta);
            }

            int m = Outputs;
            int k = Batch;
            int n = Inputs;
            GemmUtils.Gemm(1, 0, m, n, k, 1, Delta, m, state.Input, n, 1, WeightUpdates, n);

            m = Batch;
            k = Outputs;
            n = Inputs;


            if (state.Delta.Length != 0)
            {
                var b = new float[WeightsComplete.Length - WeightsIndex];
                Array.Copy(WeightsComplete, WeightsIndex, b, 0, b.Length);
                GemmUtils.Gemm(0, 0, m, n, k, 1, Delta, k, b, n, 1, state.Delta, n);
            }
        }

        public override void UpdateGpu(ref int batch, ref float learningRate, ref float momentum, ref float decay)
        {
            Blas.axpy_ongpu(Outputs, learningRate / batch, BiasUpdatesGpu, BiasesGpu);
            Blas.scal_ongpu(Outputs, momentum, ref BiasUpdatesGpu, 1);

            if (BatchNormalize)
            {
                Blas.axpy_ongpu(Outputs, learningRate / batch, ScaleUpdatesGpu, ScalesGpu);
                Blas.scal_ongpu(Outputs, momentum, ref ScaleUpdatesGpu, 1);
            }

            Blas.axpy_ongpu(Inputs * Outputs, -decay * batch, WeightsGpu, WeightUpdatesGpu);
            Blas.axpy_ongpu(Inputs * Outputs, learningRate / batch, WeightUpdatesGpu, WeightsGpu);
            Blas.scal_ongpu(Inputs * Outputs, momentum, ref WeightUpdatesGpu, 1);
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            int i;
            Blas.fill_ongpu(Outputs * Batch, 0, ref OutputGpu, 1);

            int m = Batch;
            int k = Inputs;
            int n = Outputs;
            float[] a = state.Input;
            float[] b = WeightsGpu;
            float[] c = OutputGpu;
            GemmUtils.gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, ref c, n);
            if (BatchNormalize)
            {
                ForwardGpu(ref state);
            }
            for (i = 0; i < Batch; ++i)
            {
                Blas.axpy_ongpu(Outputs, 1, BiasesGpu, OutputGpu, 0, i * Outputs);
            }
            ActivationsHelper.activate_array_ongpu(ref OutputGpu, Outputs * Batch, Activation);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            int i;
            Blas.constrain_ongpu(Outputs * Batch, 1, ref DeltaGpu, 1);
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, ref DeltaGpu);
            for (i = 0; i < Batch; ++i)
            {
                Blas.axpy_ongpu(Outputs, 1, DeltaGpu, BiasUpdatesGpu, i * Outputs);
            }

            if (BatchNormalize)
            {
                BackwardGpu(ref state);
            }

            int m = Outputs;
            int k = Batch;
            int n = Inputs;
            float[] a = DeltaGpu;
            float[] b = state.Input;
            float[] c = WeightUpdatesGpu;
            GemmUtils.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, ref c, n);

            m = Batch;
            k = Outputs;
            n = Inputs;

            a = DeltaGpu;
            b = WeightsGpu;
            c = state.Delta;

            if (c.Any()) GemmUtils.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, ref c, n);
        }

    }
}