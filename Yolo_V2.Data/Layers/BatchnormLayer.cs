using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class BatchnormLayer : Layer
    {
        public BatchnormLayer(int batch, int w, int h, int c)
        {
            Console.Error.Write($"Batch Normalization Layer: {w} x {h} x {c} Image\n");
            LayerType = Layers.Batchnorm;
            Batch = batch;
            Height = OutH = h;
            Width = OutW = w;
            NumberOfChannels = OutC = c;
            Output = new float[h * w * c * batch];
            Delta = new float[h * w * c * batch];
            Inputs = w * h * c;
            Outputs = Inputs;

            Scales = new float[c];
            ScaleUpdates = new float[c];
            int i;
            for (i = 0; i < c; ++i)
            {
                Scales[i] = 1;
            }

            Mean = new float[c];
            Variance = new float[c];

            RollingMean = new float[c];
            RollingVariance = new float[c];

            OutputGpu = new float[h * w * c * batch];
            DeltaGpu = new float[h * w * c * batch];

            ScalesGpu = new float[c];
            ScaleUpdatesGpu = new float[c];

            MeanGpu = new float[c];
            VarianceGpu = new float[c];

            RollingMeanGpu = new float[c];
            RollingVarianceGpu = new float[c];

            MeanDeltaGpu = new float[c];
            VarianceDeltaGpu = new float[c];

            XGpu = new float[Batch * Outputs];
            XNormGpu = new float[Batch * Outputs];
        }

        public override void Forward(ref NetworkState state)
        {
            if (LayerType == Layers.Batchnorm)
            {
                Blas.Copy_cpu(Outputs * Batch, state.Input, Output);
            }
            if (LayerType == Layers.Connected)
            {
                OutC = Outputs;
                OutH = OutW = 1;
            }
            if (state.Train)
            {
                Blas.Mean_cpu(Output, Batch, OutC, OutH * OutW, Mean);
                Blas.Variance_cpu(Output, Mean, Batch, OutC, OutH * OutW, Variance);

                Blas.Scal_cpu(OutC, .9f, RollingMean, 1);
                Blas.Axpy_cpu(OutC, .1f, Mean, RollingMean);
                Blas.Scal_cpu(OutC, .9f, RollingVariance, 1);
                Blas.Axpy_cpu(OutC, .1f, Variance, RollingVariance);

                Blas.Copy_cpu(Outputs * Batch, Output, X);
                Blas.Normalize_cpu(Output, Mean, Variance, Batch, OutC, OutH * OutW);
                Blas.Copy_cpu(Outputs * Batch, Output, XNorm);
            }
            else
            {
                Blas.Normalize_cpu(Output, RollingMean, RollingVariance, Batch, OutC, OutH * OutW);
            }
            Blas.Scale_bias(Output, Scales, Batch, OutC, OutH * OutW);
        }

        public override void Backward(ref NetworkState state)
        {
            backward_scale_cpu(XNorm, Delta, Batch, OutC, OutW * OutH, ScaleUpdates);

            Blas.Scale_bias(Delta, Scales, Batch, OutC, OutH * OutW);

            mean_delta_cpu(Delta, Variance, Batch, OutC, OutW * OutH, MeanDelta);
            variance_delta_cpu(X, Delta, Mean, Variance, Batch, OutC, OutW * OutH, VarianceDelta);
            normalize_delta_cpu(X, Mean, Variance, MeanDelta, VarianceDelta, Batch, OutC, OutW * OutH, Delta);
            if (LayerType == Layers.Batchnorm) Blas.Copy_cpu(Outputs * Batch, Delta, state.Delta);
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
            if (LayerType == Layers.Batchnorm)
            {
                Blas.copy_ongpu(Outputs * Batch, state.Input, ref OutputGpu);
            }
            if (LayerType == Layers.Connected)
            {
                OutC = Outputs;
                OutH = OutW = 1;
            }
            if (state.Train)
            {
                Blas.fast_mean_gpu(OutputGpu, Batch, OutC, OutH * OutW, ref MeanGpu);
                Blas.fast_variance_gpu(OutputGpu, MeanGpu, Batch, OutC, OutH * OutW, ref VarianceGpu);

                Blas.scal_ongpu(OutC, .99f, ref RollingMeanGpu, 1);
                Blas.axpy_ongpu(OutC, .01f, MeanGpu, RollingMeanGpu);
                Blas.scal_ongpu(OutC, .99f, ref RollingVarianceGpu, 1);
                Blas.axpy_ongpu(OutC, .01f, VarianceGpu, RollingVarianceGpu);

                Blas.copy_ongpu(Outputs * Batch, OutputGpu, ref XGpu);
                Blas.normalize_gpu(ref OutputGpu, MeanGpu, VarianceGpu, Batch, OutC, OutH * OutW);
                Blas.copy_ongpu(Outputs * Batch, OutputGpu, ref XNormGpu);
            }
            else
            {
                Utils.normalize_array(ref OutputGpu, OutputGpu.Length);
                //Blas.normalize_gpu(OutputGpu, RollingMeanGpu, RollingVarianceGpu, Batch, OutC, OutH * OutW);
            }

            Blas.scale_bias_gpu(ref OutputGpu, ScalesGpu, Batch, OutC, OutH * OutW);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            Blas.backward_scale_gpu(XNormGpu, DeltaGpu, Batch, OutC, OutW * OutH, ref ScaleUpdatesGpu);

            Blas.scale_bias_gpu(ref DeltaGpu, ScalesGpu, Batch, OutC, OutH * OutW);

            Blas.fast_mean_delta_gpu(DeltaGpu, VarianceGpu, Batch, OutC, OutW * OutH, ref MeanDeltaGpu);
            Blas.fast_variance_delta_gpu(XGpu, DeltaGpu, MeanGpu, VarianceGpu, Batch, OutC, OutW * OutH, ref VarianceDeltaGpu);
            Blas.normalize_delta_gpu(XGpu, MeanGpu, VarianceGpu, MeanDeltaGpu, VarianceDeltaGpu, Batch, OutC, OutW * OutH, ref DeltaGpu);
            if (LayerType == Layers.Batchnorm) Blas.copy_ongpu(Outputs * Batch, DeltaGpu, ref state.Delta);
        }

    }
}