using System;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class ConvolutionalLayer : Layer
    {

        public ConvolutionalLayer(int batch, int height, int width, int numberOfChannels, int n, int size, int stride, int padding,
            Activation activation, bool batchNormalize, bool binary, bool xnor, bool adam)
        {
            int i;
            LayerType = Layers.Convolutional;

            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            N = n;
            Binary = binary;
            Xnor = xnor;
            Batch = batch;
            Stride = stride;
            Size = size;
            Pad = padding;
            BatchNormalize = batchNormalize;

            WeightsComplete = new float[numberOfChannels * n * size * size];
            WeightsIndex = 0;
            WeightUpdates = new float[numberOfChannels * n * size * size];

            BiasesComplete = new float[n];
            BiasesIndex = 0;
            BiasUpdates = new float[n];

            // float scale = 1./(float)Math.Sqrt(size*size*numberOfChannels);
            float scale = (float)Math.Sqrt(2.0 / (size * size * numberOfChannels));
            for (i = 0; i < numberOfChannels * n * size * size; ++i) WeightsComplete[i] = scale * Utils.rand_uniform(-1, 1);
            int outH = convolutional_out_height();
            int outW = convolutional_out_width();
            OutH = outH;
            OutW = outW;
            OutC = n;
            Outputs = OutH * OutW * OutC;
            Inputs = Width * Height * NumberOfChannels;

            Output = new float[Batch * Outputs];
            Delta = new float[Batch * Outputs];

            if (binary)
            {
                BinaryWeights = new float[numberOfChannels * n * size * size];
                Scales = new float[n];
            }
            if (xnor)
            {
                BinaryWeights = new float[numberOfChannels * n * size * size];
                BinaryInput = new float[Inputs * Batch];
            }

            if (batchNormalize)
            {
                Scales = new float[n];
                ScaleUpdates = new float[n];
                for (i = 0; i < n; ++i)
                {
                    Scales[i] = 1;
                }

                Mean = new float[n];
                Variance = new float[n];

                MeanDelta = new float[n];
                VarianceDelta = new float[n];

                RollingMean = new float[n];
                RollingVariance = new float[n];
                X = new float[Batch * Outputs];
                XNorm = new float[Batch * Outputs];
            }
            if (adam)
            {
                Adam = true;
                M = new float[numberOfChannels * n * size * size];
                V = new float[numberOfChannels * n * size * size];
            }

            if (CudaUtils.UseGpu)
            {
                if (adam)
                {
                    MGpu = (float[])M.Clone();
                    VGpu = (float[])V.Clone();
                }

                WeightsGpu = (float[])WeightsComplete.Clone();
                WeightUpdatesGpu = (float[])WeightUpdates.Clone();

                BiasesGpu = (float[])BiasesComplete.Clone();
                BiasUpdatesGpu = (float[])BiasUpdates.Clone();

                DeltaGpu = (float[])Delta.Clone();
                OutputGpu = (float[])Output.Clone();

                if (binary)
                {
                    BinaryWeightsGpu = (float[])WeightsComplete.Clone();
                }
                if (xnor)
                {
                    BinaryWeightsGpu = (float[])WeightsComplete.Clone();
                    BinaryInputGpu = new float[Inputs * Batch];
                }

                if (batchNormalize)
                {
                    MeanGpu = (float[])Mean.Clone();
                    VarianceGpu = (float[])Variance.Clone();

                    RollingMeanGpu = (float[])Mean.Clone();
                    RollingVarianceGpu = (float[])Variance.Clone();

                    MeanDeltaGpu = (float[])Mean.Clone();
                    VarianceDeltaGpu = (float[])Variance.Clone();

                    ScalesGpu = (float[])Scales.Clone();
                    ScaleUpdatesGpu = (float[])ScaleUpdates.Clone();

                    XGpu = (float[])Output.Clone();
                    XNormGpu = (float[])Output.Clone();
                }

            }
            WorkspaceSize = get_workspace_size();
            Activation = activation;

            Console.Error.Write($"conv\t\t{n}\t{size} x{size} /{stride}\t{width} x{height} x{numberOfChannels}\t.\t{OutW} x{OutH} x{OutC}\n");

        }

        public override void Forward(ref NetworkState state)
        {
            int outH = convolutional_out_height();
            int outW = convolutional_out_width();
            int i;

            Blas.Fill_cpu(Outputs * Batch, 0, Output, 1);

            if (Xnor)
            {
                binarize_weights(WeightsComplete, N, NumberOfChannels * Size * Size, BinaryWeights, WeightsIndex);
                swap_binary();
                binarize_cpu(state.Input, NumberOfChannels * Height * Width * Batch, BinaryInput);
                state.Input = BinaryInput;
            }

            int m = N;
            int k = Size * Size * NumberOfChannels;
            int n = outH * outW;


            float[] a = new float[WeightsComplete.Length - WeightsIndex];
            Array.Copy(WeightsComplete, WeightsIndex, a, 0, a.Length);
            float[] b = state.Workspace;
            float[] c = Output;

            for (i = 0; i < Batch; ++i)
            {
                Im2Col.im2col_cpu(state.Input, NumberOfChannels, Height, Width,
                    Size, Stride, Pad, b);
                GemmUtils.Gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                c[i] += n * m;
                state.Input[i] += NumberOfChannels * Height * Width;
            }

            if (BatchNormalize)
            {
                throw new Exception("not a batchnorm layer");
                //forward_batchnorm_layer(l, state);
            }
            add_bias(Output, BiasesComplete, Batch, N, outH * outW, BiasesIndex);

            ActivationsHelper.Activate_array(Output, m * n * Batch, Activation);
            if (Binary || Xnor) swap_binary();
        }

        public override void Backward(ref NetworkState state)
        {
            int i;
            int m = N;
            int n = Size * Size * NumberOfChannels;
            int k = convolutional_out_height() *
                    convolutional_out_width();

            ActivationsHelper.Gradient_array(Output, m * k * Batch, Activation, Delta);
            backward_bias(BiasUpdates, Delta, Batch, N, k);

            if (BatchNormalize)
            {
                throw new Exception("not batchnorm");
                //backward_batchnorm_layer(l, state);
            }

            for (i = 0; i < Batch; ++i)
            {
                float[] a = new float[Delta.Length - i * m * k];

                Im2Col.im2col_cpu(state.Input, NumberOfChannels, Height, Width,
                    Size, Stride, Pad, state.Workspace, i * NumberOfChannels * Height * Width);
                GemmUtils.Gemm(0, 1, m, n, k, 1, a, k, state.Workspace, k, 1, WeightUpdates, n);

                if (state.Delta.Any())
                {
                    var b = a;

                    a = new float[WeightsComplete.Length - WeightsIndex];
                    Array.Copy(WeightsComplete, WeightsIndex, a, 0, a.Length);

                    GemmUtils.Gemm(1, 0, n, k, m, 1, a, n, b, k, 0, state.Workspace, k);

                    Im2Col.col2im_cpu(state.Workspace, NumberOfChannels, Height, Width, Size, Stride, Pad, state.Delta, i * NumberOfChannels * Height * Width);

                }
            }
        }

        public override void Update(ref int batch, ref float learningRate, ref float momentum, ref float decay)
        {
            int size = Size * Size * NumberOfChannels * N;
            Blas.Axpy_cpu(N, learningRate / batch, BiasUpdates, BiasesComplete, 0, BiasesIndex);
            Blas.Scal_cpu(N, momentum, BiasUpdates, 1);

            if (Scales.Any())
            {
                Blas.Axpy_cpu(N, learningRate / batch, ScaleUpdates, Scales);
                Blas.Scal_cpu(N, momentum, ScaleUpdates, 1);
            }

            Blas.Axpy_cpu(size, -decay * batch, WeightsComplete, WeightUpdates, WeightsIndex);
            Blas.Axpy_cpu(size, learningRate / batch, WeightUpdates, WeightsComplete, 0, WeightsIndex);
            Blas.Scal_cpu(size, momentum, WeightUpdates, 1);
        }

        public Image[] visualize_convolutional_layer(string window)
        {
            Image[] singleWeights = get_weights();
            LoadArgs.show_images(singleWeights, N, window);

            Image delta = get_convolutional_image();
            Image dc = LoadArgs.collapse_image_layers(delta, 1);
            string buff = $"{window}: Output";
            return singleWeights;
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            Blas.fill_ongpu(Outputs * Batch, 0, ref OutputGpu, 1);
            if (Binary)
            {
                binarize_weights_gpu(WeightsGpu, N, NumberOfChannels * Size * Size, ref BinaryWeightsGpu);
                swap_binary();
            }

            if (Xnor)
            {
                binarize_weights_gpu(WeightsGpu, N, NumberOfChannels * Size * Size, ref BinaryWeightsGpu);
                swap_binary();
                binarize_gpu(state.Input, NumberOfChannels * Height * Width * Batch, ref BinaryInputGpu);
                state.Input = BinaryInputGpu;
            }

            int i;
            int m = N;
            int k = Size * Size * NumberOfChannels;
            int n = OutW * OutH;
            for (i = 0; i < Batch; ++i)
            {
                Im2Col.im2col_ongpu(state.Input, NumberOfChannels, Height, Width, Size, Stride, Pad, ref state.Workspace, i * NumberOfChannels * Height * Width);
                float[] a = WeightsGpu;
                float[] b = state.Workspace;
                float[] c = new float[OutputGpu.Length - i * m * n];

                Array.Copy(OutputGpu, i * m * n, c, 0, c.Length);
                GemmUtils.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, ref c, n);

                Array.Copy(c, 0, OutputGpu, i * m * n, c.Length);
            }

            if (BatchNormalize)
            {
                BatchnormalForwardGpu(ref state);
            }

            Blas.add_bias_gpu(ref OutputGpu, BiasesGpu, Batch, N, OutW * OutH);

            ActivationsHelper.activate_array_ongpu(ref OutputGpu, Outputs * Batch, Activation);
            if (Binary || Xnor) swap_binary();
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, ref DeltaGpu);

            Blas.backward_bias_gpu(ref BiasUpdatesGpu, DeltaGpu, Batch, N, OutW * OutH);

            if (BatchNormalize)
            {
                throw new Exception("not batchnorm");
                //backward_batchnorm_layer_gpu(l, state);
            }
            float[] originalInput = state.Input;

            if (Xnor) state.Input = BinaryInputGpu;

            int m = N;
            int n = Size * Size * NumberOfChannels;
            int k = OutW * OutH;

            int i;
            for (i = 0; i < Batch; ++i)
            {
                float[] a = new float[DeltaGpu.Length - i * m * k];
                Array.Copy(DeltaGpu, i * m * k, a, 0, a.Length);

                Im2Col.im2col_ongpu(state.Input, NumberOfChannels, Height, Width, Size, Stride, Pad, ref state.Workspace, i * NumberOfChannels * Height * Width);
                GemmUtils.gemm_ongpu(0, 1, m, n, k, 1, a, k, state.Workspace, k, 1, ref WeightUpdatesGpu, n);

                if (state.Delta.Length > 0)
                {
                    if (Binary || Xnor) swap_binary();

                    GemmUtils.gemm_ongpu(1, 0, n, k, m, 1, WeightsGpu, n, a, k, 0, ref state.Workspace, k);

                    Im2Col.col2im_ongpu(state.Workspace, NumberOfChannels, Height, Width, Size, Stride, Pad, state.Delta, i * NumberOfChannels * Height * Width);
                    if (Binary || Xnor)
                    {
                        swap_binary();
                    }
                    if (Xnor) ActivationsHelper.gradient_array_ongpu(originalInput, NumberOfChannels * Height * Width, Activation.Hardtan, ref state.Delta, i * NumberOfChannels * Height * Width, i * NumberOfChannels * Height * Width);
                }
            }
        }

        public override void UpdateGpu(ref int batch, ref float learningRate, ref float momentum, ref float decay)
        {
            int size = Size * Size * NumberOfChannels * N;
            Blas.axpy_ongpu(N, learningRate / batch, BiasUpdatesGpu, BiasesGpu);
            Blas.scal_ongpu(N, momentum, ref BiasUpdatesGpu, 1);

            if (ScalesGpu.Any())
            {
                Blas.axpy_ongpu(N, learningRate / batch, ScaleUpdatesGpu, ScalesGpu);
                Blas.scal_ongpu(N, momentum, ref ScaleUpdatesGpu, 1);
            }

            if (Adam)
            {
                Blas.scal_ongpu(size, B1, ref MGpu, 1);
                Blas.scal_ongpu(size, B2, ref VGpu, 1);

                Blas.axpy_ongpu(size, -decay * batch, WeightsGpu, WeightUpdatesGpu);

                Blas.axpy_ongpu(size, -(1 - B1), WeightUpdatesGpu, MGpu);
                Blas.mul_ongpu(size, WeightUpdatesGpu, 1, ref WeightUpdatesGpu, 1);
                Blas.axpy_ongpu(size, (1 - B2), WeightUpdatesGpu, VGpu);

                Blas.adam_gpu(size, ref WeightsGpu, MGpu, VGpu, B1, B2, learningRate / batch, Eps, T + 1);
                Blas.fill_ongpu(size, 0, ref WeightUpdatesGpu, 1);
            }
            else
            {
                Blas.axpy_ongpu(size, -decay * batch, WeightsGpu, WeightUpdatesGpu);
                Blas.axpy_ongpu(size, learningRate / batch, WeightUpdatesGpu, WeightsGpu);
                Blas.scal_ongpu(size, momentum, ref WeightUpdatesGpu, 1);
            }
        }

        public int convolutional_out_height()
        {
            return (Height + 2 * Pad - Size) / Stride + 1;
        }

        public int convolutional_out_width()
        {
            return (Width + 2 * Pad - Size) / Stride + 1;
        }

        public Image get_convolutional_image()
        {
            int h, w, c;
            h = convolutional_out_height();
            w = convolutional_out_width();
            c = N;
            return new Image(w, h, c, Output);
        }

        public void resize_convolutional_layer(int w, int h)
        {
            Width = w;
            Height = h;
            int outW = convolutional_out_width();
            int outH = convolutional_out_height();

            OutW = outW;
            OutH = outH;

            Outputs = OutH * OutW * OutC;
            Inputs = Width * Height * NumberOfChannels;

            Array.Resize(ref Output, Batch * Outputs);
            Array.Resize(ref Delta, Batch * Outputs);
            if (BatchNormalize)
            {
                Array.Resize(ref X, Batch * Outputs);
                Array.Resize(ref XNorm, Batch * Outputs);
            }

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

            if (BatchNormalize)
            {
                XGpu = (float[])Output.Clone();
                XNormGpu = (float[])Output.Clone();
            }

            WorkspaceSize = get_workspace_size();
        }

        public void BatchnormalForwardGpu(ref NetworkState state)
        {
            if (LayerType == Layers.Batchnorm)
            {
                Blas.copy_ongpu(Outputs * Batch, state.Input, OutputGpu);
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

                Blas.copy_ongpu(Outputs * Batch, OutputGpu, XGpu);
                Blas.normalize_gpu(ref OutputGpu, MeanGpu, VarianceGpu, Batch, OutC, OutH * OutW);
                Blas.copy_ongpu(Outputs * Batch, OutputGpu, XNormGpu);
            }
            else
            {
                Blas.normalize_gpu(ref OutputGpu, RollingMeanGpu, RollingVarianceGpu, Batch, OutC, OutH * OutW);
            }

            Blas.scale_bias_gpu(ref OutputGpu, ScalesGpu, Batch, OutC, OutH * OutW);
        }

    }
}