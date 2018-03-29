using System;
using System.Collections.Generic;
using System.Linq;
using Alea.CudaDnn;
using Alea.CudaToolkit;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class Layer
    {
        public LayerType LayerType;
        public Activation Activation;
        public CostType CostType;
        public Action<NetworkState> Forward;
        public Action<NetworkState> Backward;
        public Action<int, float, float, float> Update;
        public Action<NetworkState> ForwardGpu;
        public Action<NetworkState> BackwardGpu;
        public Action<int, float, float, float> UpdateGpu;
        public int BatchNormalize;
        public int Shortcut;
        public int Batch;
        public int Forced;
        public int Flipped;
        public int Inputs;
        public int Outputs;
        public int Truths;
        public int H, W, C;
        public int OutH;
        public int OutW;
        public int OutC;
        public int N;
        public int MaxBoxes;
        public int Groups;
        public int Size;
        public int Side;
        public int Stride;
        public int Reverse;
        public int Pad;
        public int Sqrt;
        public int Flip;
        public int Index;
        public int Binary;
        public int Xnor;
        public int Steps;
        public int Hidden;
        public float Dot;
        public float Angle;
        public float Jitter;
        public float Saturation;
        public float Exposure;
        public float Shift;
        public float Ratio;
        public int Softmax;
        public int Classes;
        public int Coords;
        public int Background;
        public int Rescore;
        public int Objectness;
        public int DoesCost;
        public int Joint;
        public int Noadjust;
        public int Reorg;
        public int Log;

        public int Adam;
        public float B1;
        public float B2;
        public float Eps;
        public float[] MGpu;
        public float[] VGpu;
        public int T;
        public float[] M;
        public float[] V;

        public Tree SoftmaxTree;
        public int[] Map;

        public float Alpha;
        public float Beta;
        public float Kappa;

        public float CoordScale;
        public float ObjectScale;
        public float NoobjectScale;
        public float ClassScale;
        public int BiasMatch;
        public int Random;
        public float Thresh;
        public int Classfix;
        public int Absolute;

        public bool Dontload;
        public bool Dontloadscales;

        public float Temperature;
        public float Probability;
        public float Scale;

        public List<int> Indexes;
        public float[] Rand;
        public float[] Cost;
        public string Cweights;
        public float[] State;
        public float[] PrevState;
        public float[] ForgotState;
        public float[] ForgotDelta;
        public float[] StateDelta;

        public float[] Concat;
        public float[] ConcatDelta;

        public float[] BinaryWeights;

        public float[] Biases;
        public float[] BiasUpdates;

        public float[] Scales;
        public float[] ScaleUpdates;

        public float[] Weights;
        public float[] WeightUpdates;

        public float[] ColImage;
        public List<int> InputLayers;
        public List<int> InputSizes;
        public float[] Delta;
        public float[] Output;
        public float[] Squared;
        public float[] Norms;

        public float[] SpatialMean;
        public float[] Mean;
        public float[] Variance;

        public float[] MeanDelta;
        public float[] VarianceDelta;

        public float[] RollingMean;
        public float[] RollingVariance;

        public float[] X;
        public float[] XNorm;

        public Layer InputLayer;
        public Layer SelfLayer;
        public Layer OutputLayer;

        public Layer InputGateLayer;
        public Layer StateGateLayer;
        public Layer InputSaveLayer;
        public Layer StateSaveLayer;
        public Layer InputStateLayer;
        public Layer StateStateLayer;

        public Layer InputZLayer;
        public Layer StateZLayer;

        public Layer InputRLayer;
        public Layer StateRLayer;

        public Layer InputHLayer;
        public Layer StateHLayer;

        public float[] ZCpu;
        public float[] RCpu;
        public float[] HCpu;

        public float[] BinaryInput;

        public ulong WorkspaceSize;

        public float[] ZGpu;
        public float[] RGpu;
        public float[] HGpu;

        public List<int> IndexesGpu;
        public float[] PrevStateGpu;
        public float[] ForgotStateGpu;
        public float[] ForgotDeltaGpu;
        public float[] StateGpu;
        public float[] StateDeltaGpu;
        public float[] GateGpu;
        public float[] GateDeltaGpu;
        public float[] SaveGpu;
        public float[] SaveDeltaGpu;
        public float[] ConcatGpu;
        public float[] ConcatDeltaGpu;

        public float[] BinaryInputGpu;
        public float[] BinaryWeightsGpu;

        public float[] MeanGpu;
        public float[] VarianceGpu;

        public float[] RollingMeanGpu;
        public float[] RollingVarianceGpu;

        public float[] VarianceDeltaGpu;
        public float[] MeanDeltaGpu;

        public float[] ColImageGpu;

        public float[] XGpu;
        public float[] XNormGpu;
        public float[] WeightsGpu;
        public float[] WeightUpdatesGpu;

        public float[] BiasesGpu;
        public float[] BiasUpdatesGpu;

        public float[] ScalesGpu;
        public float[] ScaleUpdatesGpu;

        public float[] OutputGpu;
        public float[] DeltaGpu;
        public float[] RandGpu;
        public float[] SquaredGpu;
        public float[] NormsGpu;
        public cudnnFilterStruct SrcTensorDesc, DstTensorDesc;
        public cudnnFilterStruct DsrcTensorDesc, DdstTensorDesc;
        public cudnnFilterStruct WeightDesc;
        public cudnnFilterStruct DweightDesc;
        public cudnnConvolutionStruct ConvDesc;
        public cudnnConvolutionFwdAlgo_t FwAlgo;
        public cudnnConvolutionBwdDataAlgo_t BdAlgo;
        public cudnnConvolutionBwdFilterAlgo_t BfAlgo;

        public static void CombineLists(float[] start, int count, float[] end)
        {
            var temp = new float[start.Length + end.Length];
            Array.Copy(start, count, temp, 0, start.Length);
            Array.Copy(end, 0, temp, count, end.Length);
            start = temp;
        }

        public int local_out_height()
        {
            int h = H;
            if (Pad == 0) h -= Size;
            else h -= 1;
            return h / Stride + 1;
        }

        int local_out_width()
        {
            int w = W;
            if (Pad == 0) w -= Size;
            else w -= 1;
            return w / Stride + 1;
        }

        public Layer() { }

        public Layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, Activation activation)
        {
            int i;
            LayerType = LayerType.Local;

            H = h;
            W = w;
            C = c;
            N = n;
            Batch = batch;
            Stride = stride;
            Size = size;
            Pad = pad;

            int out_h = local_out_height();
            int out_w = local_out_width();
            int locations = out_h * out_w;
            OutH = out_h;
            OutW = out_w;
            OutC = n;
            Outputs = OutH * OutW * OutC;
            Inputs = W * H * C;

            Weights = new float[c * n * size * size * locations];
            WeightUpdates = new float[c * n * size * size * locations];

            Biases = new float[Outputs];
            BiasUpdates = new float[Outputs];

            float scale = (float)Math.Sqrt(2.0f / (size * size * c));
            for (i = 0; i < c * n * size * size; ++i) Weights[i] = scale * Utils.rand_uniform(-1, 1);

            ColImage = new float[out_h * out_w * size * size * c];
            Output = new float[Batch * out_h * out_w * n];
            Delta = new float[Batch * out_h * out_w * n];

            Forward = forward_local_layer;
            Backward = backward_local_layer;
            Update = update_local_layer;

            ForwardGpu = forward_local_layer_gpu;
            BackwardGpu = backward_local_layer_gpu;
            UpdateGpu = update_local_layer_gpu;

            WeightsGpu = Weights.ToArray();
            WeightUpdatesGpu = WeightUpdates.ToArray();

            BiasesGpu = Biases.ToArray();
            BiasUpdatesGpu = BiasUpdates.ToArray();

            ColImageGpu = ColImage.ToArray();
            DeltaGpu = Delta.ToArray();
            OutputGpu = Output.ToArray();

            Activation = activation;

            Console.Error.WriteLine($"Local Layer: {h} x {w} x {c} Image, {n} filters . {out_h} x {out_w} x {n} Image");
        }

        public static Layer make_batchnorm_layer(int batch, int w, int h, int c)
        {
            Console.Error.Write($"Batch Normalization Layer: {w} x {h} x {c} Image\n");
            Layer Layer = new Layer();
            Layer.LayerType = LayerType.Batchnorm;
            Layer.Batch = batch;
            Layer.H = Layer.OutH = h;
            Layer.W = Layer.OutW = w;
            Layer.C = Layer.OutC = c;
            Layer.Output = new float[h * w * c * batch];
            Layer.Delta = new float[h * w * c * batch];
            Layer.Inputs = w * h * c;
            Layer.Outputs = Layer.Inputs;

            Layer.Scales = new float[c];
            Layer.ScaleUpdates = new float[c];
            int i;
            for (i = 0; i < c; ++i)
            {
                Layer.Scales[i] = 1;
            }

            Layer.Mean = new float[c];
            Layer.Variance = new float[c];

            Layer.RollingMean = new float[c];
            Layer.RollingVariance = new float[c];

            Layer.Forward = Layer.forward_batchnorm_layer;
            Layer.Backward = Layer.backward_batchnorm_layer;
            Layer.ForwardGpu = Layer.forward_batchnorm_layer_gpu;
            Layer.BackwardGpu = Layer.backward_batchnorm_layer_gpu;

            Layer.OutputGpu = new float[h * w * c * batch];
            Layer.DeltaGpu = new float[h * w * c * batch];

            Layer.ScalesGpu = new float[c];
            Layer.ScaleUpdatesGpu = new float[c];

            Layer.MeanGpu = new float[c];
            Layer.VarianceGpu = new float[c];

            Layer.RollingMeanGpu = new float[c];
            Layer.RollingVarianceGpu = new float[c];

            Layer.MeanDeltaGpu = new float[c];
            Layer.VarianceDeltaGpu = new float[c];

            Layer.XGpu = new float[Layer.Batch * Layer.Outputs];
            Layer.XNormGpu = new float[Layer.Batch * Layer.Outputs];
            return Layer;
        }

        public static void backward_scale_cpu(float[] x_norm, float[] delta, int batch, int n, int size, float[] scale_updates)
        {
            int i, b, f;
            for (f = 0; f < n; ++f)
            {
                float sum = 0;
                for (b = 0; b < batch; ++b)
                {
                    for (i = 0; i < size; ++i)
                    {
                        int index = i + size * (f + n * b);
                        sum += delta[index] * x_norm[index];
                    }
                }
                scale_updates[f] += sum;
            }
        }

        public static void mean_delta_cpu(float[] delta, float[] variance, int batch, int filters, int spatial, float[] mean_delta)
        {

            int i, j, k;
            for (i = 0; i < filters; ++i)
            {
                mean_delta[i] = 0;
                for (j = 0; j < batch; ++j)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + i * spatial + k;
                        mean_delta[i] += delta[index];
                    }
                }
                mean_delta[i] *= (-1.0f / (float)Math.Sqrt(variance[i] + .00001f));
            }
        }

        public static void variance_delta_cpu(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, float[] variance_delta)
        {

            int i, j, k;
            for (i = 0; i < filters; ++i)
            {
                variance_delta[i] = 0;
                for (j = 0; j < batch; ++j)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + i * spatial + k;
                        variance_delta[i] += delta[index] * (x[index] - mean[i]);
                    }
                }
                variance_delta[i] *= -.5f * (float)Math.Pow(variance[i] + .00001f, (-3.0f / 2.0f));
            }
        }

        public static void normalize_delta_cpu(float[] x, float[] mean, float[] variance, float[] mean_delta, float[] variance_delta, int batch, int filters, int spatial, float[] delta)
        {
            int f, j, k;
            for (j = 0; j < batch; ++j)
            {
                for (f = 0; f < filters; ++f)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + f * spatial + k;
                        delta[index] = delta[index] * 1.0f / ((float)Math.Sqrt(variance[f]) + .00001f)
                                       + variance_delta[f] * 2.0f * (x[index] - mean[f]) / (spatial * batch)
                                       + mean_delta[f] / (spatial * batch);
                    }
                }
            }
        }

        public static void resize_batchnorm_layer(Layer Layer, int w, int h)
        {
            Console.Error.Write($"Not implemented\n");
        }

        public void forward_batchnorm_layer(NetworkState state)
        {
            if (LayerType == LayerType.Batchnorm)
            {
                Blas.Copy_cpu(Outputs * Batch, state.Input, 1, Output, 1);
            }
            if (LayerType == LayerType.Connected)
            {
                OutC = Outputs;
                OutH = OutW = 1;
            }
            if (state.Train != 0)
            {
                Blas.Mean_cpu(Output, Batch, OutC, OutH * OutW, Mean);
                Blas.Variance_cpu(Output, Mean, Batch, OutC, OutH * OutW, Variance);

                Blas.Scal_cpu(OutC, .9f, RollingMean, 1);
                Blas.Axpy_cpu(OutC, .1f, Mean, 1, RollingMean, 1);
                Blas.Scal_cpu(OutC, .9f, RollingVariance, 1);
                Blas.Axpy_cpu(OutC, .1f, Variance, 1, RollingVariance, 1);

                Blas.Copy_cpu(Outputs * Batch, Output, 1, X, 1);
                Blas.Normalize_cpu(Output, Mean, Variance, Batch, OutC, OutH * OutW);
                Blas.Copy_cpu(Outputs * Batch, Output, 1, XNorm, 1);
            }
            else
            {
                Blas.Normalize_cpu(Output, RollingMean, RollingVariance, Batch, OutC, OutH * OutW);
            }
            Blas.Scale_bias(Output, Scales, Batch, OutC, OutH * OutW);
        }

        public void backward_batchnorm_layer(NetworkState state)
        {
            backward_scale_cpu(XNorm, Delta, Batch, OutC, OutW * OutH, ScaleUpdates);

            Blas.Scale_bias(Delta, Scales, Batch, OutC, OutH * OutW);

            mean_delta_cpu(Delta, Variance, Batch, OutC, OutW * OutH, MeanDelta);
            variance_delta_cpu(X, Delta, Mean, Variance, Batch, OutC, OutW * OutH, VarianceDelta);
            normalize_delta_cpu(X, Mean, Variance, MeanDelta, VarianceDelta, Batch, OutC, OutW * OutH, Delta);
            if (LayerType == LayerType.Batchnorm) Blas.Copy_cpu(Outputs * Batch, Delta, 1, state.Delta, 1);
        }

        public static void pull_batchnorm_layer(Layer l)
        {
            Array.Copy(l.ScalesGpu, l.Scales, l.C);
            Array.Copy(l.RollingMeanGpu, l.RollingMean, l.C);
            Array.Copy(l.RollingVarianceGpu, l.RollingVariance, l.C);
        }

        public static void push_batchnorm_layer(Layer l)
        {
            Array.Copy(l.Scales, l.ScalesGpu, l.C);
            Array.Copy(l.RollingMean, l.RollingMeanGpu, l.C);
            Array.Copy(l.RollingVariance, l.RollingVarianceGpu, l.C);
        }

        public void forward_batchnorm_layer_gpu(NetworkState state)
        {
            if (LayerType == LayerType.Batchnorm)
            {
                Blas.copy_ongpu(Outputs * Batch, state.Input, 1, OutputGpu, 1);
            }
            if (LayerType == LayerType.Connected)
            {
                OutC = Outputs;
                OutH = OutW = 1;
            }
            if (state.Train != 0)
            {
                Blas.fast_mean_gpu(OutputGpu, Batch, OutC, OutH * OutW, MeanGpu);
                Blas.fast_variance_gpu(OutputGpu, MeanGpu, Batch, OutC, OutH * OutW, VarianceGpu);

                Blas.scal_ongpu(OutC, .99f, RollingMeanGpu, 1);
                Blas.axpy_ongpu(OutC, .01f, MeanGpu, 1, RollingMeanGpu, 1);
                Blas.scal_ongpu(OutC, .99f, RollingVarianceGpu, 1);
                Blas.axpy_ongpu(OutC, .01f, VarianceGpu, 1, RollingVarianceGpu, 1);

                Blas.copy_ongpu(Outputs * Batch, OutputGpu, 1, XGpu, 1);
                Blas.normalize_gpu(OutputGpu, MeanGpu, VarianceGpu, Batch, OutC, OutH * OutW);
                Blas.copy_ongpu(Outputs * Batch, OutputGpu, 1, XNormGpu, 1);
            }
            else
            {
                Blas.normalize_gpu(OutputGpu, RollingMeanGpu, RollingVarianceGpu, Batch, OutC, OutH * OutW);
            }

            Blas.scale_bias_gpu(OutputGpu, ScalesGpu, Batch, OutC, OutH * OutW);
        }

        public void backward_batchnorm_layer_gpu(NetworkState state)
        {
            Blas.backward_scale_gpu(XNormGpu, DeltaGpu, Batch, OutC, OutW * OutH, ScaleUpdatesGpu);

            Blas.scale_bias_gpu(DeltaGpu, ScalesGpu, Batch, OutC, OutH * OutW);

            Blas.fast_mean_delta_gpu(DeltaGpu, VarianceGpu, Batch, OutC, OutW * OutH, MeanDeltaGpu);
            Blas.fast_variance_delta_gpu(XGpu, DeltaGpu, MeanGpu, VarianceGpu, Batch, OutC, OutW * OutH, VarianceDeltaGpu);
            Blas.normalize_delta_gpu(XGpu, MeanGpu, VarianceGpu, MeanDeltaGpu, VarianceDeltaGpu, Batch, OutC, OutW * OutH, DeltaGpu);
            if (LayerType == LayerType.Batchnorm) Blas.copy_ongpu(Outputs * Batch, DeltaGpu, 1, state.Delta, 1);
        }

        private void forward_local_layer(NetworkState state)
        {
            int out_h = local_out_height();
            int out_w = local_out_width();
            int i, j;
            int locations = out_h * out_w;

            for (i = 0; i < Batch; ++i)
            {
                var index = i * Outputs;
                var output = Output.Skip(index).ToArray();
                Blas.Copy_cpu(Outputs, Biases, 1, output, 1);
                CombineLists(Output, index, output);
            }

            for (i = 0; i < Batch; ++i)
            {
                var index = i * W * H * C;
                float[] input = state.Input.Skip(index).ToArray();
                Im2Col.im2col_cpu(input, C, H, W, Size, Stride, Pad, ColImage);

                index = i * Outputs;
                float[] output = Output.Skip(index).ToArray();

                for (j = 0; j < locations; ++j)
                {
                    index = j * Size * Size * C * N;
                    float[] a = Weights.Skip(index).ToArray();
                    float[] b = ColImage.Skip(j).ToArray();
                    float[] c = output.Skip(j).ToArray();

                    int m = N;
                    int n = 1;
                    int k = Size * Size * C;

                    Gemm.gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                    CombineLists(Weights, index, a);
                    CombineLists(ColImage, j, b);
                    CombineLists(output, j, c);
                }
            }
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }

        private void backward_local_layer(NetworkState state)
        {
            int i, j;
            int locations = OutW * OutH;

            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);

            for (i = 0; i < Batch; ++i)
            {
                var index = i * Outputs;
                Blas.Axpy_cpu(Outputs, 1, Delta.Skip(index).ToArray(), 1, BiasUpdates, 1);
            }

            for (i = 0; i < Batch; ++i)
            {
                var index = i * W * H * C;
                float[] input = state.Input.Skip(index).ToArray();
                Im2Col.im2col_cpu(input, C, H, W,
                        Size, Stride, Pad, ColImage);

                for (j = 0; j < locations; ++j)
                {
                    var indexA = i * Outputs + j;
                    float[] a = Delta.Skip(indexA).ToArray();
                    float[] b = ColImage.Skip(j).ToArray();
                    var indexC = j * Size * Size * C * N;
                    float[] c = WeightUpdates.Skip(indexC).ToArray();
                    int m = N;
                    int n = Size * Size * C;
                    int k = 1;

                    Gemm.gemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
                    CombineLists(Delta, indexA, a);
                    CombineLists(ColImage, j, b);
                    CombineLists(WeightUpdates, indexC, c);
                }

                if (state.Delta.Any())
                {
                    for (j = 0; j < locations; ++j)
                    {
                        var indexA = j * Size * Size * C * N;
                        var indexB = i * Outputs + j;
                        float[] a = Weights.Skip(indexA).ToArray();
                        float[] b = Delta.Skip(indexB).ToArray();
                        float[] c = ColImage.Skip(j).ToArray();

                        int m = Size * Size * C;
                        int n = 1;
                        int k = N;

                        Gemm.gemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                        CombineLists(Weights, indexA, a);
                        CombineLists(Delta, indexB, b);
                        CombineLists(ColImage, j, c);
                    }

                    index = i * C * H * W;
                    var output = state.Delta.Skip(index).ToArray();
                    Im2Col.Im2Col.col2im_cpu(ColImage, C, H, W, Size, Stride, Pad, output);
                    CombineLists(state.Delta, index, output);
                }
            }
        }

        private void update_local_layer(int batch, float learning_rate, float momentum, float decay)
        {
            int locations = OutW * OutH;
            int size = Size * Size * C * N * locations;
            Blas.Axpy_cpu(Outputs, learning_rate / batch, BiasUpdates, 1, Biases, 1);
            Blas.Scal_cpu(Outputs, momentum, BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay * batch, Weights, 1, WeightUpdates, 1);
            Blas.Axpy_cpu(size, learning_rate / batch, WeightUpdates, 1, Weights, 1);
            Blas.Scal_cpu(size, momentum, WeightUpdates, 1);
        }

        private void forward_local_layer_gpu(NetworkState state)
        {
            int out_h = local_out_height();
            int out_w = local_out_width();
            int i, j;
            int locations = out_h * out_w;

            //for (i = 0; i < Batch; ++i)
            //{
            //    Blas.copy_ongpu(Outputs, BiasesGpu, 1, Output.Gpu + i * Outputs, 1);
            //}

            for (i = 0; i < Batch; ++i)
            {
                var inIndex = i * W * H * C;
                float[] input = state.Input.Skip(inIndex).ToArray();
                Im2Col.im2col_ongpu(input, C, H, W,
                        Size, Stride, Pad, ColImageGpu);
                var outIndex = i * Outputs;
                float[] output = OutputGpu.Skip(outIndex).ToArray();
                for (j = 0; j < locations; ++j)
                {
                    var aIndex = j * Size * Size * C * N;
                    float[] a = WeightsGpu.Skip(aIndex).ToArray();
                    float[] b = ColImageGpu.Skip(j).ToArray();
                    float[] c = output.Skip(j).ToArray();

                    int m = N;
                    int n = 1;
                    int k = Size * Size * C;

                    Gemm.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                    CombineLists(WeightsGpu, aIndex, a);
                    CombineLists(ColImageGpu, j, b);
                    CombineLists(output, j, c);
                }
                CombineLists(state.Input, inIndex, input);
                CombineLists(OutputGpu, outIndex, output);
            }
            ActivationsHelper.activate_array_ongpu(OutputGpu, Outputs * Batch, Activation);
        }

        private void backward_local_layer_gpu(NetworkState state)
        {
            int i, j;
            int locations = OutW * OutH;

            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, DeltaGpu);
            for (i = 0; i < Batch; ++i)
            {
                var tmp = DeltaGpu.Skip(i * Outputs).ToArray();
                Blas.axpy_ongpu(Outputs, 1, tmp, 1, BiasUpdatesGpu, 1);
            }

            for (i = 0; i < Batch; ++i)
            {
                int index = i * W * H * C;
                float[] input = state.Input.Skip(index).ToArray();
                Im2Col.im2col_ongpu(input, C, H, W,
                        Size, Stride, Pad, ColImageGpu);
                CombineLists(state.Input, index, input);
                for (j = 0; j < locations; ++j)
                {
                    int aIndex = i * Outputs + j;
                    int cIndex = j * Size * Size * C * N;
                    float[] a = DeltaGpu.Skip(aIndex).ToArray();
                    float[] b = ColImageGpu.Skip(j).ToArray();
                    float[] c = WeightUpdatesGpu.Skip(cIndex).ToArray();
                    int m = N;
                    int n = Size * Size * C;
                    int k = 1;

                    Gemm.gemm_ongpu(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
                    CombineLists(DeltaGpu, aIndex, a);
                    CombineLists(ColImageGpu, j, b);
                    CombineLists(WeightUpdatesGpu, cIndex, c);
                }

                if (state.Delta.Any())
                {
                    for (j = 0; j < locations; ++j)
                    {
                        int aIndex = j * Size * Size * C * N;
                        int bIndex = i * Outputs + j;
                        float[] a = WeightsGpu.Skip(aIndex).ToArray();
                        float[] b = DeltaGpu.Skip(bIndex).ToArray();
                        float[] c = ColImageGpu.Skip(j).ToArray();

                        int m = Size * Size * C;
                        int n = 1;
                        int k = N;

                        Gemm.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                        CombineLists(WeightsGpu, aIndex, a);
                        CombineLists(DeltaGpu, bIndex, b);
                        CombineLists(ColImageGpu, j, c);
                    }

                    var dIndex = i * C * H * W;
                    var delta = state.Delta.Skip(dIndex).ToArray();
                    Im2Col.col2im_ongpu(ColImageGpu, C, H, W, Size, Stride, Pad, delta);
                    CombineLists(state.Delta, dIndex, delta);
                }
            }
        }

        private void update_local_layer_gpu(int batch, float learning_rate, float momentum, float decay)
        {
            int locations = OutW * OutH;
            int size = Size * Size * C * N * locations;
            Blas.axpy_ongpu(Outputs, learning_rate / batch, BiasUpdatesGpu, 1, BiasesGpu, 1);
            Blas.scal_ongpu(Outputs, momentum, BiasUpdatesGpu, 1);

            Blas.axpy_ongpu(size, -decay * batch, WeightsGpu, 1, WeightUpdatesGpu, 1);
            Blas.axpy_ongpu(size, learning_rate / batch, WeightUpdatesGpu, 1, WeightsGpu, 1);
            Blas.scal_ongpu(size, momentum, WeightUpdatesGpu, 1);
        }


        void swap_binary(Layer l)
        {
            float[] swap = l.Weights;
            l.Weights = l.BinaryWeights;
            l.BinaryWeights = swap;

            swap = l.WeightsGpu;
            l.WeightsGpu = l.BinaryWeightsGpu;
            l.BinaryWeightsGpu = swap;
        }

        void binarize_weights(float[] weights, int n, int size, float[] binary)
        {
            int i, f;
            for (f = 0; f < n; ++f)
            {
                float mean = 0;
                for (i = 0; i < size; ++i)
                {
                    mean += (float)Math.Abs(weights[f * size + i]);
                }
                mean = mean / size;
                for (i = 0; i < size; ++i)
                {
                    binary[f * size + i] = (weights[f * size + i] > 0) ? mean : -mean;
                }
            }
        }

        void binarize_cpu(float[] input, int n, float[] binary)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                binary[i] = (input[i] > 0) ? 1 : -1;
            }
        }

        void binarize_input(float[] input, int n, int size, float[] binary)
        {
            int i, s;
            for (s = 0; s < size; ++s)
            {
                float mean = 0;
                for (i = 0; i < n; ++i)
                {
                    mean += (float)Math.Abs(input[i * size + s]);
                }
                mean = mean / n;
                for (i = 0; i < n; ++i)
                {
                    binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
                }
            }
        }

        int convolutional_out_height(Layer l)
        {
            return (l.H + 2 * l.Pad - l.Size) / l.Stride + 1;
        }

        int convolutional_out_width(Layer l)
        {
            return (l.W + 2 * l.Pad - l.Size) / l.Stride + 1;
        }

        Image get_convolutional_image(Layer l)
        {
            int h, w, c;
            h = convolutional_out_height(l);
            w = convolutional_out_width(l);
            c = l.N;
            return new Image(w, h, c, l.Output);
        }

        Image get_convolutional_delta(Layer l)
        {
            int h, w, c;
            h = convolutional_out_height(l);
            w = convolutional_out_width(l);
            c = l.N;
            return new Image(w, h, c, l.Delta);
        }

        int get_workspace_size(Layer l)
        {
            if (CudaUtils.UseGpu)
            {
                size_t most = 0;
                size_t s = 0;
                cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                        l.srcTensorDesc,
                        l.weightDesc,
                        l.convDesc,
                        l.dstTensorDesc,
                        l.fw_algo,
                        &s);
                if (s > most) most = s;
                cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                        l.srcTensorDesc,
                        l.ddstTensorDesc,
                        l.convDesc,
                        l.dweightDesc,
                        l.bf_algo,
                        &s);
                if (s > most) most = s;
                cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                        l.weightDesc,
                        l.ddstTensorDesc,
                        l.convDesc,
                        l.dsrcTensorDesc,
                        l.bd_algo,
                        &s);
                if (s > most) most = s;
                return most;
            }
            return (int)l.OutH * l.OutW * l.Size * l.Size * l.C * sizeof(float);
        }

        void cudnn_convolutional_setup(Layer l)
        {
            cudnnSetTensor4dDescriptor(l.dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.Batch, l.C, l.H, l.W);
            cudnnSetTensor4dDescriptor(l.ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.Batch, l.OutC, l.OutH, l.OutW);
            cudnnSetFilter4dDescriptor(l.dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l.N, l.C, l.Size, l.Size);

            cudnnSetTensor4dDescriptor(l.srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.Batch, l.C, l.H, l.W);
            cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.Batch, l.OutC, l.OutH, l.OutW);
            cudnnSetFilter4dDescriptor(l.weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l.N, l.C, l.Size, l.Size);
            cudnnSetConvolution2dDescriptor(l.convDesc, l.Pad, l.Pad, l.Stride, l.Stride, 1, 1, CUDNN_CROSS_CORRELATION);
            cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
                    l.srcTensorDesc,
                    l.weightDesc,
                    l.convDesc,
                    l.dstTensorDesc,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0,
                    &l.fw_algo);
            cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
                    l.weightDesc,
                    l.ddstTensorDesc,
                    l.convDesc,
                    l.dsrcTensorDesc,
                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                    0,
                    &l.bd_algo);
            cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
                    l.srcTensorDesc,
                    l.ddstTensorDesc,
                    l.convDesc,
                    l.dweightDesc,
                    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                    0,
                    &l.bf_algo);
        }

        Layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, Activation activation, int batch_normalize, int binary, int xnor, int adam)
        {
            int i;
            Layer l = new Layer();
            l.LayerType = CONVOLUTIONAL;

            l.H = h;
            l.W = w;
            l.C = c;
            l.N = n;
            l.Binary = binary;
            l.Xnor = xnor;
            l.Batch = batch;
            l.Stride = stride;
            l.Size = size;
            l.Pad = padding;
            l.BatchNormalize = batch_normalize;

            l.Weights = calloc(c * n * size * size, sizeof(float));
            l.WeightUpdates = calloc(c * n * size * size, sizeof(float));

            l.Biases = calloc(n, sizeof(float));
            l.BiasUpdates = calloc(n, sizeof(float));

            // float scale = 1./(float)Math.Sqrt(size*size*c);
            float scale = (float)Math.Sqrt(2./ (size * size * c));
            for (i = 0; i < c * n * size * size; ++i) l.Weights[i] = scale * rand_uniform(-1, 1);
            int out_h = convolutional_out_height(l);
            int out_w = convolutional_out_width(l);
            l.OutH = out_h;
            l.OutW = out_w;
            l.OutC = n;
            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = l.W * l.H * l.C;

            l.Output = calloc(l.Batch * l.Outputs, sizeof(float));
            l.Delta = calloc(l.Batch * l.Outputs, sizeof(float));

            l.Forward = forward_convolutional_layer;
            l.Backward = backward_convolutional_layer;
            l.Update = update_convolutional_layer;
            if (binary)
            {
                l.BinaryWeights = calloc(c * n * size * size, sizeof(float));
                l.Cweights = calloc(c * n * size * size, sizeof(char));
                l.Scales = calloc(n, sizeof(float));
            }
            if (xnor)
            {
                l.BinaryWeights = calloc(c * n * size * size, sizeof(float));
                l.BinaryInput = calloc(l.Inputs * l.Batch, sizeof(float));
            }

            if (batch_normalize)
            {
                l.Scales = calloc(n, sizeof(float));
                l.scale_updates = calloc(n, sizeof(float));
                for (i = 0; i < n; ++i)
                {
                    l.Scales[i] = 1;
                }

                l.Mean = calloc(n, sizeof(float));
                l.variance = calloc(n, sizeof(float));

                l.MeanDelta = calloc(n, sizeof(float));
                l.variance_delta = calloc(n, sizeof(float));

                l.RollingMean = calloc(n, sizeof(float));
                l.RollingVariance = calloc(n, sizeof(float));
                l.X = calloc(l.Batch * l.Outputs, sizeof(float));
                l.XNorm = calloc(l.Batch * l.Outputs, sizeof(float));
            }
            if (adam)
            {
                l.adam = 1;
                l.m = calloc(c * n * size * size, sizeof(float));
                l.v = calloc(c * n * size * size, sizeof(float));
            }

            l.ForwardGpu = forward_convolutional_layer_gpu;
            l.BackwardGpu = backward_convolutional_layer_gpu;
            l.UpdateGpu = update_convolutional_layer_gpu;

            if (CudaUtils.UseGpu)
            {
                if (adam)
                {
                    l.m_gpu = cuda_make_array(l.m, c * n * size * size);
                    l.v_gpu = cuda_make_array(l.v, c * n * size * size);
                }

                l.WeightsGpu = cuda_make_array(l.Weights, c * n * size * size);
                l.WeightUpdatesGpu = cuda_make_array(l.WeightUpdates, c * n * size * size);

                l.BiasesGpu = cuda_make_array(l.Biases, n);
                l.BiasUpdatesGpu = cuda_make_array(l.BiasUpdates, n);

                l.DeltaGpu = cuda_make_array(l.Delta, l.Batch * out_h * out_w * n);
                l.OutputGpu = cuda_make_array(l.Output, l.Batch * out_h * out_w * n);

                if (binary)
                {
                    l.BinaryWeightsGpu = cuda_make_array(l.Weights, c * n * size * size);
                }
                if (xnor)
                {
                    l.BinaryWeightsGpu = cuda_make_array(l.Weights, c * n * size * size);
                    l.binary_input_gpu = cuda_make_array(0, l.Inputs * l.Batch);
                }

                if (batch_normalize)
                {
                    l.MeanGpu = cuda_make_array(l.Mean, n);
                    l.variance_gpu = cuda_make_array(l.variance, n);

                    l.RollingMeanGpu = cuda_make_array(l.Mean, n);
                    l.rolling_variance_gpu = cuda_make_array(l.variance, n);

                    l.MeanDeltaGpu = cuda_make_array(l.Mean, n);
                    l.variance_delta_gpu = cuda_make_array(l.variance, n);

                    l.scales_gpu = cuda_make_array(l.Scales, n);
                    l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

                    l.XGpu = cuda_make_array(l.Output, l.Batch * out_h * out_w * n);
                    l.XNormGpu = cuda_make_array(l.Output, l.Batch * out_h * out_w * n);
                }
                cudnnCreateTensorDescriptor(&l.srcTensorDesc);
                cudnnCreateTensorDescriptor(&l.dstTensorDesc);
                cudnnCreateFilterDescriptor(&l.weightDesc);
                cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
                cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
                cudnnCreateFilterDescriptor(&l.dweightDesc);
                cudnnCreateConvolutionDescriptor(&l.convDesc);
                cudnn_convolutional_setup(&l);
            }
            l.WorkspaceSize = get_workspace_size(l);
            l.Activation = activation;

            Console.Error.Write($"conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   .  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.OutW, l.OutH, l.OutC);

            return l;
        }

        void denormalize_convolutional_layer(Layer l)
        {
            int i, j;
            for (i = 0; i < l.N; ++i)
            {
                float scale = l.Scales[i] / (float)Math.Sqrt(l.RollingVariance[i] + .00001);
                for (j = 0; j < l.C * l.Size * l.Size; ++j)
                {
                    l.Weights[i * l.C * l.Size * l.Size + j] *= scale;
                }
                l.Biases[i] -= l.RollingMean[i] * scale;
                l.Scales[i] = 1;
                l.RollingMean[i] = 0;
                l.RollingVariance[i] = 1;
            }
        }

        void test_convolutional_layer()
        {
            Layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
            l.BatchNormalize = 1;
            float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
            NetworkState state = new Layer();
            state.Input = data;
            forward_convolutional_layer(l, state);
        }

        void resize_convolutional_layer(Layer* l, int w, int h)
        {
            l.W = w;
            l.H = h;
            int out_w = convolutional_out_width(*l);
            int out_h = convolutional_out_height(*l);

            l.OutW = out_w;
            l.OutH = out_h;

            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = l.W * l.H * l.C;

            l.Output = realloc(l.Output, l.Batch * l.Outputs * sizeof(float));
            l.Delta = realloc(l.Delta, l.Batch * l.Outputs * sizeof(float));
            if (l.BatchNormalize)
            {
                l.X = realloc(l.X, l.Batch * l.Outputs * sizeof(float));
                l.XNorm = realloc(l.XNorm, l.Batch * l.Outputs * sizeof(float));
            }

            cuda_free(l.DeltaGpu);
            cuda_free(l.OutputGpu);

            l.DeltaGpu = cuda_make_array(l.Delta, l.Batch * l.Outputs);
            l.OutputGpu = cuda_make_array(l.Output, l.Batch * l.Outputs);

            if (l.BatchNormalize)
            {
                cuda_free(l.XGpu);
                cuda_free(l.XNormGpu);

                l.XGpu = cuda_make_array(l.Output, l.Batch * l.Outputs);
                l.XNormGpu = cuda_make_array(l.Output, l.Batch * l.Outputs);
            }
            cudnn_convolutional_setup(l);
            l.WorkspaceSize = get_workspace_size(*l);
        }

        void add_bias(float[] output, float[] biases, int batch, int n, int size)
        {
            int i, j, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    for (j = 0; j < size; ++j)
                    {
                        output[(b * n + i) * size + j] += biases[i];
                    }
                }
            }
        }

        void scale_bias(float[] output, float[] scales, int batch, int n, int size)
        {
            int i, j, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    for (j = 0; j < size; ++j)
                    {
                        output[(b * n + i) * size + j] *= scales[i];
                    }
                }
            }
        }

        void backward_bias(float[] bias_updates, float[] delta, int batch, int n, int size)
        {
            int i, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    bias_updates[i] += sum_array(delta + size * (i + b * n), size);
                }
            }
        }

        void forward_convolutional_layer(Layer l, NetworkState state)
        {
            int out_h = convolutional_out_height(l);
            int out_w = convolutional_out_width(l);
            int i;

            Blas.Fill_cpu(l.Outputs * l.Batch, 0, l.Output, 1);

            if (l.Xnor)
            {
                binarize_weights(l.Weights, l.N, l.C * l.Size * l.Size, l.BinaryWeights);
                swap_binary(&l);
                binarize_cpu(state.Input, l.C * l.H * l.W * l.Batch, l.BinaryInput);
                state.Input = l.BinaryInput;
            }

            int m = l.N;
            int k = l.Size * l.Size * l.C;
            int n = out_h * out_w;


            float[] a = l.Weights;
            float[] b = state.Workspace;
            float[] c = l.Output;

            for (i = 0; i < l.Batch; ++i)
            {
                Im2Col.im2col_cpu(state.Input, l.C, l.H, l.W,
                        l.Size, l.Stride, l.Pad, b);
                Gemm.gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                c += n * m;
                state.Input += l.C * l.H * l.W;
            }

            if (l.BatchNormalize)
            {
                forward_batchnorm_layer(l, state);
            }
            add_bias(l.Output, l.Biases, l.Batch, l.N, out_h * out_w);

            activate_array(l.Output, m * n * l.Batch, l.Activation);
            if (l.Binary || l.Xnor) swap_binary(&l);
        }

        void backward_convolutional_layer(Layer l, NetworkState state)
        {
            int i;
            int m = l.N;
            int n = l.Size * l.Size * l.C;
            int k = convolutional_out_height(l) *
                convolutional_out_width(l);

            gradient_array(l.Output, m * k * l.Batch, l.Activation, l.Delta);
            backward_bias(l.BiasUpdates, l.Delta, l.Batch, l.N, k);

            if (l.BatchNormalize)
            {
                backward_batchnorm_layer(l, state);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] a = l.Delta + i * m * k;
                float[] b = state.Workspace;
                float[] c = l.WeightUpdates;

                float[] im = state.Input + i * l.C * l.H * l.W;

                Im2Col.im2col_cpu(im, l.C, l.H, l.W,
                        l.Size, l.Stride, l.Pad, b);
                Gemm.gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

                if (state.Delta)
                {
                    a = l.Weights;
                    b = l.Delta + i * m * k;
                    c = state.Workspace;

                    Gemm.gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

                    Im2Col.col2im_cpu(state.Workspace, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, state.Delta + i * l.C * l.H * l.W);
                }
            }
        }

        void update_convolutional_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int size = l.Size * l.Size * l.C * l.N;
            Blas.Axpy_cpu(l.N, learning_rate / batch, l.BiasUpdates, 1, l.Biases, 1);
            Blas.Scal_cpu(l.N, momentum, l.BiasUpdates, 1);

            if (l.Scales)
            {
                Blas.Axpy_cpu(l.N, learning_rate / batch, l.scale_updates, 1, l.Scales, 1);
                Blas.Scal_cpu(l.N, momentum, l.scale_updates, 1);
            }

            Blas.Axpy_cpu(size, -decay * batch, l.Weights, 1, l.WeightUpdates, 1);
            Blas.Axpy_cpu(size, learning_rate / batch, l.WeightUpdates, 1, l.Weights, 1);
            Blas.Scal_cpu(size, momentum, l.WeightUpdates, 1);
        }


        Image get_convolutional_weight(Layer l, int i)
        {
            int h = l.Size;
            int w = l.Size;
            int c = l.C;
            return new Image(w, h, c, l.Weights + i * h * w * c);
        }

        void rgbgr_weights(Layer l)
        {
            int i;
            for (i = 0; i < l.N; ++i)
            {
                Image im = get_convolutional_weight(l, i);
                if (im.C == 3)
                {
                    rgbgr_image(im);
                }
            }
        }

        void rescale_weights(Layer l, float scale, float trans)
        {
            int i;
            for (i = 0; i < l.N; ++i)
            {
                Image im = get_convolutional_weight(l, i);
                if (im.C == 3)
                {
                    scale_image(im, scale);
                    float sum = sum_array(im.data, im.W * im.H * im.C);
                    l.Biases[i] += sum * trans;
                }
            }
        }

        Image* get_weights(Layer l)
        {
            Image* weights = calloc(l.N, sizeof(Image));
            int i;
            for (i = 0; i < l.N; ++i)
            {
                weights[i] = copy_image(get_convolutional_weight(l, i));
                //normalize_image(weights[i]);
            }
            return weights;
        }

        Image* visualize_convolutional_layer(Layer l, string window, Image* prev_weights)
        {
            Image* single_weights = get_weights(l);
            show_images(single_weights, l.N, window);

            Image delta = get_convolutional_image(l);
            Image dc = collapse_image_layers(delta, 1);
            string buff = $"{window}: Output";
            return single_weights;
        }

        Layer make_activation_layer(int batch, int inputs, Activation activation)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Active;

            l.Inputs = inputs;
            l.Outputs = inputs;
            l.Batch = batch;

            l.Output = calloc(batch * inputs, sizeof(float[]));
            l.Delta = calloc(batch * inputs, sizeof(float[]));

            l.Forward = forward_activation_layer;
            l.Backward = backward_activation_layer;
            l.ForwardGpu = forward_activation_layer_gpu;
            l.BackwardGpu = backward_activation_layer_gpu;

            l.OutputGpu = cuda_make_array(l.Output, inputs * batch);
            l.DeltaGpu = cuda_make_array(l.Delta, inputs * batch);
            l.Activation = activation;
            Console.Error.Write($"Activation Layer: %d inputs\n", inputs);
            return l;
        }

        void forward_activation_layer(Layer l, NetworkState state)
        {
            Blas.Copy_cpu(l.Outputs * l.Batch, state.Input, 1, l.Output, 1);
            activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        void backward_activation_layer(Layer l, NetworkState state)
        {
            gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);
            Blas.Copy_cpu(l.Outputs * l.Batch, l.Delta, 1, state.Delta, 1);
        }

        void forward_activation_layer_gpu(Layer l, NetworkState state)
        {
            Blas.copy_ongpu(l.Outputs * l.Batch, state.Input, 1, l.OutputGpu, 1);
            activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }

        void backward_activation_layer_gpu(Layer l, NetworkState state)
        {
            gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            Blas.copy_ongpu(l.Outputs * l.Batch, l.DeltaGpu, 1, state.Delta, 1);
        }
        Layer make_avgpool_layer(int batch, int w, int h, int c)
        {
            Console.Error.Write($"avg                     %4d x%4d x%4d   .  %4d\n", w, h, c, c);
            Layer l = new Layer();
            l.LayerType = LayerType.Avgpool;
            l.Batch = batch;
            l.H = h;
            l.W = w;
            l.C = c;
            l.OutW = 1;
            l.OutH = 1;
            l.OutC = c;
            l.Outputs = l.OutC;
            l.Inputs = h * w * c;
            int output_size = l.Outputs * batch;
            l.Output = calloc(output_size, sizeof(float));
            l.Delta = calloc(output_size, sizeof(float));
            l.Forward = forward_avgpool_layer;
            l.Backward = backward_avgpool_layer;
            l.ForwardGpu = forward_avgpool_layer_gpu;
            l.BackwardGpu = backward_avgpool_layer_gpu;
            l.OutputGpu = cuda_make_array(l.Output, output_size);
            l.DeltaGpu = cuda_make_array(l.Delta, output_size);
            return l;
        }

        void resize_avgpool_layer(Layer l, int w, int h)
        {
            l.W = w;
            l.H = h;
            l.Inputs = h * w * l.C;
        }

        void forward_avgpool_layer(Layer l, NetworkState state)
        {
            int b, i, k;

            for (b = 0; b < l.Batch; ++b)
            {
                for (k = 0; k < l.C; ++k)
                {
                    int out_index = k + b * l.C;
                    l.Output[out_index] = 0;
                    for (i = 0; i < l.H * l.W; ++i)
                    {
                        int in_index = i + l.H * l.W * (k + b * l.C);
                        l.Output[out_index] += state.Input[in_index];
                    }
                    l.Output[out_index] /= l.H * l.W;
                }
            }
        }

        void backward_avgpool_layer(Layer l, NetworkState state)
        {
            int b, i, k;

            for (b = 0; b < l.Batch; ++b)
            {
                for (k = 0; k < l.C; ++k)
                {
                    int out_index = k + b * l.C;
                    for (i = 0; i < l.H * l.W; ++i)
                    {
                        int in_index = i + l.H * l.W * (k + b * l.C);
                        state.Delta[in_index] += l.Delta[out_index] / (l.H * l.W);
                    }
                }
            }
        }
        Layer make_connected_layer(int batch, int inputs, int outputs, Activation activation, int batch_normalize)
        {
            int i;
            Layer l = new Layer();
            l.LayerType = LayerType.Connected;

            l.Inputs = inputs;
            l.Outputs = outputs;
            l.Batch = batch;
            l.BatchNormalize = batch_normalize;
            l.H = 1;
            l.W = 1;
            l.C = inputs;
            l.OutH = 1;
            l.OutW = 1;
            l.OutC = outputs;

            l.Output = calloc(batch * outputs, sizeof(float));
            l.Delta = calloc(batch * outputs, sizeof(float));

            l.WeightUpdates = calloc(inputs * outputs, sizeof(float));
            l.BiasUpdates = calloc(outputs, sizeof(float));

            l.Weights = calloc(outputs * inputs, sizeof(float));
            l.Biases = calloc(outputs, sizeof(float));

            l.Forward = forward_connected_layer;
            l.Backward = backward_connected_layer;
            l.Update = update_connected_layer;

            //float scale = 1./(float)Math.Sqrt(inputs);
            float scale = (float)Math.Sqrt(2./ inputs);
            for (i = 0; i < outputs * inputs; ++i)
            {
                l.Weights[i] = scale * rand_uniform(-1, 1);
            }

            for (i = 0; i < outputs; ++i)
            {
                l.Biases[i] = 0;
            }

            if (batch_normalize)
            {
                l.Scales = calloc(outputs, sizeof(float));
                l.scale_updates = calloc(outputs, sizeof(float));
                for (i = 0; i < outputs; ++i)
                {
                    l.Scales[i] = 1;
                }

                l.Mean = calloc(outputs, sizeof(float));
                l.MeanDelta = calloc(outputs, sizeof(float));
                l.variance = calloc(outputs, sizeof(float));
                l.variance_delta = calloc(outputs, sizeof(float));

                l.RollingMean = calloc(outputs, sizeof(float));
                l.RollingVariance = calloc(outputs, sizeof(float));

                l.X = calloc(batch * outputs, sizeof(float));
                l.XNorm = calloc(batch * outputs, sizeof(float));
            }

            l.ForwardGpu = forward_connected_layer_gpu;
            l.BackwardGpu = backward_connected_layer_gpu;
            l.UpdateGpu = update_connected_layer_gpu;

            l.WeightsGpu = cuda_make_array(l.Weights, outputs * inputs);
            l.BiasesGpu = cuda_make_array(l.Biases, outputs);

            l.WeightUpdatesGpu = cuda_make_array(l.WeightUpdates, outputs * inputs);
            l.BiasUpdatesGpu = cuda_make_array(l.BiasUpdates, outputs);

            l.OutputGpu = cuda_make_array(l.Output, outputs * batch);
            l.DeltaGpu = cuda_make_array(l.Delta, outputs * batch);
            if (batch_normalize)
            {
                l.scales_gpu = cuda_make_array(l.Scales, outputs);
                l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

                l.MeanGpu = cuda_make_array(l.Mean, outputs);
                l.variance_gpu = cuda_make_array(l.variance, outputs);

                l.RollingMeanGpu = cuda_make_array(l.Mean, outputs);
                l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

                l.MeanDeltaGpu = cuda_make_array(l.Mean, outputs);
                l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

                l.XGpu = cuda_make_array(l.Output, l.Batch * outputs);
                l.XNormGpu = cuda_make_array(l.Output, l.Batch * outputs);
            }
            l.Activation = activation;
            Console.Error.Write($"connected                            %4d  .  %4d\n", inputs, outputs);
            return l;
        }

        void update_connected_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            Blas.Axpy_cpu(l.Outputs, learning_rate / batch, l.BiasUpdates, 1, l.Biases, 1);
            Blas.Scal_cpu(l.Outputs, momentum, l.BiasUpdates, 1);

            if (l.BatchNormalize)
            {
                Blas.Axpy_cpu(l.Outputs, learning_rate / batch, l.scale_updates, 1, l.Scales, 1);
                Blas.Scal_cpu(l.Outputs, momentum, l.scale_updates, 1);
            }

            Blas.Axpy_cpu(l.Inputs * l.Outputs, -decay * batch, l.Weights, 1, l.WeightUpdates, 1);
            Blas.Axpy_cpu(l.Inputs * l.Outputs, learning_rate / batch, l.WeightUpdates, 1, l.Weights, 1);
            Blas.Scal_cpu(l.Inputs * l.Outputs, momentum, l.WeightUpdates, 1);
        }

        void forward_connected_layer(Layer l, NetworkState state)
        {
            int i;
            Blas.Fill_cpu(l.Outputs * l.Batch, 0, l.Output, 1);
            int m = l.Batch;
            int k = l.Inputs;
            int n = l.Outputs;
            float[] a = state.Input;
            float[] b = l.Weights;
            float[] c = l.Output;
            Gemm.gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
            if (l.BatchNormalize)
            {
                if (state.Train)
                {
                    mean_cpu(l.Output, l.Batch, l.Outputs, 1, l.Mean);
                    variance_cpu(l.Output, l.Mean, l.Batch, l.Outputs, 1, l.variance);

                    Blas.Scal_cpu(l.Outputs, .95, l.RollingMean, 1);
                    Blas.Axpy_cpu(l.Outputs, .05, l.Mean, 1, l.RollingMean, 1);
                    Blas.Scal_cpu(l.Outputs, .95, l.RollingVariance, 1);
                    Blas.Axpy_cpu(l.Outputs, .05, l.variance, 1, l.RollingVariance, 1);

                    Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, 1, l.X, 1);
                    normalize_cpu(l.Output, l.Mean, l.variance, l.Batch, l.Outputs, 1);
                    Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, 1, l.XNorm, 1);
                }
                else
                {
                    normalize_cpu(l.Output, l.RollingMean, l.RollingVariance, l.Batch, l.Outputs, 1);
                }
                scale_bias(l.Output, l.Scales, l.Batch, l.Outputs, 1);
            }
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.Biases, 1, l.Output + i * l.Outputs, 1);
            }
            activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        void backward_connected_layer(Layer l, NetworkState state)
        {
            int i;
            gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.Delta + i * l.Outputs, 1, l.BiasUpdates, 1);
            }
            if (l.BatchNormalize)
            {
                backward_scale_cpu(l.XNorm, l.Delta, l.Batch, l.Outputs, 1, l.scale_updates);

                scale_bias(l.Delta, l.Scales, l.Batch, l.Outputs, 1);

                mean_delta_cpu(l.Delta, l.variance, l.Batch, l.Outputs, 1, l.MeanDelta);
                variance_delta_cpu(l.X, l.Delta, l.Mean, l.variance, l.Batch, l.Outputs, 1, l.variance_delta);
                normalize_delta_cpu(l.X, l.Mean, l.variance, l.MeanDelta, l.variance_delta, l.Batch, l.Outputs, 1, l.Delta);
            }

            int m = l.Outputs;
            int k = l.Batch;
            int n = l.Inputs;
            float[] a = l.Delta;
            float[] b = state.Input;
            float[] c = l.WeightUpdates;
            Gemm.gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

            m = l.Batch;
            k = l.Outputs;
            n = l.Inputs;

            a = l.Delta;
            b = l.Weights;
            c = state.Delta;

            if (c) Gemm.gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }


        void denormalize_connected_layer(Layer l)
        {
            int i, j;
            for (i = 0; i < l.Outputs; ++i)
            {
                float scale = l.Scales[i] / (float)Math.Sqrt(l.RollingVariance[i] + .000001);
                for (j = 0; j < l.Inputs; ++j)
                {
                    l.Weights[i * l.Inputs + j] *= scale;
                }
                l.Biases[i] -= l.RollingMean[i] * scale;
                l.Scales[i] = 1;
                l.RollingMean[i] = 0;
                l.RollingVariance[i] = 1;
            }
        }


        void statistics_connected_layer(Layer l)
        {
            if (l.BatchNormalize)
            {
                Console.Write($"Scales ");
                print_statistics(l.Scales, l.Outputs);
                /*
                Console.Write($"Rolling Mean ");
                print_statistics(l.RollingMean, l.Outputs);
                Console.Write($"Rolling Variance ");
                print_statistics(l.RollingVariance, l.Outputs);
                */
            }
            Console.Write($"Biases ");
            print_statistics(l.Biases, l.Outputs);
            Console.Write($"Weights ");
            print_statistics(l.Weights, l.Outputs);
        }

        void pull_connected_layer(Layer l)
        {
            Array.Copy(l.WeightsGpu, l.Weights, l.Inputs * l.Outputs);
            Array.Copy(l.BiasesGpu, l.Biases, l.Outputs);
            Array.Copy(l.WeightUpdatesGpu, l.WeightUpdates, l.Inputs * l.Outputs);
            Array.Copy(l.BiasUpdatesGpu, l.BiasUpdates, l.Outputs);
            if (l.BatchNormalize)
            {
                Array.Copy(l.scales_gpu, l.Scales, l.Outputs);
                Array.Copy(l.RollingMeanGpu, l.RollingMean, l.Outputs);
                Array.Copy(l.rolling_variance_gpu, l.RollingVariance, l.Outputs);
            }
        }

        void push_connected_layer(Layer l)
        {
            cuda_push_array(l.WeightsGpu, l.Weights, l.Inputs * l.Outputs);
            cuda_push_array(l.BiasesGpu, l.Biases, l.Outputs);
            cuda_push_array(l.WeightUpdatesGpu, l.WeightUpdates, l.Inputs * l.Outputs);
            cuda_push_array(l.BiasUpdatesGpu, l.BiasUpdates, l.Outputs);
            if (l.BatchNormalize)
            {
                cuda_push_array(l.scales_gpu, l.Scales, l.Outputs);
                cuda_push_array(l.RollingMeanGpu, l.RollingMean, l.Outputs);
                cuda_push_array(l.rolling_variance_gpu, l.RollingVariance, l.Outputs);
            }
        }

        void update_connected_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            Blas.axpy_ongpu(l.Outputs, learning_rate / batch, l.BiasUpdatesGpu, 1, l.BiasesGpu, 1);
            Blas.scal_ongpu(l.Outputs, momentum, l.BiasUpdatesGpu, 1);

            if (l.BatchNormalize)
            {
                Blas.axpy_ongpu(l.Outputs, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
                Blas.scal_ongpu(l.Outputs, momentum, l.scale_updates_gpu, 1);
            }

            Blas.axpy_ongpu(l.Inputs * l.Outputs, -decay * batch, l.WeightsGpu, 1, l.WeightUpdatesGpu, 1);
            Blas.axpy_ongpu(l.Inputs * l.Outputs, learning_rate / batch, l.WeightUpdatesGpu, 1, l.WeightsGpu, 1);
            Blas.scal_ongpu(l.Inputs * l.Outputs, momentum, l.WeightUpdatesGpu, 1);
        }

        void forward_connected_layer_gpu(Layer l, NetworkState state)
        {
            int i;
            fill_ongpu(l.Outputs * l.Batch, 0, l.OutputGpu, 1);

            int m = l.Batch;
            int k = l.Inputs;
            int n = l.Outputs;
            float[] a = state.Input;
            float[] b = l.WeightsGpu;
            float[] c = l.OutputGpu;
            Gemm.gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
            if (l.BatchNormalize)
            {
                forward_batchnorm_layer_gpu(l, state);
            }
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.axpy_ongpu(l.Outputs, 1, l.BiasesGpu, 1, l.OutputGpu + i * l.Outputs, 1);
            }
            activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }

        void backward_connected_layer_gpu(Layer l, NetworkState state)
        {
            int i;
            constrain_ongpu(l.Outputs * l.Batch, 1, l.DeltaGpu, 1);
            gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.axpy_ongpu(l.Outputs, 1, l.DeltaGpu + i * l.Outputs, 1, l.BiasUpdatesGpu, 1);
            }

            if (l.BatchNormalize)
            {
                backward_batchnorm_layer_gpu(l, state);
            }

            int m = l.Outputs;
            int k = l.Batch;
            int n = l.Inputs;
            float[] a = l.DeltaGpu;
            float[] b = state.Input;
            float[] c = l.WeightUpdatesGpu;
            Gemm.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

            m = l.Batch;
            k = l.Outputs;
            n = l.Inputs;

            a = l.DeltaGpu;
            b = l.WeightsGpu;
            c = state.Delta;

            if (c) Gemm.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
        CostType get_cost_type(string s)
        {
            if (strcmp(s, "sse") == 0) return CostType.Sse;
            if (strcmp(s, "masked") == 0) return CostType.Masked;
            if (strcmp(s, "smooth") == 0) return CostType.Smooth;
            Console.Error.Write($"Couldn't find cost type %s, going with CostType.Sse\n", s);
            return CostType.Sse;
        }

        string get_cost_string(CostType a)
        {
            switch (a)
            {
                case CostType.Sse:
                    return "sse";
                case CostType.Masked:
                    return "masked";
                case CostType.Smooth:
                    return "smooth";
            }
            return "sse";
        }

        Layer make_cost_layer(int batch, int inputs, CostType cost_type, float scale)
        {
            Console.Error.Write($"cost                                           %4d\n", inputs);
            Layer l = new Layer();
            l.LayerType = LayerType.Cost;

            l.Scale = scale;
            l.Batch = batch;
            l.Inputs = inputs;
            l.Outputs = inputs;
            l.CostType = cost_type;
            l.Delta = calloc(inputs * batch, sizeof(float));
            l.Output = calloc(inputs * batch, sizeof(float));
            l.Cost = calloc(1, sizeof(float));

            l.Forward = forward_cost_layer;
            l.Backward = backward_cost_layer;

            l.ForwardGpu = forward_cost_layer_gpu;
            l.BackwardGpu = backward_cost_layer_gpu;

            l.DeltaGpu = cuda_make_array(l.Output, inputs * batch);
            l.OutputGpu = cuda_make_array(l.Delta, inputs * batch);

            return l;
        }

        void resize_cost_layer(Layer l, int inputs)
        {
            l.Inputs = inputs;
            l.Outputs = inputs;
            l.Delta = realloc(l.Delta, inputs * l.Batch * sizeof(float));
            l.Output = realloc(l.Output, inputs * l.Batch * sizeof(float));

            cuda_free(l.DeltaGpu);
            cuda_free(l.OutputGpu);
            l.DeltaGpu = cuda_make_array(l.Delta, inputs * l.Batch);
            l.OutputGpu = cuda_make_array(l.Output, inputs * l.Batch);

        }

        void forward_cost_layer(Layer l, NetworkState state)
        {
            if (!state.Truth.Any()) return;
            if (l.CostType == CostType.Masked)
            {
                int i;
                for (i = 0; i < l.Batch * l.Inputs; ++i)
                {
                    if (state.Truth[i] == SECRET_NUM) state.Input[i] = SECRET_NUM;
                }
            }
            if (l.CostType == CostType.Smooth)
            {
                smooth_l1_cpu(l.Batch * l.Inputs, state.Input, state.Truth, l.Delta, l.Output);
            }
            else
            {
                l2_cpu(l.Batch * l.Inputs, state.Input, state.Truth, l.Delta, l.Output);
            }
            l.Cost[0] = sum_array(l.Output, l.Batch * l.Inputs);
        }

        void backward_cost_layer(Layer l, NetworkState state)
        {
            Blas.Axpy_cpu(l.Batch * l.Inputs, l.Scale, l.Delta, 1, state.Delta, 1);
        }



        void pull_cost_layer(Layer l)
        {
            Array.Copy(l.DeltaGpu, l.Delta, l.Batch * l.Inputs);
        }

        void push_cost_layer(Layer l)
        {
            cuda_push_array(l.DeltaGpu, l.Delta, l.Batch * l.Inputs);
        }

        int float_abs_compare(void* a, void* b)
        {
            float fa = *(float[])a;
            if (fa < 0) fa = -fa;
            float fb = *(float[])b;
            if (fb < 0) fb = -fb;
            return (fa > fb) - (fa < fb);
        }

        void forward_cost_layer_gpu(Layer l, NetworkState state)
        {
            if (!state.Truth) return;
            if (l.CostType == CostType.Masked)
            {
                mask_ongpu(l.Batch * l.Inputs, state.Input, SECRET_NUM, state.Truth);
            }

            if (l.CostType == CostType.Smooth)
            {
                smooth_l1_gpu(l.Batch * l.Inputs, state.Input, state.Truth, l.DeltaGpu, l.OutputGpu);
            }
            else
            {
                l2_gpu(l.Batch * l.Inputs, state.Input, state.Truth, l.DeltaGpu, l.OutputGpu);
            }

            if (l.Ratio)
            {
                Array.Copy(l.DeltaGpu, l.Delta, l.Batch * l.Inputs);
                qsort(l.Delta, l.Batch * l.Inputs, sizeof(float), float_abs_compare);
                int n = (1 - l.Ratio) * l.Batch * l.Inputs;
                float thresh = l.Delta[n];
                thresh = 0;
                Console.Write($"%f\n", thresh);
                supp_ongpu(l.Batch * l.Inputs, thresh, l.DeltaGpu, 1);
            }

            Array.Copy(l.OutputGpu, l.Output, l.Batch * l.Inputs);
            l.Cost[0] = sum_array(l.Output, l.Batch * l.Inputs);
        }

        void backward_cost_layer_gpu(Layer l, NetworkState state)
        {
            Blas.axpy_ongpu(l.Batch * l.Inputs, l.Scale, l.DeltaGpu, 1, state.Delta, 1);
        }
        static void increment_layer(Layer l, int steps)
        {
            int num = l.Outputs * l.Batch * steps;
            l.Output += num;
            l.Delta += num;
            l.X += num;
            l.XNorm += num;


            l.OutputGpu += num;
            l.DeltaGpu += num;
            l.XGpu += num;
            l.XNormGpu += num;

        }

        Layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, Activation activation, int batch_normalize)
        {
            Console.Error.Write($"LayerType.Crnn Layer: %d x %d x %d Image, %d filters\n", h, w, c, output_filters);
            batch = batch / steps;
            Layer l = new Layer();
            l.Batch = batch;
            l.LayerType = LayerType.Crnn;
            l.Steps = steps;
            l.H = h;
            l.W = w;
            l.C = c;
            l.OutH = h;
            l.OutW = w;
            l.OutC = output_filters;
            l.Inputs = h * w * c;
            l.Hidden = h * w * hidden_filters;
            l.Outputs = l.OutH * l.OutW * l.OutC;

            l.State = calloc(l.Hidden * batch * (steps + 1), sizeof(float));

            l.InputLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.InputLayer) = make_convolutional_layer(batch * steps, h, w, c, hidden_filters, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
            l.InputLayer.Batch = batch;

            l.InputLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.InputLayer) = make_convolutional_layer(batch * steps, h, w, hidden_filters, hidden_filters, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
            l.InputLayer.Batch = batch;

            l.OutputLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.OutputLayer) = make_convolutional_layer(batch * steps, h, w, hidden_filters, output_filters, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
            l.OutputLayer.Batch = batch;

            l.Output = l.OutputLayer.Output;
            l.Delta = l.OutputLayer.Delta;

            l.Forward = forward_crnn_layer;
            l.Backward = backward_crnn_layer;
            l.Update = update_crnn_layer;


            l.ForwardGpu = forward_crnn_layer_gpu;
            l.BackwardGpu = backward_crnn_layer_gpu;
            l.UpdateGpu = update_crnn_layer_gpu;

            l.StateGpu = cuda_make_array(l.State, l.Hidden * batch * (steps + 1));
            l.OutputGpu = l.OutputLayer.OutputGpu;
            l.DeltaGpu = l.OutputLayer.DeltaGpu;


            return l;
        }

        void update_crnn_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_convolutional_layer(*(l.InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer(*(l.InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer(*(l.OutputLayer), batch, learning_rate, momentum, decay);
        }

        void forward_crnn_layer(Layer l, NetworkState state)
        {
            NetworkState s = new Layer();
            s.Train = state.Train;
            int i;
            layer input_layer = *(l.InputLayer);
            layer self_layer = *(l.InputLayer);
            layer output_layer = *(l.OutputLayer);

            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, output_layer.Delta, 1);
            Blas.Fill_cpu(l.Hidden * l.Batch * l.Steps, 0, self_layer.Delta, 1);
            Blas.Fill_cpu(l.Hidden * l.Batch * l.Steps, 0, input_layer.Delta, 1);
            if (state.Train) Blas.Fill_cpu(l.Hidden * l.Batch, 0, l.State, 1);

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = state.Input;
                forward_convolutional_layer(input_layer, s);

                s.Input = l.State;
                forward_convolutional_layer(self_layer, s);

                float[] old_state = l.State;
                if (state.Train) l.State += l.Hidden * l.Batch;
                if (l.shortcut)
                {
                    Blas.Copy_cpu(l.Hidden * l.Batch, old_state, 1, l.State, 1);
                }
                else
                {
                    Blas.Fill_cpu(l.Hidden * l.Batch, 0, l.State, 1);
                }
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, input_layer.Output, 1, l.State, 1);
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output, 1, l.State, 1);

                s.Input = l.State;
                forward_convolutional_layer(output_layer, s);

                state.Input += l.Inputs * l.Batch;
                increment_layer(&input_layer, 1);
                increment_layer(&self_layer, 1);
                increment_layer(&output_layer, 1);
            }
        }

        void backward_crnn_layer(Layer l, NetworkState state)
        {
            NetworkState s = new Layer();
            s.Train = state.Train;
            int i;
            layer input_layer = *(l.InputLayer);
            layer self_layer = *(l.InputLayer);
            layer output_layer = *(l.OutputLayer);

            increment_layer(&input_layer, l.Steps - 1);
            increment_layer(&self_layer, l.Steps - 1);
            increment_layer(&output_layer, l.Steps - 1);

            l.State += l.Hidden * l.Batch * l.Steps;
            for (i = l.Steps - 1; i >= 0; --i)
            {
                Blas.Copy_cpu(l.Hidden * l.Batch, input_layer.Output, 1, l.State, 1);
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output, 1, l.State, 1);

                s.Input = l.State;
                s.Delta = self_layer.Delta;
                backward_convolutional_layer(output_layer, s);

                l.State -= l.Hidden * l.Batch;
                /*
                   if(i > 0){
                   Blas.Copy_cpu(l.Hidden * l.Batch, input_layer.Output - l.Hidden*l.Batch, 1, l.State, 1);
                   Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output - l.Hidden*l.Batch, 1, l.State, 1);
                   }else{
                   Blas.Fill_cpu(l.Hidden * l.Batch, 0, l.State, 1);
                   }
                 */

                s.Input = l.State;
                s.Delta = self_layer.Delta - l.Hidden * l.Batch;
                if (i == 0) s.Delta = 0;
                backward_convolutional_layer(self_layer, s);

                Blas.Copy_cpu(l.Hidden * l.Batch, self_layer.Delta, 1, input_layer.Delta, 1);
                if (i > 0 && l.shortcut) Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Delta, 1, self_layer.Delta - l.Hidden * l.Batch, 1);
                s.Input = state.Input + i * l.Inputs * l.Batch;
                if (state.Delta) s.Delta = state.Delta + i * l.Inputs * l.Batch;
                else s.Delta = 0;
                backward_convolutional_layer(input_layer, s);

                increment_layer(&input_layer, -1);
                increment_layer(&self_layer, -1);
                increment_layer(&output_layer, -1);
            }
        }



        void pull_crnn_layer(Layer l)
        {
            pull_convolutional_layer(*(l.InputLayer));
            pull_convolutional_layer(*(l.InputLayer));
            pull_convolutional_layer(*(l.OutputLayer));
        }

        void push_crnn_layer(Layer l)
        {
            push_convolutional_layer(*(l.InputLayer));
            push_convolutional_layer(*(l.InputLayer));
            push_convolutional_layer(*(l.OutputLayer));
        }

        void update_crnn_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_convolutional_layer_gpu(*(l.InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer_gpu(*(l.InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer_gpu(*(l.OutputLayer), batch, learning_rate, momentum, decay);
        }

        void forward_crnn_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new Layer();
            s.Train = state.Train;
            int i;
            layer input_layer = *(l.InputLayer);
            layer self_layer = *(l.InputLayer);
            layer output_layer = *(l.OutputLayer);

            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, output_layer.DeltaGpu, 1);
            fill_ongpu(l.Hidden * l.Batch * l.Steps, 0, self_layer.DeltaGpu, 1);
            fill_ongpu(l.Hidden * l.Batch * l.Steps, 0, input_layer.DeltaGpu, 1);
            if (state.Train) fill_ongpu(l.Hidden * l.Batch, 0, l.StateGpu, 1);

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = state.Input;
                forward_convolutional_layer_gpu(input_layer, s);

                s.Input = l.StateGpu;
                forward_convolutional_layer_gpu(self_layer, s);

                float[] old_state = l.StateGpu;
                if (state.Train) l.StateGpu += l.Hidden * l.Batch;
                if (l.shortcut)
                {
                    Blas.copy_ongpu(l.Hidden * l.Batch, old_state, 1, l.StateGpu, 1);
                }
                else
                {
                    fill_ongpu(l.Hidden * l.Batch, 0, l.StateGpu, 1);
                }
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, input_layer.OutputGpu, 1, l.StateGpu, 1);
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.OutputGpu, 1, l.StateGpu, 1);

                s.Input = l.StateGpu;
                forward_convolutional_layer_gpu(output_layer, s);

                state.Input += l.Inputs * l.Batch;
                increment_layer(&input_layer, 1);
                increment_layer(&self_layer, 1);
                increment_layer(&output_layer, 1);
            }
        }

        void backward_crnn_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new Layer();
            s.Train = state.Train;
            int i;
            layer input_layer = *(l.InputLayer);
            layer self_layer = *(l.InputLayer);
            layer output_layer = *(l.OutputLayer);
            increment_layer(&input_layer, l.Steps - 1);
            increment_layer(&self_layer, l.Steps - 1);
            increment_layer(&output_layer, l.Steps - 1);
            l.StateGpu += l.Hidden * l.Batch * l.Steps;
            for (i = l.Steps - 1; i >= 0; --i)
            {
                Blas.copy_ongpu(l.Hidden * l.Batch, input_layer.OutputGpu, 1, l.StateGpu, 1);
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.OutputGpu, 1, l.StateGpu, 1);

                s.Input = l.StateGpu;
                s.Delta = self_layer.DeltaGpu;
                backward_convolutional_layer_gpu(output_layer, s);

                l.StateGpu -= l.Hidden * l.Batch;

                s.Input = l.StateGpu;
                s.Delta = self_layer.DeltaGpu - l.Hidden * l.Batch;
                if (i == 0) s.Delta = 0;
                backward_convolutional_layer_gpu(self_layer, s);

                Blas.copy_ongpu(l.Hidden * l.Batch, self_layer.DeltaGpu, 1, input_layer.DeltaGpu, 1);
                if (i > 0 && l.shortcut) Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.DeltaGpu, 1, self_layer.DeltaGpu - l.Hidden * l.Batch, 1);
                s.Input = state.Input + i * l.Inputs * l.Batch;
                if (state.Delta) s.Delta = state.Delta + i * l.Inputs * l.Batch;
                else s.Delta = 0;
                backward_convolutional_layer_gpu(input_layer, s);

                increment_layer(&input_layer, -1);
                increment_layer(&self_layer, -1);
                increment_layer(&output_layer, -1);
            }
        }
        Image get_crop_image(Layer l)
        {
            int h = l.OutH;
            int w = l.OutW;
            int c = l.OutC;
            return new Image(w, h, c, l.Output);
        }

        void backward_crop_layer(Layer l, NetworkState state) { }
        void backward_crop_layer_gpu(Layer l, NetworkState state) { }

        Layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
        {
            Console.Error.Write($"Crop Layer: %d x %d . %d x %d x %d Image\n", h, w, crop_height, crop_width, c);
            Layer l = new Layer();
            l.LayerType = CROP;
            l.Batch = batch;
            l.H = h;
            l.W = w;
            l.C = c;
            l.Scale = (float)crop_height / h;
            l.Flip = flip;
            l.angle = angle;
            l.saturation = saturation;
            l.exposure = exposure;
            l.OutW = crop_width;
            l.OutH = crop_height;
            l.OutC = c;
            l.Inputs = l.W * l.H * l.C;
            l.Outputs = l.OutW * l.OutH * l.OutC;
            l.Output = calloc(l.Outputs * batch, sizeof(float));
            l.Forward = forward_crop_layer;
            l.Backward = backward_crop_layer;


            l.ForwardGpu = forward_crop_layer_gpu;
            l.BackwardGpu = backward_crop_layer_gpu;
            l.OutputGpu = cuda_make_array(l.Output, l.Outputs * batch);
            l.RandGpu = cuda_make_array(0, l.Batch * 8);

            return l;
        }

        void resize_crop_layer(Layer l, int w, int h)
        {
            l.W = w;
            l.H = h;

            l.OutW = l.Scale * w;
            l.OutH = l.Scale * h;

            l.Inputs = l.W * l.H * l.C;
            l.Outputs = l.OutH * l.OutW * l.OutC;

            l.Output = realloc(l.Output, l.Batch * l.Outputs * sizeof(float));

            cuda_free(l.OutputGpu);
            l.OutputGpu = cuda_make_array(l.Output, l.Outputs * l.Batch);

        }


        void forward_crop_layer(Layer l, NetworkState state)
        {
            int i, j, c, b, row, col;
            int index;
            int count = 0;
            int flip = (l.Flip && Utils.Rand.Next() % 2);
            int dh = Utils.Rand.Next() % (l.H - l.OutH + 1);
            int dw = Utils.Rand.Next() % (l.W - l.OutW + 1);
            float scale = 2;
            float trans = -1;
            if (l.Noadjust)
            {
                scale = 1;
                trans = 0;
            }
            if (!state.Train)
            {
                flip = 0;
                dh = (l.H - l.OutH) / 2;
                dw = (l.W - l.OutW) / 2;
            }
            for (b = 0; b < l.Batch; ++b)
            {
                for (c = 0; c < l.C; ++c)
                {
                    for (i = 0; i < l.OutH; ++i)
                    {
                        for (j = 0; j < l.OutW; ++j)
                        {
                            if (flip)
                            {
                                col = l.W - dw - j - 1;
                            }
                            else
                            {
                                col = j + dw;
                            }
                            row = i + dh;
                            index = col + l.W * (row + l.H * (c + l.C * b));
                            l.Output[count++] = state.Input[index] * scale + trans;
                        }
                    }
                }
            }
        }
        int deconvolutional_out_height(Layer l)
        {
            int h = l.Stride * (l.H - 1) + l.Size;
            return h;
        }

        int deconvolutional_out_width(Layer l)
        {
            int w = l.Stride * (l.W - 1) + l.Size;
            return w;
        }

        int deconvolutional_out_size(Layer l)
        {
            return deconvolutional_out_height(l) * deconvolutional_out_width(l);
        }

        Image get_deconvolutional_image(Layer l)
        {
            int h, w, c;
            h = deconvolutional_out_height(l);
            w = deconvolutional_out_width(l);
            c = l.N;
            return new Image(w, h, c, l.Output);
        }

        Image get_deconvolutional_delta(Layer l)
        {
            int h, w, c;
            h = deconvolutional_out_height(l);
            w = deconvolutional_out_width(l);
            c = l.N;
            return new Image(w, h, c, l.Delta);
        }

        Layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, Activation activation)
        {
            int i;
            Layer l = new Layer();
            l.LayerType = DECONVOLUTIONAL;

            l.H = h;
            l.W = w;
            l.C = c;
            l.N = n;
            l.Batch = batch;
            l.Stride = stride;
            l.Size = size;

            l.Weights = calloc(c * n * size * size, sizeof(float));
            l.WeightUpdates = calloc(c * n * size * size, sizeof(float));

            l.Biases = calloc(n, sizeof(float));
            l.BiasUpdates = calloc(n, sizeof(float));
            float scale = 1./ (float)Math.Sqrt(size * size * c);
            for (i = 0; i < c * n * size * size; ++i) l.Weights[i] = scale * rand_normal();
            for (i = 0; i < n; ++i)
            {
                l.Biases[i] = scale;
            }
            int out_h = deconvolutional_out_height(l);
            int out_w = deconvolutional_out_width(l);

            l.OutH = out_h;
            l.OutW = out_w;
            l.OutC = n;
            l.Outputs = l.OutW * l.OutH * l.OutC;
            l.Inputs = l.W * l.H * l.C;

            l.ColImage = calloc(h * w * size * size * n, sizeof(float));
            l.Output = calloc(l.Batch * out_h * out_w * n, sizeof(float));
            l.Delta = calloc(l.Batch * out_h * out_w * n, sizeof(float));

            l.Forward = forward_deconvolutional_layer;
            l.Backward = backward_deconvolutional_layer;
            l.Update = update_deconvolutional_layer;


            l.WeightsGpu = cuda_make_array(l.Weights, c * n * size * size);
            l.WeightUpdatesGpu = cuda_make_array(l.WeightUpdates, c * n * size * size);

            l.BiasesGpu = cuda_make_array(l.Biases, n);
            l.BiasUpdatesGpu = cuda_make_array(l.BiasUpdates, n);

            l.ColImageGpu = cuda_make_array(l.ColImage, h * w * size * size * n);
            l.DeltaGpu = cuda_make_array(l.Delta, l.Batch * out_h * out_w * n);
            l.OutputGpu = cuda_make_array(l.Output, l.Batch * out_h * out_w * n);


            l.Activation = activation;

            Console.Error.Write($"Deconvolutional Layer: %d x %d x %d Image, %d filters . %d x %d x %d Image\n", h, w, c, n, out_h, out_w, n);

            return l;
        }

        void resize_deconvolutional_layer(Layer* l, int h, int w)
        {
            l.H = h;
            l.W = w;
            int out_h = deconvolutional_out_height(*l);
            int out_w = deconvolutional_out_width(*l);

            l.ColImage = realloc(l.ColImage,
                                        out_h * out_w * l.Size * l.Size * l.C * sizeof(float));
            l.Output = realloc(l.Output,
                                        l.Batch * out_h * out_w * l.N * sizeof(float));
            l.Delta = realloc(l.Delta,
                                        l.Batch * out_h * out_w * l.N * sizeof(float));

            cuda_free(l.ColImageGpu);
            cuda_free(l.DeltaGpu);
            cuda_free(l.OutputGpu);

            l.ColImageGpu = cuda_make_array(l.ColImage, out_h * out_w * l.Size * l.Size * l.C);
            l.DeltaGpu = cuda_make_array(l.Delta, l.Batch * out_h * out_w * l.N);
            l.OutputGpu = cuda_make_array(l.Output, l.Batch * out_h * out_w * l.N);

        }

        void forward_deconvolutional_layer(Layer l, NetworkState state)
        {
            int i;
            int out_h = deconvolutional_out_height(l);
            int out_w = deconvolutional_out_width(l);
            int size = out_h * out_w;

            int m = l.Size * l.Size * l.N;
            int n = l.H * l.W;
            int k = l.C;

            Blas.Fill_cpu(l.Outputs * l.Batch, 0, l.Output, 1);

            for (i = 0; i < l.Batch; ++i)
            {
                float[] a = l.Weights;
                float[] b = state.Input + i * l.C * l.H * l.W;
                float[] c = l.ColImage;

                Gemm.gemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

                Im2Col.col2im_cpu(c, l.N, out_h, out_w, l.Size, l.Stride, 0, l.Output + i * l.N * size);
            }
            add_bias(l.Output, l.Biases, l.Batch, l.N, size);
            activate_array(l.Output, l.Batch * l.N * size, l.Activation);
        }

        void backward_deconvolutional_layer(Layer l, NetworkState state)
        {
            float alpha = 1./ l.Batch;
            int out_h = deconvolutional_out_height(l);
            int out_w = deconvolutional_out_width(l);
            int size = out_h * out_w;
            int i;

            gradient_array(l.Output, size * l.N * l.Batch, l.Activation, l.Delta);
            backward_bias(l.BiasUpdates, l.Delta, l.Batch, l.N, size);

            for (i = 0; i < l.Batch; ++i)
            {
                int m = l.C;
                int n = l.Size * l.Size * l.N;
                int k = l.H * l.W;

                float[] a = state.Input + i * m * n;
                float[] b = l.ColImage;
                float[] c = l.WeightUpdates;

                Im2Col.im2col_cpu(l.Delta + i * l.N * size, l.N, out_h, out_w,
                        l.Size, l.Stride, 0, b);
                Gemm.gemm(0, 1, m, n, k, alpha, a, k, b, k, 1, c, n);

                if (state.Delta)
                {
                    int m = l.C;
                    int n = l.H * l.W;
                    int k = l.Size * l.Size * l.N;

                    float[] a = l.Weights;
                    float[] b = l.ColImage;
                    float[] c = state.Delta + i * n * m;

                    Gemm.gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                }
            }
        }

        void update_deconvolutional_layer(Layer l, float learning_rate, float momentum, float decay)
        {
            int size = l.Size * l.Size * l.C * l.N;
            Blas.Axpy_cpu(l.N, learning_rate, l.BiasUpdates, 1, l.Biases, 1);
            Blas.Scal_cpu(l.N, momentum, l.BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay, l.Weights, 1, l.WeightUpdates, 1);
            Blas.Axpy_cpu(size, learning_rate, l.WeightUpdates, 1, l.Weights, 1);
            Blas.Scal_cpu(size, momentum, l.WeightUpdates, 1);
        }

        Layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Detection;

            l.N = n;
            l.Batch = batch;
            l.Inputs = inputs;
            l.Classes = classes;
            l.Coords = coords;
            l.Rescore = rescore;
            l.Side = side;
            l.W = side;
            l.H = side;
            assert(side * side * ((1 + l.Coords) * l.N + l.Classes) == inputs);
            l.Cost = calloc(1, sizeof(float));
            l.Outputs = l.Inputs;
            l.Truths = l.Side * l.Side * (1 + l.Coords + l.Classes);
            l.Output = calloc(batch * l.Outputs, sizeof(float));
            l.Delta = calloc(batch * l.Outputs, sizeof(float));

            l.Forward = forward_detection_layer;
            l.Backward = backward_detection_layer;

            l.ForwardGpu = forward_detection_layer_gpu;
            l.BackwardGpu = backward_detection_layer_gpu;
            l.OutputGpu = cuda_make_array(l.Output, batch * l.Outputs);
            l.DeltaGpu = cuda_make_array(l.Delta, batch * l.Outputs);


            Console.Error.Write($"Detection Layer\n");
            srand(0);

            return l;
        }

        void forward_detection_layer(Layer l, NetworkState state)
        {
            int locations = l.Side * l.Side;
            int i, j;
            memcpy(l.Output, state.Input, l.Outputs * l.Batch * sizeof(float));
            //if(l.reorg) reorg(l.Output, l.W*l.H, size*l.N, l.Batch, 1);
            int b;
            if (l.softmax)
            {
                for (b = 0; b < l.Batch; ++b)
                {
                    int index = b * l.Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int offset = i * l.Classes;
                        softmax(l.Output + index + offset, l.Classes, 1,
                                l.Output + index + offset);
                    }
                }
            }
            if (state.Train)
            {
                float avg_iou = 0;
                float avg_cat = 0;
                float avg_allcat = 0;
                float avg_obj = 0;
                float avg_anyobj = 0;
                int count = 0;
                *(l.Cost) = 0;
                int size = l.Inputs * l.Batch;
                memset(l.Delta, 0, size * sizeof(float));
                for (b = 0; b < l.Batch; ++b)
                {
                    int index = b * l.Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int truth_index = (b * locations + i) * (1 + l.Coords + l.Classes);
                        int is_obj = state.Truth[truth_index];
                        for (j = 0; j < l.N; ++j)
                        {
                            int p_index = index + locations * l.Classes + i * l.N + j;
                            l.Delta[p_index] = l.noobject_scale * (0 - l.Output[p_index]);
                            *(l.Cost) += l.noobject_scale * (float)Math.Pow(l.Output[p_index], 2);
                            avg_anyobj += l.Output[p_index];
                        }

                        int best_index = -1;
                        float best_iou = 0;
                        float best_rmse = 20;

                        if (!is_obj)
                        {
                            continue;
                        }

                        int class_index = index + i * l.Classes;
                        for (j = 0; j < l.Classes; ++j)
                        {
                            l.Delta[class_index + j] = l.class_scale * (state.Truth[truth_index + 1 + j] - l.Output[class_index + j]);
                            *(l.Cost) += l.class_scale * (float)Math.Pow(state.Truth[truth_index + 1 + j] - l.Output[class_index + j], 2);
                            if (state.Truth[truth_index + 1 + j]) avg_cat += l.Output[class_index + j];
                            avg_allcat += l.Output[class_index + j];
                        }

                        Box truth = float_to_box(state.Truth + truth_index + 1 + l.Classes);
                        truth.X /= l.Side;
                        truth.y /= l.Side;

                        for (j = 0; j < l.N; ++j)
                        {
                            int box_index = index + locations * (l.Classes + l.N) + (i * l.N + j) * l.Coords;
                            Box outputout = float_to_box(l.Output + box_index);
                            outputout.X /= l.Side;
                            outputout.y /= l.Side;

                            if (l.(float)Math.Sqrt)
                            {
                                outputout.W = outputout.W * outputout.W;
                                outputout.H = outputout.H * outputout.H;
                            }

                            float iou = box_iou(outputout, truth);
                            //iou = 0;
                            float rmse = box_rmse(outputout, truth);
                            if (best_iou > 0 || iou > 0)
                            {
                                if (iou > best_iou)
                                {
                                    best_iou = iou;
                                    best_index = j;
                                }
                            }
                            else
                            {
                                if (rmse < best_rmse)
                                {
                                    best_rmse = rmse;
                                    best_index = j;
                                }
                            }
                        }

                        if (l.forced)
                        {
                            if (truth.W * truth.H < .1)
                            {
                                best_index = 1;
                            }
                            else
                            {
                                best_index = 0;
                            }
                        }
                        if (l.random && *(state.net.seen) < 64000)
                        {
                            best_index = Utils.Rand.Next() % l.N;
                        }

                        int box_index = index + locations * (l.Classes + l.N) + (i * l.N + best_index) * l.Coords;
                        int tbox_index = truth_index + 1 + l.Classes;

                        Box outputout = float_to_box(l.Output + box_index);
                        outputout.X /= l.Side;
                        outputout.y /= l.Side;
                        if (l.(float)Math.Sqrt)
                        {
                            outputout.W = outputout.W * outputout.W;
                            outputout.H = outputout.H * outputout.H;
                        }
                        float iou = box_iou(outputout, truth);

                        //Console.Write($"%d,", best_index);
                        int p_index = index + locations * l.Classes + i * l.N + best_index;
                        *(l.Cost) -= l.noobject_scale * (float)Math.Pow(l.Output[p_index], 2);
                        *(l.Cost) += l.object_scale * (float)Math.Pow(1 - l.Output[p_index], 2);
                        avg_obj += l.Output[p_index];
                        l.Delta[p_index] = l.object_scale * (1.- l.Output[p_index]);

                        if (l.Rescore)
                        {
                            l.Delta[p_index] = l.object_scale * (iou - l.Output[p_index]);
                        }

                        l.Delta[box_index + 0] = l.coord_scale * (state.Truth[tbox_index + 0] - l.Output[box_index + 0]);
                        l.Delta[box_index + 1] = l.coord_scale * (state.Truth[tbox_index + 1] - l.Output[box_index + 1]);
                        l.Delta[box_index + 2] = l.coord_scale * (state.Truth[tbox_index + 2] - l.Output[box_index + 2]);
                        l.Delta[box_index + 3] = l.coord_scale * (state.Truth[tbox_index + 3] - l.Output[box_index + 3]);
                        if (l.(float)Math.Sqrt)
                        {
                            l.Delta[box_index + 2] = l.coord_scale * ((float)Math.Sqrt(state.Truth[tbox_index + 2]) - l.Output[box_index + 2]);
                            l.Delta[box_index + 3] = l.coord_scale * ((float)Math.Sqrt(state.Truth[tbox_index + 3]) - l.Output[box_index + 3]);
                        }

                        *(l.Cost) += (float)Math.Pow(1 - iou, 2);
                        avg_iou += iou;
                        ++count;
                    }
                }

                if (0)
                {
                    float[] costs = calloc(l.Batch * locations * l.N, sizeof(float));
                    for (b = 0; b < l.Batch; ++b)
                    {
                        int index = b * l.Inputs;
                        for (i = 0; i < locations; ++i)
                        {
                            for (j = 0; j < l.N; ++j)
                            {
                                int p_index = index + locations * l.Classes + i * l.N + j;
                                costs[b * locations * l.N + i * l.N + j] = l.Delta[p_index] * l.Delta[p_index];
                            }
                        }
                    }
                    int indexes[100];
                    top_k(costs, l.Batch * locations * l.N, 100, indexes);
                    float cutoff = costs[indexes[99]];
                    for (b = 0; b < l.Batch; ++b)
                    {
                        int index = b * l.Inputs;
                        for (i = 0; i < locations; ++i)
                        {
                            for (j = 0; j < l.N; ++j)
                            {
                                int p_index = index + locations * l.Classes + i * l.N + j;
                                if (l.Delta[p_index] * l.Delta[p_index] < cutoff) l.Delta[p_index] = 0;
                            }
                        }
                    }
                    free(costs);
                }


                *(l.Cost) = (float)Math.Pow(mag_array(l.Delta, l.Outputs * l.Batch), 2);


                Console.Write($"Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou / count, avg_cat / count, avg_allcat / (count * l.Classes), avg_obj / count, avg_anyobj / (l.Batch * locations * l.N), count);
                //if(l.reorg) reorg(l.Delta, l.W*l.H, size*l.N, l.Batch, 0);
            }
        }

        void backward_detection_layer(Layer l, NetworkState state)
        {
            Blas.Axpy_cpu(l.Batch * l.Inputs, 1, l.Delta, 1, state.Delta, 1);
        }

        void get_detection_boxes(Layer l, int w, int h, float thresh, float[][] probs, Box[] boxes, int only_objectness)
        {
            int i, j, n;
            float[] predictions = l.Output;
            //int per_cell = 5*num+classes;
            for (i = 0; i < l.Side * l.Side; ++i)
            {
                int row = i / l.Side;
                int col = i % l.Side;
                for (n = 0; n < l.N; ++n)
                {
                    int index = i * l.N + n;
                    int p_index = l.Side * l.Side * l.Classes + i * l.N + n;
                    float scale = predictions[p_index];
                    int box_index = l.Side * l.Side * (l.Classes + l.N) + (i * l.N + n) * 4;
                    boxes[index].X = (predictions[box_index + 0] + col) / l.Side * w;
                    boxes[index].y = (predictions[box_index + 1] + row) / l.Side * h;
                    boxes[index].W = (float)Math.Pow(predictions[box_index + 2], (l.(float)Math.Sqrt ? 2 : 1)) * w;
                    boxes[index].H = (float)Math.Pow(predictions[box_index + 3], (l.(float)Math.Sqrt ? 2 : 1)) * h;
                    for (j = 0; j < l.Classes; ++j)
                    {
                        int class_index = i * l.Classes;
                        float prob = scale * predictions[class_index + j];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                    if (only_objectness)
                    {
                        probs[index][0] = scale;
                    }
                }
            }
        }



        void forward_detection_layer_gpu(Layer l, NetworkState state)
        {
            if (!state.Train)
            {
                Blas.copy_ongpu(l.Batch * l.Inputs, state.Input, 1, l.OutputGpu, 1);
                return;
            }

            float[] in_cpu = calloc(l.Batch * l.Inputs, sizeof(float));
            float[] truth_cpu = 0;
            if (state.Truth)
            {
                int num_truth = l.Batch * l.Side * l.Side * (1 + l.Coords + l.Classes);
                truth_cpu = calloc(num_truth, sizeof(float));
                Array.Copy(state.Truth, truth_cpu, num_truth);
            }
            Array.Copy(state.Input, in_cpu, l.Batch * l.Inputs);
            NetworkState cpu_state = state;
            cpu_state.Train = state.Train;
            cpu_state.Truth = truth_cpu;
            cpu_state.Input = in_cpu;
            forward_detection_layer(l, cpu_state);
            cuda_push_array(l.OutputGpu, l.Output, l.Batch * l.Outputs);
            cuda_push_array(l.DeltaGpu, l.Delta, l.Batch * l.Inputs);
            free(cpu_state.Input);
            if (cpu_state.Truth) free(cpu_state.Truth);
        }

        void backward_detection_layer_gpu(Layer l, NetworkState state)
        {
            Blas.axpy_ongpu(l.Batch * l.Inputs, 1, l.DeltaGpu, 1, state.Delta, 1);
            //Blas.copy_ongpu(l.Batch*l.Inputs, l.DeltaGpu, 1, state.Delta, 1);
        }

        Layer make_dropout_layer(int batch, int inputs, float probability)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Dropout;
            l.Probability = probability;
            l.Inputs = inputs;
            l.Outputs = inputs;
            l.Batch = batch;
            l.Rand = calloc(inputs * batch, sizeof(float));
            l.Scale = 1./ (1.- probability);
            l.Forward = forward_dropout_layer;
            l.Backward = backward_dropout_layer;

            l.ForwardGpu = forward_dropout_layer_gpu;
            l.BackwardGpu = backward_dropout_layer_gpu;
            l.RandGpu = cuda_make_array(l.Rand, inputs * batch);

            Console.Error.Write($"dropout       p = %.2f               %4d  .  %4d\n", probability, inputs, inputs);
            return l;
        }

        void resize_dropout_layer(Layer l, int inputs)
        {
            l.Rand = realloc(l.Rand, l.Inputs * l.Batch * sizeof(float));

            cuda_free(l.RandGpu);

            l.RandGpu = cuda_make_array(l.Rand, inputs * l.Batch);

        }

        void forward_dropout_layer(Layer l, NetworkState state)
        {
            int i;
            if (!state.Train) return;
            for (i = 0; i < l.Batch * l.Inputs; ++i)
            {
                float r = rand_uniform(0, 1);
                l.Rand[i] = r;
                if (r < l.Probability) state.Input[i] = 0;
                else state.Input[i] *= l.Scale;
            }
        }

        void backward_dropout_layer(Layer l, NetworkState state)
        {
            int i;
            if (!state.Delta) return;
            for (i = 0; i < l.Batch * l.Inputs; ++i)
            {
                float r = l.Rand[i];
                if (r < l.Probability) state.Delta[i] = 0;
                else state.Delta[i] *= l.Scale;
            }
        }

        static void increment_layer(Layer l, int steps)
        {
            int num = l.Outputs * l.Batch * steps;
            l.Output += num;
            l.Delta += num;
            l.X += num;
            l.XNorm += num;


            l.OutputGpu += num;
            l.DeltaGpu += num;
            l.XGpu += num;
            l.XNormGpu += num;

        }

        Layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize)
        {
            Console.Error.Write($"GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
            batch = batch / steps;
            Layer l = new Layer();
            l.Batch = batch;
            l.LayerType = LayerType.Gru;
            l.Steps = steps;
            l.Inputs = inputs;

            l.InputZLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.InputZLayer) = make_connected_layer(batch * steps, inputs, outputs, LINEAR, batch_normalize);
            l.InputZLayer.Batch = batch;

            l.StateZLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.StateZLayer) = make_connected_layer(batch * steps, outputs, outputs, LINEAR, batch_normalize);
            l.StateZLayer.Batch = batch;



            l.InputRLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.InputRLayer) = make_connected_layer(batch * steps, inputs, outputs, LINEAR, batch_normalize);
            l.InputRLayer.Batch = batch;

            l.StateRLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            *(l.StateRLayer) = make_connected_layer(batch * steps, outputs, outputs, LINEAR, batch_normalize);
            l.StateRLayer.Batch = batch;



            l.InputHLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            (l.InputHLayer) = make_connected_layer(batch * steps, inputs, outputs, LINEAR, batch_normalize);
            l.InputHLayer.Batch = batch;

            l.StateHLayer = malloc(sizeof(layer));
            Console.Error.Write($"\t\t");
            (l.StateHLayer) = make_connected_layer(batch * steps, outputs, outputs, LINEAR, batch_normalize);
            l.StateHLayer.Batch = batch;

            l.BatchNormalize = batch_normalize;


            l.Outputs = outputs;
            l.Output = calloc(outputs * batch * steps, sizeof(float));
            l.Delta = calloc(outputs * batch * steps, sizeof(float));
            l.State = calloc(outputs * batch, sizeof(float));
            l.PrevState = calloc(outputs * batch, sizeof(float));
            l.ForgotState = calloc(outputs * batch, sizeof(float));
            l.ForgotDelta = calloc(outputs * batch, sizeof(float));

            l.RCpu = calloc(outputs * batch, sizeof(float));
            l.ZCpu = calloc(outputs * batch, sizeof(float));
            l.HCpu = calloc(outputs * batch, sizeof(float));

            l.Forward = forward_gru_layer;
            l.Backward = backward_gru_layer;
            l.Update = update_gru_layer;


            l.ForwardGpu = forward_gru_layer_gpu;
            l.BackwardGpu = backward_gru_layer_gpu;
            l.UpdateGpu = update_gru_layer_gpu;

            l.ForgotStateGpu = cuda_make_array(l.Output, batch * outputs);
            l.ForgotDeltaGpu = cuda_make_array(l.Output, batch * outputs);
            l.PrevStateGpu = cuda_make_array(l.Output, batch * outputs);
            l.StateGpu = cuda_make_array(l.Output, batch * outputs);
            l.OutputGpu = cuda_make_array(l.Output, batch * outputs * steps);
            l.DeltaGpu = cuda_make_array(l.Delta, batch * outputs * steps);
            l.RGpu = cuda_make_array(l.OutputGpu, batch * outputs);
            l.ZGpu = cuda_make_array(l.OutputGpu, batch * outputs);
            l.HGpu = cuda_make_array(l.OutputGpu, batch * outputs);


            return l;
        }

        void update_gru_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_connected_layer((l.InputLayer), batch, learning_rate, momentum, decay);
            update_connected_layer((l.InputLayer), batch, learning_rate, momentum, decay);
            update_connected_layer((l.OutputLayer), batch, learning_rate, momentum, decay);
        }

        void forward_gru_layer(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_z_layer = (l.InputZLayer);
            Layer input_r_layer = (l.InputRLayer);
            Layer input_h_layer = (l.InputHLayer);

            Layer state_z_layer = (l.StateZLayer);
            Layer state_r_layer = (l.StateRLayer);
            Layer state_h_layer = (l.StateHLayer);

            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, input_z_layer.Delta, 1);
            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, input_r_layer.Delta, 1);
            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, input_h_layer.Delta, 1);

            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, state_z_layer.Delta, 1);
            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, state_r_layer.Delta, 1);
            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, state_h_layer.Delta, 1);
            if (state.Train)
            {
                Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, l.Delta, 1);
                Blas.Copy_cpu(l.Outputs * l.Batch, l.State, 1, l.PrevState, 1);
            }

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = l.State;
                forward_connected_layer(state_z_layer, s);
                forward_connected_layer(state_r_layer, s);

                s.Input = state.Input;
                forward_connected_layer(input_z_layer, s);
                forward_connected_layer(input_r_layer, s);
                forward_connected_layer(input_h_layer, s);


                Blas.Copy_cpu(l.Outputs * l.Batch, input_z_layer.Output, 1, l.ZCpu, 1);
                Blas.Axpy_cpu(l.Outputs * l.Batch, 1, state_z_layer.Output, 1, l.ZCpu, 1);

                Blas.Copy_cpu(l.Outputs * l.Batch, input_r_layer.Output, 1, l.RCpu, 1);
                Blas.Axpy_cpu(l.Outputs * l.Batch, 1, state_r_layer.Output, 1, l.RCpu, 1);

                activate_array(l.ZCpu, l.Outputs * l.Batch, LOGISTIC);
                activate_array(l.RCpu, l.Outputs * l.Batch, LOGISTIC);

                Blas.Copy_cpu(l.Outputs * l.Batch, l.State, 1, l.ForgotState, 1);
                Blas.Mul_cpu(l.Outputs * l.Batch, l.RCpu, 1, l.ForgotState, 1);

                s.Input = l.ForgotState;
                forward_connected_layer(state_h_layer, s);

                Blas.Copy_cpu(l.Outputs * l.Batch, input_h_layer.Output, 1, l.HCpu, 1);
                Blas.Axpy_cpu(l.Outputs * l.Batch, 1, state_h_layer.Output, 1, l.HCpu, 1);

                // USET activate_array(l.HCpu, l.Outputs * l.Batch, TANH);
                activate_array(l.HCpu, l.Outputs * l.Batch, LOGISTIC);


                weighted_sum_cpu(l.State, l.HCpu, l.ZCpu, l.Outputs * l.Batch, l.Output);

                Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, 1, l.State, 1);

                state.Input += l.Inputs * l.Batch;
                l.Output += l.Outputs * l.Batch;
                increment_layer(&input_z_layer, 1);
                increment_layer(&input_r_layer, 1);
                increment_layer(&input_h_layer, 1);

                increment_layer(&state_z_layer, 1);
                increment_layer(&state_r_layer, 1);
                increment_layer(&state_h_layer, 1);
            }
        }

        void backward_gru_layer(Layer l, NetworkState state)
        {
        }



        void pull_gru_layer(Layer l)
        {
        }

        void push_gru_layer(Layer l)
        {
        }

        void update_gru_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_connected_layer_gpu(*(l.InputRLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(*(l.InputZLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(*(l.InputHLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(*(l.StateRLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(*(l.StateZLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(*(l.StateHLayer), batch, learning_rate, momentum, decay);
        }

        void forward_gru_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new Layer();
            s.Train = state.Train;
            int i;
            layer input_z_layer = *(l.InputZLayer);
            layer input_r_layer = *(l.InputRLayer);
            layer input_h_layer = *(l.InputHLayer);

            layer state_z_layer = *(l.StateZLayer);
            layer state_r_layer = *(l.StateRLayer);
            layer state_h_layer = *(l.StateHLayer);

            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, input_z_layer.DeltaGpu, 1);
            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, input_r_layer.DeltaGpu, 1);
            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, input_h_layer.DeltaGpu, 1);

            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, state_z_layer.DeltaGpu, 1);
            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, state_r_layer.DeltaGpu, 1);
            fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, state_h_layer.DeltaGpu, 1);
            if (state.Train)
            {
                fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, l.DeltaGpu, 1);
                Blas.copy_ongpu(l.Outputs * l.Batch, l.StateGpu, 1, l.PrevStateGpu, 1);
            }

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = l.StateGpu;
                forward_connected_layer_gpu(state_z_layer, s);
                forward_connected_layer_gpu(state_r_layer, s);

                s.Input = state.Input;
                forward_connected_layer_gpu(input_z_layer, s);
                forward_connected_layer_gpu(input_r_layer, s);
                forward_connected_layer_gpu(input_h_layer, s);


                Blas.copy_ongpu(l.Outputs * l.Batch, input_z_layer.OutputGpu, 1, l.ZGpu, 1);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_z_layer.OutputGpu, 1, l.ZGpu, 1);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_r_layer.OutputGpu, 1, l.RGpu, 1);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_r_layer.OutputGpu, 1, l.RGpu, 1);

                activate_array_ongpu(l.ZGpu, l.Outputs * l.Batch, LOGISTIC);
                activate_array_ongpu(l.RGpu, l.Outputs * l.Batch, LOGISTIC);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.StateGpu, 1, l.ForgotStateGpu, 1);
                mul_ongpu(l.Outputs * l.Batch, l.RGpu, 1, l.ForgotStateGpu, 1);

                s.Input = l.ForgotStateGpu;
                forward_connected_layer_gpu(state_h_layer, s);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_h_layer.OutputGpu, 1, l.HGpu, 1);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_h_layer.OutputGpu, 1, l.HGpu, 1);

                // USET activate_array_ongpu(l.HGpu, l.Outputs * l.Batch, TANH);
                activate_array_ongpu(l.HGpu, l.Outputs * l.Batch, LOGISTIC);


                weighted_sum_gpu(l.StateGpu, l.HGpu, l.ZGpu, l.Outputs * l.Batch, l.OutputGpu);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.OutputGpu, 1, l.StateGpu, 1);

                state.Input += l.Inputs * l.Batch;
                l.OutputGpu += l.Outputs * l.Batch;
                increment_layer(&input_z_layer, 1);
                increment_layer(&input_r_layer, 1);
                increment_layer(&input_h_layer, 1);

                increment_layer(&state_z_layer, 1);
                increment_layer(&state_r_layer, 1);
                increment_layer(&state_h_layer, 1);
            }
        }

        void backward_gru_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new Layer();
            s.Train = state.Train;
            int i;
            layer input_z_layer = *(l.InputZLayer);
            layer input_r_layer = *(l.InputRLayer);
            layer input_h_layer = *(l.InputHLayer);

            layer state_z_layer = *(l.StateZLayer);
            layer state_r_layer = *(l.StateRLayer);
            layer state_h_layer = *(l.StateHLayer);

            increment_layer(&input_z_layer, l.Steps - 1);
            increment_layer(&input_r_layer, l.Steps - 1);
            increment_layer(&input_h_layer, l.Steps - 1);

            increment_layer(&state_z_layer, l.Steps - 1);
            increment_layer(&state_r_layer, l.Steps - 1);
            increment_layer(&state_h_layer, l.Steps - 1);

            state.Input += l.Inputs * l.Batch * (l.Steps - 1);
            if (state.Delta) state.Delta += l.Inputs * l.Batch * (l.Steps - 1);
            l.OutputGpu += l.Outputs * l.Batch * (l.Steps - 1);
            l.DeltaGpu += l.Outputs * l.Batch * (l.Steps - 1);
            for (i = l.Steps - 1; i >= 0; --i)
            {
                if (i != 0) Blas.copy_ongpu(l.Outputs * l.Batch, l.OutputGpu - l.Outputs * l.Batch, 1, l.PrevStateGpu, 1);
                float[] prev_delta_gpu = (i == 0) ? 0 : l.DeltaGpu - l.Outputs * l.Batch;

                Blas.copy_ongpu(l.Outputs * l.Batch, input_z_layer.OutputGpu, 1, l.ZGpu, 1);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_z_layer.OutputGpu, 1, l.ZGpu, 1);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_r_layer.OutputGpu, 1, l.RGpu, 1);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_r_layer.OutputGpu, 1, l.RGpu, 1);

                activate_array_ongpu(l.ZGpu, l.Outputs * l.Batch, LOGISTIC);
                activate_array_ongpu(l.RGpu, l.Outputs * l.Batch, LOGISTIC);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_h_layer.OutputGpu, 1, l.HGpu, 1);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_h_layer.OutputGpu, 1, l.HGpu, 1);

                // USET activate_array_ongpu(l.HGpu, l.Outputs * l.Batch, TANH);
                activate_array_ongpu(l.HGpu, l.Outputs * l.Batch, LOGISTIC);


                weighted_delta_gpu(l.PrevStateGpu, l.HGpu, l.ZGpu, prev_delta_gpu, input_h_layer.DeltaGpu, input_z_layer.DeltaGpu, l.Outputs * l.Batch, l.DeltaGpu);

                // USET gradient_array_ongpu(l.HGpu, l.Outputs * l.Batch, TANH, input_h_layer.DeltaGpu);
                gradient_array_ongpu(l.HGpu, l.Outputs * l.Batch, LOGISTIC, input_h_layer.DeltaGpu);


                Blas.copy_ongpu(l.Outputs * l.Batch, input_h_layer.DeltaGpu, 1, state_h_layer.DeltaGpu, 1);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.PrevStateGpu, 1, l.ForgotStateGpu, 1);
                mul_ongpu(l.Outputs * l.Batch, l.RGpu, 1, l.ForgotStateGpu, 1);
                fill_ongpu(l.Outputs * l.Batch, 0, l.ForgotDeltaGpu, 1);

                s.Input = l.ForgotStateGpu;
                s.Delta = l.ForgotDeltaGpu;

                backward_connected_layer_gpu(state_h_layer, s);
                if (prev_delta_gpu) mult_add_into_gpu(l.Outputs * l.Batch, l.ForgotDeltaGpu, l.RGpu, prev_delta_gpu);
                mult_add_into_gpu(l.Outputs * l.Batch, l.ForgotDeltaGpu, l.PrevStateGpu, input_r_layer.DeltaGpu);

                gradient_array_ongpu(l.RGpu, l.Outputs * l.Batch, LOGISTIC, input_r_layer.DeltaGpu);
                Blas.copy_ongpu(l.Outputs * l.Batch, input_r_layer.DeltaGpu, 1, state_r_layer.DeltaGpu, 1);

                gradient_array_ongpu(l.ZGpu, l.Outputs * l.Batch, LOGISTIC, input_z_layer.DeltaGpu);
                Blas.copy_ongpu(l.Outputs * l.Batch, input_z_layer.DeltaGpu, 1, state_z_layer.DeltaGpu, 1);

                s.Input = l.PrevStateGpu;
                s.Delta = prev_delta_gpu;

                backward_connected_layer_gpu(state_r_layer, s);
                backward_connected_layer_gpu(state_z_layer, s);

                s.Input = state.Input;
                s.Delta = state.Delta;

                backward_connected_layer_gpu(input_h_layer, s);
                backward_connected_layer_gpu(input_r_layer, s);
                backward_connected_layer_gpu(input_z_layer, s);


                state.Input -= l.Inputs * l.Batch;
                if (state.Delta) state.Delta -= l.Inputs * l.Batch;
                l.OutputGpu -= l.Outputs * l.Batch;
                l.DeltaGpu -= l.Outputs * l.Batch;
                increment_layer(&input_z_layer, -1);
                increment_layer(&input_r_layer, -1);
                increment_layer(&input_h_layer, -1);

                increment_layer(&state_z_layer, -1);
                increment_layer(&state_r_layer, -1);
                increment_layer(&state_h_layer, -1);
            }
        }

        int local_out_height(Layer l)
        {
            int h = l.H;
            if (!l.Pad) h -= l.Size;
            else h -= 1;
            return h / l.Stride + 1;
        }

        int local_out_width(Layer l)
        {
            int w = l.W;
            if (!l.Pad) w -= l.Size;
            else w -= 1;
            return w / l.Stride + 1;
        }

        Layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, Activation activation)
        {
            int i;
            Layer l = new Layer();
            l.LayerType = LayerType.Local;

            l.H = h;
            l.W = w;
            l.C = c;
            l.N = n;
            l.Batch = batch;
            l.Stride = stride;
            l.Size = size;
            l.Pad = pad;

            int out_h = local_out_height(l);
            int out_w = local_out_width(l);
            int locations = out_h * out_w;
            l.OutH = out_h;
            l.OutW = out_w;
            l.OutC = n;
            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = l.W * l.H * l.C;

            l.Weights = calloc(c * n * size * size * locations, sizeof(float));
            l.WeightUpdates = calloc(c * n * size * size * locations, sizeof(float));

            l.Biases = calloc(l.Outputs, sizeof(float));
            l.BiasUpdates = calloc(l.Outputs, sizeof(float));

            // float scale = 1./(float)Math.Sqrt(size*size*c);
            float scale = (float)Math.Sqrt(2./ (size * size * c));
            for (i = 0; i < c * n * size * size; ++i) l.Weights[i] = scale * rand_uniform(-1, 1);

            l.ColImage = calloc(out_h * out_w * size * size * c, sizeof(float));
            l.Output = calloc(l.Batch * out_h * out_w * n, sizeof(float));
            l.Delta = calloc(l.Batch * out_h * out_w * n, sizeof(float));

            l.Forward = forward_local_layer;
            l.Backward = backward_local_layer;
            l.Update = update_local_layer;


            l.ForwardGpu = forward_local_layer_gpu;
            l.BackwardGpu = backward_local_layer_gpu;
            l.UpdateGpu = update_local_layer_gpu;

            l.WeightsGpu = cuda_make_array(l.Weights, c * n * size * size * locations);
            l.WeightUpdatesGpu = cuda_make_array(l.WeightUpdates, c * n * size * size * locations);

            l.BiasesGpu = cuda_make_array(l.Biases, l.Outputs);
            l.BiasUpdatesGpu = cuda_make_array(l.BiasUpdates, l.Outputs);

            l.ColImageGpu = cuda_make_array(l.ColImage, out_h * out_w * size * size * c);
            l.DeltaGpu = cuda_make_array(l.Delta, l.Batch * out_h * out_w * n);
            l.OutputGpu = cuda_make_array(l.Output, l.Batch * out_h * out_w * n);


            l.Activation = activation;

            Console.Error.Write($"Local Layer: %d x %d x %d Image, %d filters . %d x %d x %d Image\n", h, w, c, n, out_h, out_w, n);

            return l;
        }

        void forward_local_layer(Layer l, NetworkState state)
        {
            int out_h = local_out_height(l);
            int out_w = local_out_width(l);
            int i, j;
            int locations = out_h * out_w;

            for (i = 0; i < l.Batch; ++i)
            {
                Blas.Copy_cpu(l.Outputs, l.Biases, 1, l.Output + i * l.Outputs, 1);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] input = state.Input + i * l.W * l.H * l.C;
                Im2Col.im2col_cpu(input, l.C, l.H, l.W,
                        l.Size, l.Stride, l.Pad, l.ColImage);
                float[] output = l.Output + i * l.Outputs;
                for (j = 0; j < locations; ++j)
                {
                    float[] a = l.Weights + j * l.Size * l.Size * l.C * l.N;
                    float[] b = l.ColImage + j;
                    float[] c = output + j;

                    int m = l.N;
                    int n = 1;
                    int k = l.Size * l.Size * l.C;

                    Gemm.gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                }
            }
            activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        void backward_local_layer(Layer l, NetworkState state)
        {
            int i, j;
            int locations = l.OutW * l.OutH;

            gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);

            for (i = 0; i < l.Batch; ++i)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.Delta + i * l.Outputs, 1, l.BiasUpdates, 1);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] input = state.Input + i * l.W * l.H * l.C;
                Im2Col.im2col_cpu(input, l.C, l.H, l.W,
                        l.Size, l.Stride, l.Pad, l.ColImage);

                for (j = 0; j < locations; ++j)
                {
                    float[] a = l.Delta + i * l.Outputs + j;
                    float[] b = l.ColImage + j;
                    float[] c = l.WeightUpdates + j * l.Size * l.Size * l.C * l.N;
                    int m = l.N;
                    int n = l.Size * l.Size * l.C;
                    int k = 1;

                    Gemm.gemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
                }

                if (state.Delta)
                {
                    for (j = 0; j < locations; ++j)
                    {
                        float[] a = l.Weights + j * l.Size * l.Size * l.C * l.N;
                        float[] b = l.Delta + i * l.Outputs + j;
                        float[] c = l.ColImage + j;

                        int m = l.Size * l.Size * l.C;
                        int n = 1;
                        int k = l.N;

                        Gemm.gemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                    }

                    Im2Col.Im2Col.col2im_cpu(l.ColImage, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, state.Delta + i * l.C * l.H * l.W);
                }
            }
        }

        void update_local_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            Blas.Axpy_cpu(l.Outputs, learning_rate / batch, l.BiasUpdates, 1, l.Biases, 1);
            Blas.Scal_cpu(l.Outputs, momentum, l.BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay * batch, l.Weights, 1, l.WeightUpdates, 1);
            Blas.Axpy_cpu(size, learning_rate / batch, l.WeightUpdates, 1, l.Weights, 1);
            Blas.Scal_cpu(size, momentum, l.WeightUpdates, 1);
        }



        void forward_local_layer_gpu(Layer l, NetworkState state)
        {
            int out_h = local_out_height(l);
            int out_w = local_out_width(l);
            int i, j;
            int locations = out_h * out_w;

            for (i = 0; i < l.Batch; ++i)
            {
                Blas.copy_ongpu(l.Outputs, l.BiasesGpu, 1, l.OutputGpu + i * l.Outputs, 1);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] input = state.Input + i * l.W * l.H * l.C;
                Im2Col.im2col_ongpu(input, l.C, l.H, l.W,
                        l.Size, l.Stride, l.Pad, l.ColImageGpu);
                float[] output = l.OutputGpu + i * l.Outputs;
                for (j = 0; j < locations; ++j)
                {
                    float[] a = l.WeightsGpu + j * l.Size * l.Size * l.C * l.N;
                    float[] b = l.ColImageGpu + j;
                    float[] c = output + j;

                    int m = l.N;
                    int n = 1;
                    int k = l.Size * l.Size * l.C;

                    Gemm.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                }
            }
            activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }

        void backward_local_layer_gpu(Layer l, NetworkState state)
        {
            int i, j;
            int locations = l.OutW * l.OutH;

            gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.axpy_ongpu(l.Outputs, 1, l.DeltaGpu + i * l.Outputs, 1, l.BiasUpdatesGpu, 1);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] input = state.Input + i * l.W * l.H * l.C;
                Im2Col.im2col_ongpu(input, l.C, l.H, l.W,
                        l.Size, l.Stride, l.Pad, l.ColImageGpu);

                for (j = 0; j < locations; ++j)
                {
                    float[] a = l.DeltaGpu + i * l.Outputs + j;
                    float[] b = l.ColImageGpu + j;
                    float[] c = l.WeightUpdatesGpu + j * l.Size * l.Size * l.C * l.N;
                    int m = l.N;
                    int n = l.Size * l.Size * l.C;
                    int k = 1;

                    Gemm.gemm_ongpu(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
                }

                if (state.Delta)
                {
                    for (j = 0; j < locations; ++j)
                    {
                        float[] a = l.WeightsGpu + j * l.Size * l.Size * l.C * l.N;
                        float[] b = l.DeltaGpu + i * l.Outputs + j;
                        float[] c = l.ColImageGpu + j;

                        int m = l.Size * l.Size * l.C;
                        int n = 1;
                        int k = l.N;

                        Gemm.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                    }

                    Im2Col.col2im_ongpu(l.ColImageGpu, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, state.Delta + i * l.C * l.H * l.W);
                }
            }
        }

        void update_local_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            Blas.axpy_ongpu(l.Outputs, learning_rate / batch, l.BiasUpdatesGpu, 1, l.BiasesGpu, 1);
            Blas.scal_ongpu(l.Outputs, momentum, l.BiasUpdatesGpu, 1);

            Blas.axpy_ongpu(size, -decay * batch, l.WeightsGpu, 1, l.WeightUpdatesGpu, 1);
            Blas.axpy_ongpu(size, learning_rate / batch, l.WeightUpdatesGpu, 1, l.WeightsGpu, 1);
            Blas.scal_ongpu(size, momentum, l.WeightUpdatesGpu, 1);
        }

        void pull_local_layer(Layer l)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            Array.Copy(l.WeightsGpu, l.Weights, size);
            Array.Copy(l.BiasesGpu, l.Biases, l.Outputs);
        }

        void push_local_layer(Layer l)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            cuda_push_array(l.WeightsGpu, l.Weights, size);
            cuda_push_array(l.BiasesGpu, l.Biases, l.Outputs);
        }

        Image get_maxpool_image(Layer l)
        {
            int h = l.OutH;
            int w = l.OutW;
            int c = l.C;
            return new Image(w, h, c, l.Output);
        }

        Image get_maxpool_delta(Layer l)
        {
            int h = l.OutH;
            int w = l.OutW;
            int c = l.C;
            return new Image(w, h, c, l.Delta);
        }

        Layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
        {
            Layer l = new Layer();
            l.LayerType = MAXPOOL;
            l.Batch = batch;
            l.H = h;
            l.W = w;
            l.C = c;
            l.Pad = padding;
            l.OutW = (w + 2 * padding) / stride;
            l.OutH = (h + 2 * padding) / stride;
            l.OutC = c;
            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = h * w * c;
            l.Size = size;
            l.Stride = stride;
            int output_size = l.OutH * l.OutW * l.OutC * batch;
            l.Indexes = calloc(output_size, sizeof(int));
            l.Output = calloc(output_size, sizeof(float));
            l.Delta = calloc(output_size, sizeof(float));
            l.Forward = forward_maxpool_layer;
            l.Backward = backward_maxpool_layer;

            l.ForwardGpu = forward_maxpool_layer_gpu;
            l.BackwardGpu = backward_maxpool_layer_gpu;
            l.IndexesGpu = cuda_make_int_array(output_size);
            l.OutputGpu = cuda_make_array(l.Output, output_size);
            l.DeltaGpu = cuda_make_array(l.Delta, output_size);

            Console.Error.Write($"max          %d x %d / %d  %4d x%4d x%4d   .  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.OutW, l.OutH, l.OutC);
            return l;
        }

        void resize_maxpool_layer(Layer l, int w, int h)
        {
            l.H = h;
            l.W = w;
            l.Inputs = h * w * l.C;

            l.OutW = (w + 2 * l.Pad) / l.Stride;
            l.OutH = (h + 2 * l.Pad) / l.Stride;
            l.Outputs = l.OutW * l.OutH * l.C;
            int output_size = l.Outputs * l.Batch;

            l.Indexes = realloc(l.Indexes, output_size * sizeof(int));
            l.Output = realloc(l.Output, output_size * sizeof(float));
            l.Delta = realloc(l.Delta, output_size * sizeof(float));

            l.IndexesGpu = cuda_make_int_array(output_size);
            l.OutputGpu = cuda_make_array(l.Output, output_size);
            l.DeltaGpu = cuda_make_array(l.Delta, output_size);

        }

        void forward_maxpool_layer(Layer l, NetworkState state)
        {
            int b, i, j, k, m, n;
            int w_offset = -l.Pad;
            int h_offset = -l.Pad;

            int h = l.OutH;
            int w = l.OutW;
            int c = l.C;

            for (b = 0; b < l.Batch; ++b)
            {
                for (k = 0; k < c; ++k)
                {
                    for (i = 0; i < h; ++i)
                    {
                        for (j = 0; j < w; ++j)
                        {
                            int out_index = j + w * (i + h * (k + c * b));
                            float max = float.MinValue;
                            int max_i = -1;
                            for (n = 0; n < l.Size; ++n)
                            {
                                for (m = 0; m < l.Size; ++m)
                                {
                                    int cur_h = h_offset + i * l.Stride + n;
                                    int cur_w = w_offset + j * l.Stride + m;
                                    int index = cur_w + l.W * (cur_h + l.H * (k + b * l.C));
                                    int valid = (cur_h >= 0 && cur_h < l.H &&
                                                 cur_w >= 0 && cur_w < l.W);
                                    float val = (valid != 0) ? state.Input[index] : float.MinValue;
                                    max_i = (val > max) ? index : max_i;
                                    max = (val > max) ? val : max;
                                }
                            }
                            l.Output[out_index] = max;
                            l.Indexes[out_index] = max_i;
                        }
                    }
                }
            }
        }

        void backward_maxpool_layer(Layer l, NetworkState state)
        {
            int i;
            int h = l.OutH;
            int w = l.OutW;
            int c = l.C;
            for (i = 0; i < h * w * c * l.Batch; ++i)
            {
                int index = l.Indexes[i];
                state.Delta[index] += l.Delta[i];
            }
        }

        Layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
        {
            Console.Error.Write($"Local Response Normalization Layer: %d x %d x %d Image, %d size\n", w, h, c, size);
            Layer layer = new Layer();
            layer.LayerType = LayerType.Normalization;
            layer.Batch = batch;
            layer.H = layer.OutH = h;
            layer.W = layer.OutW = w;
            layer.C = layer.OutC = c;
            layer.Kappa = kappa;
            layer.Size = size;
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.Output = calloc(h * w * c * batch, sizeof(float));
            layer.Delta = calloc(h * w * c * batch, sizeof(float));
            layer.Squared = calloc(h * w * c * batch, sizeof(float));
            layer.Norms = calloc(h * w * c * batch, sizeof(float));
            layer.Inputs = w * h * c;
            layer.Outputs = layer.Inputs;

            layer.Forward = forward_normalization_layer;
            layer.Backward = backward_normalization_layer;

            layer.ForwardGpu = forward_normalization_layer_gpu;
            layer.BackwardGpu = backward_normalization_layer_gpu;

            layer.OutputGpu = cuda_make_array(layer.Output, h * w * c * batch);
            layer.DeltaGpu = cuda_make_array(layer.Delta, h * w * c * batch);
            layer.SquaredGpu = cuda_make_array(layer.Squared, h * w * c * batch);
            layer.NormsGpu = cuda_make_array(layer.Norms, h * w * c * batch);

            return layer;
        }

        void resize_normalization_layer(Layer layer, int w, int h)
        {
            int c = layer.C;
            int batch = layer.Batch;
            layer.H = h;
            layer.W = w;
            layer.OutH = h;
            layer.OutW = w;
            layer.Inputs = w * h * c;
            layer.Outputs = layer.Inputs;
            layer.Output = realloc(layer.Output, h * w * c * batch * sizeof(float));
            layer.Delta = realloc(layer.Delta, h * w * c * batch * sizeof(float));
            layer.Squared = realloc(layer.Squared, h * w * c * batch * sizeof(float));
            layer.Norms = realloc(layer.Norms, h * w * c * batch * sizeof(float));

            layer.OutputGpu = cuda_make_array(layer.Output, h * w * c * batch);
            layer.DeltaGpu = cuda_make_array(layer.Delta, h * w * c * batch);
            layer.SquaredGpu = cuda_make_array(layer.Squared, h * w * c * batch);
            layer.NormsGpu = cuda_make_array(layer.Norms, h * w * c * batch);

        }

        void forward_normalization_layer(Layer layer, NetworkState state)
        {
            int k, b;
            int w = layer.W;
            int h = layer.H;
            int c = layer.C;
            Blas.Scal_cpu(w * h * c * layer.Batch, 0, layer.Squared, 1);

            for (b = 0; b < layer.Batch; ++b)
            {
                float[] squared = layer.Squared + w * h * c * b;
                float[] norms = layer.Norms + w * h * c * b;
                float[] input = state.Input + w * h * c * b;
                Blas.Pow_cpu(w * h * c, 2, input, 1, squared, 1);

                Blas.Const_cpu(w * h, layer.Kappa, norms, 1);
                for (k = 0; k < layer.Size / 2; ++k)
                {
                    Blas.Axpy_cpu(w * h, layer.Alpha, squared + w * h * k, 1, norms, 1);
                }

                for (k = 1; k < layer.C; ++k)
                {
                    Blas.Copy_cpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
                    int prev = k - ((layer.Size - 1) / 2) - 1;
                    int next = k + (layer.Size / 2);
                    if (prev >= 0) Blas.Axpy_cpu(w * h, -layer.Alpha, squared + w * h * prev, 1, norms + w * h * k, 1);
                    if (next < layer.C) Blas.Axpy_cpu(w * h, layer.Alpha, squared + w * h * next, 1, norms + w * h * k, 1);
                }
            }
            Blas.Pow_cpu(w * h * c * layer.Batch, -layer.Beta, layer.Norms, 1, layer.Output, 1);
            Blas.Mul_cpu(w * h * c * layer.Batch, state.Input, 1, layer.Output, 1);
        }

        void backward_normalization_layer(Layer layer, NetworkState state)
        {
            // TODO This is approximate ;-)
            // Also this should add in to delta instead of overwritting.

            int w = layer.W;
            int h = layer.H;
            int c = layer.C;
            Blas.Pow_cpu(w * h * c * layer.Batch, -layer.Beta, layer.Norms, 1, state.Delta, 1);
            Blas.Mul_cpu(w * h * c * layer.Batch, layer.Delta, 1, state.Delta, 1);
        }


        void forward_normalization_layer_gpu(Layer layer, NetworkState state)
        {
            int k, b;
            int w = layer.W;
            int h = layer.H;
            int c = layer.C;
            Blas.scal_ongpu(w * h * c * layer.Batch, 0, layer.SquaredGpu, 1);

            for (b = 0; b < layer.Batch; ++b)
            {
                float[] squared = layer.SquaredGpu + w * h * c * b;
                float[] norms = layer.NormsGpu + w * h * c * b;
                float[] input = state.Input + w * h * c * b;
                pow_ongpu(w * h * c, 2, input, 1, squared, 1);

                const_ongpu(w * h, layer.Kappa, norms, 1);
                for (k = 0; k < layer.Size / 2; ++k)
                {
                    Blas.axpy_ongpu(w * h, layer.Alpha, squared + w * h * k, 1, norms, 1);
                }

                for (k = 1; k < layer.C; ++k)
                {
                    Blas.copy_ongpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
                    int prev = k - ((layer.Size - 1) / 2) - 1;
                    int next = k + (layer.Size / 2);
                    if (prev >= 0) Blas.axpy_ongpu(w * h, -layer.Alpha, squared + w * h * prev, 1, norms + w * h * k, 1);
                    if (next < layer.C) Blas.axpy_ongpu(w * h, layer.Alpha, squared + w * h * next, 1, norms + w * h * k, 1);
                }
            }
            pow_ongpu(w * h * c * layer.Batch, -layer.Beta, layer.NormsGpu, 1, layer.OutputGpu, 1);
            mul_ongpu(w * h * c * layer.Batch, state.Input, 1, layer.OutputGpu, 1);
        }

        void backward_normalization_layer_gpu(Layer layer, NetworkState state)
        {
            // TODO This is approximate ;-)

            int w = layer.W;
            int h = layer.H;
            int c = layer.C;
            pow_ongpu(w * h * c * layer.Batch, -layer.Beta, layer.NormsGpu, 1, state.Delta, 1);
            mul_ongpu(w * h * c * layer.Batch, layer.DeltaGpu, 1, state.Delta, 1);
        }

    }
}