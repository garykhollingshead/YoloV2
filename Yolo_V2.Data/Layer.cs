using System;
using System.Collections.Generic;
using System.Linq;
using Alea;
using Alea.cuDNN;
using Alea.CSharp;
using Alea.CudaDnn;
using Alea.CudaToolkit;
using Yolo_V2.Data.Enums;
using dim3 = Alea.dim3;

namespace Yolo_V2.Data
{
    public class Layer
    {
        public LayerType LayerType;
        public Activation Activation;
        public CostType CostType;
        public Action<Layer, NetworkState> Forward;
        public Action<Layer, NetworkState> Backward;
        public Action<Layer, int, float, float, float> Update;
        public Action<Layer, NetworkState> ForwardGpu;
        public Action<Layer, NetworkState> BackwardGpu;
        public Action<Layer, int, float, float, float> UpdateGpu;
        public bool BatchNormalize;
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
        public bool Flip;
        public int Index;
        public bool Binary;
        public bool Xnor;
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
        public bool Rescore;
        public int Objectness;
        public int DoesCost;
        public int Joint;
        public bool Noadjust;
        public int Reorg;
        public int Log;

        public bool Adam;
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
        public float? Cost;
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
        public cudnnTensorStruct SrcTensorDesc, DstTensorDesc;
        public cudnnTensorStruct DsrcTensorDesc, DdstTensorDesc;
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
            if (state.Train)
            {
                Blas.Mean_cpu(Output, Batch, OutC, OutH * OutW, Mean);
                Blas.Variance_cpu(Output, Mean, Batch, OutC, OutH * OutW, Variance);

                Blas.Scal_cpu(OutC, .9f, RollingMean, 1);
                Blas.Axpy_cpu(OutC, .1f, Mean, RollingMean);
                Blas.Scal_cpu(OutC, .9f, RollingVariance, 1);
                Blas.Axpy_cpu(OutC, .1f, Variance, RollingVariance);

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

        public void pull_batchnorm_layer()
        {
            Array.Copy(ScalesGpu, Scales, C);
            Array.Copy(RollingMeanGpu, RollingMean, C);
            Array.Copy(RollingVarianceGpu, RollingVariance, C);
        }

        public void push_batchnorm_layer()
        {
            Array.Copy(Scales, ScalesGpu, C);
            Array.Copy(RollingMean, RollingMeanGpu, C);
            Array.Copy(RollingVariance, RollingVarianceGpu, C);
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
            if (state.Train)
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
                Blas.Axpy_cpu(Outputs, 1, Delta.Skip(index).ToArray(), BiasUpdates);
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
                    Im2Col.col2im_cpu(ColImage, C, H, W, Size, Stride, Pad, output);
                    CombineLists(state.Delta, index, output);
                }
            }
        }

        private void update_local_layer(int batch, float learning_rate, float momentum, float decay)
        {
            int locations = OutW * OutH;
            int size = Size * Size * C * N * locations;
            Blas.Axpy_cpu(Outputs, learning_rate / batch, BiasUpdates, Biases);
            Blas.Scal_cpu(Outputs, momentum, BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay * batch, Weights, WeightUpdates);
            Blas.Axpy_cpu(size, learning_rate / batch, WeightUpdates, Weights);
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

        public void swap_binary()
        {
            float[] swap = Weights;
            Weights = BinaryWeights;
            BinaryWeights = swap;

            swap = WeightsGpu;
            WeightsGpu = BinaryWeightsGpu;
            BinaryWeightsGpu = swap;
        }
        
        public static void binarize_weights(float[] weights, int n, int size, float[] binary)
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
        
        public static void binarize_cpu(float[] input, int n, float[] binary)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                binary[i] = (input[i] > 0) ? 1 : -1;
            }
        }
        
        public static void binarize_input(float[] input, int n, int size, float[] binary)
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
        
        public int convolutional_out_height()
        {
            return (H + 2 * Pad - Size) / Stride + 1;
        }
        
        public int convolutional_out_width()
        {
            return (W + 2 * Pad - Size) / Stride + 1;
        }
        
        public Image get_convolutional_image()
        {
            int h, w, c;
            h = convolutional_out_height();
            w = convolutional_out_width();
            c = N;
            return new Image(w, h, c, Output);
        }
        
        public Image get_convolutional_delta()
        {
            int h, w, c;
            h = convolutional_out_height();
            w = convolutional_out_width();
            c = N;
            return new Image(w, h, c, Delta);
        }
        
        public unsafe ulong get_workspace_size()
        {
            if (CudaUtils.UseGpu)
            {
                ulong most = 0;
                ulong s = 0;
                using (var gpuSrcTensorDesc = Alea.Interop.Marshal.Align(SrcTensorDesc))
                using (var gpuWeightDesc = Alea.Interop.Marshal.Align(WeightDesc))
                using (var gpuConvDesc = Alea.Interop.Marshal.Align(ConvDesc))
                using (var gpuDstTensorDesc = Alea.Interop.Marshal.Align(DstTensorDesc))
                using (var gpuDdstTensorDesc = Alea.Interop.Marshal.Align(DdstTensorDesc))
                using (var gpuDweightDesc = Alea.Interop.Marshal.Align(DweightDesc))
                using (var gpuDsrcTensorDesc = Alea.Interop.Marshal.Align(DsrcTensorDesc))
                {
                    CuDnn.cudnnGetConvolutionForwardWorkspaceSize(CudaUtils.cudnn_handle(),
                        (cudnnTensorStruct*)gpuSrcTensorDesc.Handle,
                        (cudnnFilterStruct*)gpuWeightDesc.Handle,
                        (cudnnConvolutionStruct*)gpuConvDesc.Handle,
                        (cudnnTensorStruct*)gpuDstTensorDesc.Handle,
                        FwAlgo,
                        &s);
                    if (s > most) most = s;
                    CuDnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(CudaUtils.cudnn_handle(),
                        (cudnnTensorStruct*)gpuSrcTensorDesc.Handle,
                        (cudnnTensorStruct*)gpuDdstTensorDesc.Handle,
                        (cudnnConvolutionStruct*)gpuConvDesc.Handle,
                        (cudnnFilterStruct*)gpuDweightDesc.Handle,
                        BfAlgo,
                        &s);
                    if (s > most) most = s;
                    CuDnn.cudnnGetConvolutionBackwardDataWorkspaceSize(CudaUtils.cudnn_handle(),
                        (cudnnFilterStruct*)gpuWeightDesc.Handle,
                        (cudnnTensorStruct*)gpuDdstTensorDesc.Handle,
                        (cudnnConvolutionStruct*)gpuConvDesc.Handle,
                        (cudnnTensorStruct*)gpuDsrcTensorDesc.Handle,
                        BdAlgo,
                        &s);
                }
                if (s > most) most = s;
                return most;
            }
            return (ulong)(OutH * OutW * Size * Size * C * sizeof(float));
        }
        
        public unsafe void cudnn_convolutional_setup()
        {
            using (var gpuDsrcTensorDesc = Alea.Interop.Marshal.Align(DsrcTensorDesc))
            using (var gpuDdstTensorDesc = Alea.Interop.Marshal.Align(DdstTensorDesc))
            using (var gpuDweightDesc = Alea.Interop.Marshal.Align(DweightDesc))
            using (var gpuSrcTensorDesc = Alea.Interop.Marshal.Align(SrcTensorDesc))
            using (var gpuDstTensorDesc = Alea.Interop.Marshal.Align(DstTensorDesc))
            using (var gpuWeightDesc = Alea.Interop.Marshal.Align(WeightDesc))
            using (var gpuConvDesc = Alea.Interop.Marshal.Align(ConvDesc))
            using (var gpuFwAlgo = Alea.Interop.Marshal.Align(FwAlgo))
            using (var gpuBdAlgo = Alea.Interop.Marshal.Align(BdAlgo))
            using (var gpuBfAlgo = Alea.Interop.Marshal.Align(BfAlgo))
            {
                CuDnn.cudnnSetTensor4dDescriptor((cudnnTensorStruct*)gpuDsrcTensorDesc.Handle, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, cudnnDataType_t.CUDNN_DATA_FLOAT, Batch, C,
                    H, W);
                CuDnn.cudnnSetTensor4dDescriptor((cudnnTensorStruct*)gpuDdstTensorDesc.Handle, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, cudnnDataType_t.CUDNN_DATA_FLOAT, Batch, OutC,
                    OutH, OutW);
                CuDnn.cudnnSetFilter4dDescriptor((cudnnFilterStruct*)gpuDweightDesc.Handle, cudnnDataType_t.CUDNN_DATA_FLOAT, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, N, C, Size,
                    Size);

                CuDnn.cudnnSetTensor4dDescriptor((cudnnTensorStruct*)gpuSrcTensorDesc.Handle, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, cudnnDataType_t.CUDNN_DATA_FLOAT, Batch, C,
                    H, W);
                CuDnn.cudnnSetTensor4dDescriptor((cudnnTensorStruct*)gpuDstTensorDesc.Handle, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, cudnnDataType_t.CUDNN_DATA_FLOAT, Batch, OutC,
                    OutH, OutW);
                CuDnn.cudnnSetFilter4dDescriptor((cudnnFilterStruct*)gpuWeightDesc.Handle, cudnnDataType_t.CUDNN_DATA_FLOAT, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, N, C, Size,
                    Size);
                CuDnn.cudnnSetConvolution2dDescriptor((cudnnConvolutionStruct*)gpuConvDesc.Handle, Pad, Pad, Stride, Stride, 1, 1,
                    cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION, cudnnDataType_t.CUDNN_DATA_FLOAT);
                CuDnn.cudnnGetConvolutionForwardAlgorithm(CudaUtils.cudnn_handle(),
                    (cudnnTensorStruct*)gpuSrcTensorDesc.Handle,
                    (cudnnFilterStruct*)gpuWeightDesc.Handle,
                    (cudnnConvolutionStruct*)gpuConvDesc.Handle,
                    (cudnnTensorStruct*)gpuDstTensorDesc.Handle,
                    cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0,
                    (cudnnConvolutionFwdAlgo_t*)gpuFwAlgo.Handle);
                CuDnn.cudnnGetConvolutionBackwardDataAlgorithm(CudaUtils.cudnn_handle(),
                    (cudnnFilterStruct*)gpuWeightDesc.Handle,
                    (cudnnTensorStruct*)gpuDdstTensorDesc.Handle,
                    (cudnnConvolutionStruct*)gpuConvDesc.Handle,
                    (cudnnTensorStruct*)gpuDsrcTensorDesc.Handle,
                    cudnnConvolutionBwdDataPreference_t.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                    0,
                    (cudnnConvolutionBwdDataAlgo_t*)gpuBdAlgo.Handle);
                CuDnn.cudnnGetConvolutionBackwardFilterAlgorithm(CudaUtils.cudnn_handle(),
                    (cudnnTensorStruct*)gpuSrcTensorDesc.Handle,
                    (cudnnTensorStruct*)gpuDdstTensorDesc.Handle,
                    (cudnnConvolutionStruct*)gpuConvDesc.Handle,
                    (cudnnFilterStruct*)gpuDweightDesc.Handle,
                    cudnnConvolutionBwdFilterPreference_t.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                    0,
                    (cudnnConvolutionBwdFilterAlgo_t*)gpuBfAlgo.Handle);
            }
        }
        
        public static Layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, 
            Activation activation, bool batch_normalize, bool binary, bool xnor, bool adam)
        {
            int i;
            Layer l = new Layer();
            l.LayerType = LayerType.Convolutional;

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

            l.Weights = new float[c * n * size * size];
            l.WeightUpdates = new float[c * n * size * size];

            l.Biases = new float[n];
            l.BiasUpdates = new float[n];

            // float scale = 1./(float)Math.Sqrt(size*size*c);
            float scale = (float)Math.Sqrt(2.0 / (size * size * c));
            for (i = 0; i < c * n * size * size; ++i) l.Weights[i] = scale * Utils.rand_uniform(-1, 1);
            int out_h = l.convolutional_out_height();
            int out_w = l.convolutional_out_width();
            l.OutH = out_h;
            l.OutW = out_w;
            l.OutC = n;
            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = l.W * l.H * l.C;

            l.Output = new float[l.Batch * l.Outputs];
            l.Delta = new float[l.Batch * l.Outputs];

            l.Forward = forward_convolutional_layer;
            l.Backward = backward_convolutional_layer;
            l.Update = update_convolutional_layer;
            if (binary)
            {
                l.BinaryWeights = new float[c * n * size * size];
                l.Scales = new float[n];
            }
            if (xnor)
            {
                l.BinaryWeights = new float[c * n * size * size];
                l.BinaryInput = new float[l.Inputs * l.Batch];
            }

            if (batch_normalize)
            {
                l.Scales = new float[n];
                l.ScaleUpdates = new float[n];
                for (i = 0; i < n; ++i)
                {
                    l.Scales[i] = 1;
                }

                l.Mean = new float[n];
                l.Variance = new float[n];

                l.MeanDelta = new float[n];
                l.VarianceDelta = new float[n];

                l.RollingMean = new float[n];
                l.RollingVariance = new float[n];
                l.X = new float[l.Batch * l.Outputs];
                l.XNorm = new float[l.Batch * l.Outputs];
            }
            if (adam != 0)
            {
                l.Adam = true;
                l.M = new float[c * n * size * size];
                l.V = new float[c * n * size * size];
            }

            l.ForwardGpu = forward_convolutional_layer_gpu;
            l.BackwardGpu = backward_convolutional_layer_gpu;
            l.UpdateGpu = update_convolutional_layer_gpu;

            if (CudaUtils.UseGpu)
            {
                if (adam != 0)
                {
                    l.MGpu = (float[])l.M.Clone();
                    l.VGpu = (float[])l.V.Clone();
                }

                l.WeightsGpu = (float[])l.Weights.Clone();
                l.WeightUpdatesGpu = (float[])l.WeightUpdates.Clone();

                l.BiasesGpu = (float[])l.Biases.Clone();
                l.BiasUpdatesGpu = (float[])l.BiasUpdates.Clone();

                l.DeltaGpu = (float[])l.Delta.Clone();
                l.OutputGpu = (float[])l.Output.Clone();

                if (binary)
                {
                    l.BinaryWeightsGpu = (float[])l.Weights.Clone();
                }
                if (xnor)
                {
                    l.BinaryWeightsGpu = (float[])l.Weights.Clone();
                    l.BinaryInputGpu = new float[l.Inputs * l.Batch];
                }

                if (batch_normalize)
                {
                    l.MeanGpu = (float[])l.Mean.Clone();
                    l.VarianceGpu = (float[])l.Variance.Clone();

                    l.RollingMeanGpu = (float[])l.Mean.Clone();
                    l.RollingVarianceGpu = (float[])l.Variance.Clone();

                    l.MeanDeltaGpu = (float[])l.Mean.Clone();
                    l.VarianceDeltaGpu = (float[])l.Variance.Clone();

                    l.ScalesGpu = (float[])l.Scales.Clone();
                    l.ScaleUpdatesGpu = (float[])l.ScaleUpdates.Clone();

                    l.XGpu = (float[])l.Output.Clone();
                    l.XNormGpu = (float[])l.Output.Clone();
                }

                unsafe
                {
                    using (var SrcTensorDesc = Alea.Interop.Marshal.Align(l.SrcTensorDesc))
                    using (var DstTensorDesc = Alea.Interop.Marshal.Align(l.DstTensorDesc))
                    using (var WeightDesc = Alea.Interop.Marshal.Align(l.WeightDesc))
                    using (var DsrcTensorDesc = Alea.Interop.Marshal.Align(l.DsrcTensorDesc))
                    using (var DdstTensorDesc = Alea.Interop.Marshal.Align(l.DdstTensorDesc))
                    using (var DweightDesc = Alea.Interop.Marshal.Align(l.DweightDesc))
                    using (var ConvDesc = Alea.Interop.Marshal.Align(l.ConvDesc))
                    {
                        CuDnn.cudnnCreateTensorDescriptor((cudnnTensorStruct**)SrcTensorDesc.Handle);
                        CuDnn.cudnnCreateTensorDescriptor((cudnnTensorStruct**)DstTensorDesc.Handle);
                        CuDnn.cudnnCreateFilterDescriptor((cudnnFilterStruct**)WeightDesc.Handle);
                        CuDnn.cudnnCreateTensorDescriptor((cudnnTensorStruct**)DsrcTensorDesc.Handle);
                        CuDnn.cudnnCreateTensorDescriptor((cudnnTensorStruct**)DdstTensorDesc.Handle);
                        CuDnn.cudnnCreateFilterDescriptor((cudnnFilterStruct**)DweightDesc.Handle);
                        CuDnn.cudnnCreateConvolutionDescriptor((cudnnConvolutionStruct**)ConvDesc.Handle);
                    }
                }
                cudnn_convolutional_setup(l);
            }
            l.WorkspaceSize = get_workspace_size(l);
            l.Activation = activation;

            Console.Error.Write($"conv  {n:5} {size:2} x{size:2} /{stride:2}  {w:4} x{h:4} x{c:4}   .  {l.OutW:4} x{l.OutH:4} x{l.OutC:4}\n", n, size, size, stride, w, h, c, l.OutW, l.OutH, l.OutC);

            return l;
        }
        
        public void denormalize_convolutional_layer()
        {
            int i, j;
            for (i = 0; i < N; ++i)
            {
                float scale = Scales[i] / (float)Math.Sqrt(RollingVariance[i] + .00001);
                for (j = 0; j < C * Size * Size; ++j)
                {
                    Weights[i * C * Size * Size + j] *= scale;
                }
                Biases[i] -= RollingMean[i] * scale;
                Scales[i] = 1;
                RollingMean[i] = 0;
                RollingVariance[i] = 1;
            }
        }
        
        public static void test_convolutional_layer()
        {
            Layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, Activation.Leaky, true, false, false, 0);
            l.BatchNormalize = true;
            float[] data = {1f,1f,1f,1f,1f,
                1f,1f,1f,1f,1f,
                1f,1f,1f,1f,1f,
                1f,1f,1f,1f,1f,
                1f,1f,1f,1f,1f,
                2f,2f,2f,2f,2f,
                2f,2f,2f,2f,2f,
                2f,2f,2f,2f,2f,
                2f,2f,2f,2f,2f,
                2f,2f,2f,2f,2f,
                3f,3f,3f,3f,3f,
                3f,3f,3f,3f,3f,
                3f,3f,3f,3f,3f,
                3f,3f,3f,3f,3f,
                3f,3f,3f,3f,3f};
            NetworkState state = new NetworkState();
            state.Input = data;
            l.forward_convolutional_layer(state);
        }
        
        public void resize_convolutional_layer( int w, int h)
        {
            W = w;
            H = h;
            int out_w = convolutional_out_width();
            int out_h = convolutional_out_height();

            OutW = out_w;
            OutH = out_h;

            Outputs = OutH * OutW * OutC;
            Inputs = W * H * C;

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
            cudnn_convolutional_setup();
            WorkspaceSize = get_workspace_size();
        }
        
        public static void add_bias(float[] output, float[] biases, int batch, int n, int size)
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
        
        public static void scale_bias(float[] output, float[] scales, int batch, int n, int size)
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
        
        public static void backward_bias(float[] bias_updates, float[] delta, int batch, int n, int size)
        {
            int i, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    var part = new float[size];
                    Array.Copy(delta, size * (i + b * n), part, 0, size);
                    bias_updates[i] += part.Sum();
                }
            }
        }
        
        public static void forward_convolutional_layer(Layer l, NetworkState state)
        {
            int out_h = l.convolutional_out_height();
            int out_w = l.convolutional_out_width();
            int i;

            Blas.Fill_cpu(l.Outputs * l.Batch, 0, l.Output, 1);

            if (l.Xnor)
            {
                binarize_weights(l.Weights, l.N, l.C * l.Size * l.Size, l.BinaryWeights);
                l.swap_binary();
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
                c[i] += n * m;
                state.Input[i] += l.C * l.H * l.W;
            }

            if (l.BatchNormalize)
            {
                l.forward_batchnorm_layer(state);
            }
            add_bias(l.Output, l.Biases, l.Batch, l.N, out_h * out_w);

            ActivationsHelper.Activate_array(l.Output, m * n * l.Batch, l.Activation);
            if (l.Binary || l.Xnor) l.swap_binary();
        }
        
        public void backward_convolutional_layer(NetworkState state)
        {
            int i;
            int m = N;
            int n = Size * Size * C;
            int k = convolutional_out_height(this) *
                convolutional_out_width(this);

            ActivationsHelper.Gradient_array(Output, m * k * Batch, Activation, Delta);
            backward_bias(BiasUpdates, Delta, Batch, N, k);

            if (BatchNormalize)
            {
                backward_batchnorm_layer(state);
            }

            for (i = 0; i < Batch; ++i)
            {
                float[] a = Delta.Skip(i * m * k).ToArray();
                float[] b = state.Workspace;
                float[] c = WeightUpdates;

                float[] im = state.Input.Skip(i * C * H * W).ToArray();

                Im2Col.im2col_cpu(im, C, H, W,
                        Size, Stride, Pad, b);
                Gemm.gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

                Array.Copy(a, 0, Delta, i * m * k, a.Length);
                Array.Copy(im, 0, state.Input, i * C * H * W, im.Length);
                if (state.Delta.Any())
                {
                    a = Weights;
                    b = Delta.Skip(i * m * k).ToArray();
                    Array.Copy(Delta, i * m * k, b, 0, b.Length);
                    c = state.Workspace;
                    var d = state.Delta.Skip(i * C * H * W).ToArray();

                    Gemm.gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

                    Im2Col.col2im_cpu(state.Workspace, C, H, W, Size, Stride, Pad, d);
                    Array.Copy(b, 0, Delta, i * m * k, b.Length);
                    Array.Copy(d, 0, state.Delta, i * C * H * W, d.Length);

                }
            }
        }
        
        public void update_convolutional_layer(int batch, float learning_rate, float momentum, float decay)
        {
            int size = Size * Size * C * N;
            Blas.Axpy_cpu(N, learning_rate / batch, BiasUpdates, Biases);
            Blas.Scal_cpu(N, momentum, BiasUpdates, 1);

            if (Scales.Any())
            {
                Blas.Axpy_cpu(N, learning_rate / batch, ScaleUpdates, Scales);
                Blas.Scal_cpu(N, momentum, ScaleUpdates, 1);
            }

            Blas.Axpy_cpu(size, -decay * batch, Weights, WeightUpdates);
            Blas.Axpy_cpu(size, learning_rate / batch, WeightUpdates, Weights);
            Blas.Scal_cpu(size, momentum, WeightUpdates, 1);
        }
        
        public Image get_convolutional_weight( int i)
        {
            int h = Size;
            int w = Size;
            int c = C;
            return new Image(w, h, c, Weights.Skip(i * h * w * c).ToArray());
        }
        
        public void rgbgr_weights()
        {
            int i;
            for (i = 0; i < N; ++i)
            {
                Image im = get_convolutional_weight(l, i);
                if (im.C == 3)
                {
                    LoadArgs.rgbgr_image(im);
                }
            }
        }
        
        public void rescale_weights( float scale, float trans)
        {
            int i;
            for (i = 0; i < N; ++i)
            {
                Image im = get_convolutional_weight( i);
                if (im.C == 3)
                {
                    LoadArgs.scale_image(im, scale);
                    float sum = im.Data.Sum();
                    Biases[i] += sum * trans;
                }
            }
        }
        
        public Image[] get_weights()
        {
            Image[] weights = new Image[N];
            int i;
            for (i = 0; i < N; ++i)
            {
                weights[i] = new Image(get_convolutional_weight(l, i));
            }
            return weights;
        }
        
        public Image[] visualize_convolutional_layer( string window)
        {
            Image[] single_weights = get_weights();
            LoadArgs.show_images(single_weights, N, window);

            Image delta = get_convolutional_image();
            Image dc = LoadArgs.collapse_image_layers(delta, 1);
            string buff = $"{window}: Output";
            return single_weights;
        }
        
        public static Layer make_activation_layer(int batch, int inputs, Activation activation)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Active;

            l.Inputs = inputs;
            l.Outputs = inputs;
            l.Batch = batch;

            l.Output = new float[batch];
            l.Delta = new float[batch];

            l.Forward = forward_activation_layer;
            l.Backward = backward_activation_layer;
            l.ForwardGpu = forward_activation_layer_gpu;
            l.BackwardGpu = backward_activation_layer_gpu;

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            l.Activation = activation;
            Console.Error.Write($"Activation Layer: %d inputs\n", inputs);
            return l;
        }
        
        public void forward_activation_layer(NetworkState state)
        {
            Blas.Copy_cpu(Outputs * Batch, state.Input, 1, Output, 1);
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }
        
        public void backward_activation_layer(NetworkState state)
        {
            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);
            Blas.Copy_cpu(Outputs * Batch, Delta, 1, state.Delta, 1);
        }
        
        public void forward_activation_layer_gpu(NetworkState state)
        {
            Blas.copy_ongpu(Outputs * Batch, state.Input, 1, OutputGpu, 1);
            ActivationsHelper.activate_array_ongpu(OutputGpu, Outputs * Batch, Activation);
        }
        
        public void backward_activation_layer_gpu(NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, DeltaGpu);
            Blas.copy_ongpu(Outputs * Batch, DeltaGpu, 1, state.Delta, 1);
        }
        
        public static Layer make_avgpool_layer(int batch, int w, int h, int c)
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
            l.Output = new float[output_size];
            l.Delta = new float[output_size];
            l.Forward = forward_avgpool_layer;
            l.Backward = backward_avgpool_layer;
            l.ForwardGpu = forward_avgpool_layer_gpu;
            l.BackwardGpu = backward_avgpool_layer_gpu;
            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            return l;
        }
        
        public void resize_avgpool_layer(int w, int h)
        {
            W = w;
            H = h;
            Inputs = h * w * C;
        }
        
        public void forward_avgpool_layer(NetworkState state)
        {
            int b, i, k;

            for (b = 0; b < Batch; ++b)
            {
                for (k = 0; k < C; ++k)
                {
                    int out_index = k + b * C;
                    Output[out_index] = 0;
                    for (i = 0; i < H * W; ++i)
                    {
                        int in_index = i + H * W * (k + b * C);
                        Output[out_index] += state.Input[in_index];
                    }
                    Output[out_index] /= H * W;
                }
            }
        }
        
        public void backward_avgpool_layer(NetworkState state)
        {
            int b, i, k;

            for (b = 0; b < Batch; ++b)
            {
                for (k = 0; k < C; ++k)
                {
                    int out_index = k + b * C;
                    for (i = 0; i < H * W; ++i)
                    {
                        int in_index = i + H * W * (k + b * C);
                        state.Delta[in_index] += Delta[out_index] / (H * W);
                    }
                }
            }
        }
        
        public static Layer make_connected_layer(int batch, int inputs, int outputs, Activation activation, bool batch_normalize)
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

            l.Output = new float[batch * outputs];
            l.Delta = new float[batch * outputs];

            l.WeightUpdates = new float[inputs * outputs];
            l.BiasUpdates = new float[outputs];

            l.Weights = new float[outputs * inputs]);
            l.Biases = new float[outputs];

            l.Forward = forward_connected_layer;
            l.Backward = backward_connected_layer;
            l.Update = update_connected_layer;

            float scale = (float)Math.Sqrt(2.0 / inputs);
            for (i = 0; i < outputs * inputs; ++i)
            {
                l.Weights[i] = scale * Utils.rand_uniform(-1, 1);
            }

            for (i = 0; i < outputs; ++i)
            {
                l.Biases[i] = 0;
            }

            if (batch_normalize)
            {
                l.Scales = new float[outputs];
                l.ScaleUpdates = new float[outputs];
                for (i = 0; i < outputs; ++i)
                {
                    l.Scales[i] = 1;
                }

                l.Mean = new float[outputs];
                l.MeanDelta = new float[outputs];
                l.Variance = new float[outputs];
                l.VarianceDelta = new float[outputs];

                l.RollingMean = new float[outputs];
                l.RollingVariance = new float[outputs];

                l.X = new float[batch * outputs];
                l.XNorm = new float[batch * outputs];
            }

            l.ForwardGpu = forward_connected_layer_gpu;
            l.BackwardGpu = backward_connected_layer_gpu;
            l.UpdateGpu = update_connected_layer_gpu;

            l.WeightsGpu = (float[])l.Weights.Clone();
            l.BiasesGpu = (float[])l.Biases.Clone();

            l.WeightUpdatesGpu = (float[])l.WeightUpdates.Clone();
            l.BiasUpdatesGpu = (float[])l.BiasUpdates.Clone();

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            if (batch_normalize)
            {
                l.ScalesGpu = (float[])l.Scales.Clone();
                l.ScaleUpdatesGpu = (float[])l.ScaleUpdates.Clone();

                l.MeanGpu = (float[])l.Mean.Clone();
                l.VarianceGpu = (float[])l.Variance.Clone();

                l.RollingMeanGpu = (float[])l.Mean.Clone();
                l.RollingVarianceGpu = (float[])l.Variance.Clone();

                l.MeanDeltaGpu = (float[])l.Mean.Clone();
                l.VarianceDeltaGpu = (float[])l.Variance.Clone();

                l.XGpu = (float[])l.Output.Clone();
                l.XNormGpu = (float[])l.Output.Clone();
            }
            l.Activation = activation;
            Console.Error.Write($"connected                            %4d  .  %4d\n", inputs, outputs);
            return l;
        }
        
        public void update_connected_layer(int batch, float learning_rate, float momentum, float decay)
        {
            Blas.Axpy_cpu(Outputs, learning_rate / batch, BiasUpdates, Biases);
            Blas.Scal_cpu(Outputs, momentum, BiasUpdates, 1);

            if (BatchNormalize)
            {
                Blas.Axpy_cpu(Outputs, learning_rate / batch, ScaleUpdates, Scales);
                Blas.Scal_cpu(Outputs, momentum, ScaleUpdates, 1);
            }

            Blas.Axpy_cpu(Inputs * Outputs, -decay * batch, Weights, WeightUpdates);
            Blas.Axpy_cpu(Inputs * Outputs, learning_rate / batch, WeightUpdates, Weights);
            Blas.Scal_cpu(Inputs * Outputs, momentum, WeightUpdates, 1);
        }
        
        public void forward_connected_layer(NetworkState state)
        {
            int i;
            Blas.Fill_cpu(Outputs * Batch, 0, Output, 1);
            int m = Batch;
            int k = Inputs;
            int n = Outputs;
            float[] a = state.Input;
            float[] b = Weights;
            float[] c = Output;
            Gemm.gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
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

                    Blas.Copy_cpu(Outputs * Batch, Output, 1, X, 1);
                    Blas.Normalize_cpu(Output, Mean, Variance, Batch, Outputs, 1);
                    Blas.Copy_cpu(Outputs * Batch, Output, 1, XNorm, 1);
                }
                else
                {
                    Blas.Normalize_cpu(Output, RollingMean, RollingVariance, Batch, Outputs, 1);
                }
                scale_bias(Output, Scales, Batch, Outputs, 1);
            }
            for (i = 0; i < Batch; ++i)
            {
                var a = Output.Skip(i * Outputs).ToArray();
                Blas.Axpy_cpu(Outputs, 1, Biases, Output + i * Outputs);
            }
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }
        
        public void backward_connected_layer(NetworkState state)
        {
            int i;
            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);
            for (i = 0; i < Batch; ++i)
            {
                Blas.Axpy_cpu(Outputs, 1, Delta + i * Outputs, 1, BiasUpdates, 1);
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
            float[] a = Delta;
            float[] b = state.Input;
            float[] c = WeightUpdates;
            Gemm.gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

            m = Batch;
            k = Outputs;
            n = Inputs;

            a = Delta;
            b = Weights;
            c = state.Delta;

            if (c) Gemm.gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
        
        public void denormalize_connected_layer()
        {
            int i, j;
            for (i = 0; i < Outputs; ++i)
            {
                float scale = Scales[i] / (float)Math.Sqrt(RollingVariance[i] + .000001);
                for (j = 0; j < Inputs; ++j)
                {
                    Weights[i * Inputs + j] *= scale;
                }
                Biases[i] -= RollingMean[i] * scale;
                Scales[i] = 1;
                RollingMean[i] = 0;
                RollingVariance[i] = 1;
            }
        }
        
        public void statistics_connected_layer()
        {
            if (BatchNormalize)
            {
                Console.Write($"Scales ");
                Utils.print_statistics(Scales, Outputs);
            }
            Console.Write($"Biases ");
            Utils.print_statistics(Biases, Outputs);
            Console.Write($"Weights ");
            Utils.print_statistics(Weights, Outputs);
        }
        
        public void pull_connected_layer()
        {
            Array.Copy(WeightsGpu, Weights, Inputs * Outputs);
            Array.Copy(BiasesGpu, Biases, Outputs);
            Array.Copy(WeightUpdatesGpu, WeightUpdates, Inputs * Outputs);
            Array.Copy(BiasUpdatesGpu, BiasUpdates, Outputs);
            if (BatchNormalize)
            {
                Array.Copy(ScalesGpu, Scales, Outputs);
                Array.Copy(RollingMeanGpu, RollingMean, Outputs);
                Array.Copy(RollingVarianceGpu, RollingVariance, Outputs);
            }
        }
        
        public void push_connected_layer()
        {
            Array.Copy(Weights, WeightsGpu, Inputs * Outputs);
            Array.Copy(Biases, BiasesGpu, Outputs);
            Array.Copy(WeightUpdates, WeightUpdatesGpu, Inputs * Outputs);
            Array.Copy(BiasUpdates, BiasUpdatesGpu, Outputs);
            if (BatchNormalize)
            {
                Array.Copy(Scales, ScalesGpu, Outputs);
                Array.Copy(RollingMean, RollingMeanGpu, Outputs);
                Array.Copy(RollingVariance, RollingVarianceGpu, Outputs);
            }
        }
        
        public void update_connected_layer_gpu(int batch, float learning_rate, float momentum, float decay)
        {
            Blas.axpy_ongpu(Outputs, learning_rate / batch, BiasUpdatesGpu, 1, BiasesGpu, 1);
            Blas.scal_ongpu(Outputs, momentum, BiasUpdatesGpu, 1);

            if (BatchNormalize)
            {
                Blas.axpy_ongpu(Outputs, learning_rate / batch, ScaleUpdatesGpu, 1, ScalesGpu, 1);
                Blas.scal_ongpu(Outputs, momentum, ScaleUpdatesGpu, 1);
            }

            Blas.axpy_ongpu(Inputs * Outputs, -decay * batch, WeightsGpu, 1, WeightUpdatesGpu, 1);
            Blas.axpy_ongpu(Inputs * Outputs, learning_rate / batch, WeightUpdatesGpu, 1, WeightsGpu, 1);
            Blas.scal_ongpu(Inputs * Outputs, momentum, WeightUpdatesGpu, 1);
        }
        
        public void forward_connected_layer_gpu(NetworkState state)
        {
            int i;
            Blas.fill_ongpu(Outputs * Batch, 0, OutputGpu, 1);

            int m = Batch;
            int k = Inputs;
            int n = Outputs;
            float[] a = state.Input;
            float[] b = WeightsGpu;
            float[] c = OutputGpu;
            Gemm.gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
            if (BatchNormalize)
            {
                forward_batchnorm_layer_gpu(state);
            }
            for (i = 0; i < Batch; ++i)
            {
                Blas.axpy_ongpu(Outputs, 1, BiasesGpu, 1, OutputGpu + i * Outputs, 1);
            }
            ActivationsHelper.activate_array_ongpu(OutputGpu, Outputs * Batch, Activation);
        }
        
        public void backward_connected_layer_gpu(NetworkState state)
        {
            int i;
            Blas.constrain_ongpu(Outputs * Batch, 1, DeltaGpu, 1);
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, DeltaGpu);
            for (i = 0; i < Batch; ++i)
            {
                Blas.axpy_ongpu(Outputs, 1, DeltaGpu + i * Outputs, 1, BiasUpdatesGpu, 1);
            }

            if (BatchNormalize)
            {
                backward_batchnorm_layer_gpu(state);
            }

            int m = Outputs;
            int k = Batch;
            int n = Inputs;
            float[] a = DeltaGpu;
            float[] b = state.Input;
            float[] c = WeightUpdatesGpu;
            Gemm.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

            m = Batch;
            k = Outputs;
            n = Inputs;

            a = DeltaGpu;
            b = WeightsGpu;
            c = state.Delta;

            if (c.Any()) Gemm.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
        
        public static CostType get_cost_type(string s)
        {
            if (s == "sse") return CostType.Sse;
            if (s == "masked") return CostType.Masked;
            if (s == "smooth") return CostType.Smooth;
            Console.Error.Write($"Couldn't find cost type %s, going with CostType.Sse\n", s);
            return CostType.Sse;
        }
        
        public static string get_cost_string(CostType a)
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
        
        public static Layer make_cost_layer(int batch, int inputs, CostType cost_type, float scale)
        {
            Console.Error.Write($"cost                                           %4d\n", inputs);
            Layer l = new Layer();
            l.LayerType = LayerType.Cost;

            l.Scale = scale;
            l.Batch = batch;
            l.Inputs = inputs;
            l.Outputs = inputs;
            l.CostType = cost_type;
            l.Delta = new float[inputs * batch];
            l.Output = new float[inputs * batch];
            l.Cost = 0;

            l.Forward = forward_cost_layer;
            l.Backward = backward_cost_layer;

            l.ForwardGpu = forward_cost_layer_gpu;
            l.BackwardGpu = backward_cost_layer_gpu;

            l.DeltaGpu = (float[])l.Output.Clone();
            l.OutputGpu = (float[])l.Delta.Clone();

            return l;
        }
        
        public  void resize_cost_layer(int inputs)
        {
            Inputs = inputs;
            Outputs = inputs;
            Array.Resize(ref Delta, inputs * Batch);
            Array.Resize(ref Output, inputs * Batch);

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

        }
        
        public void forward_cost_layer(NetworkState state)
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
            Cost[0] = Output.Sum();
        }
        
        public void backward_cost_layer(NetworkState state)
        {
            Blas.Axpy_cpu(Batch * Inputs, Scale, Delta, state.Delta);
        }
        
        public static void pull_cost_layer()
        {
            Array.Copy(DeltaGpu, Delta, Batch * Inputs);
        }
        
        public static void push_cost_layer()
        {
            Array.Copy(Delta, DeltaGpu, Batch * Inputs);
        }
        
        public void forward_cost_layer_gpu(NetworkState state)
        {
            if (!state.Truth.Any()) return;
            if (CostType == CostType.Masked)
            {
                Blas.mask_ongpu(Batch * Inputs, state.Input, Utils.SecretNum, state.Truth);
            }

            if (CostType == CostType.Smooth)
            {
                Blas.smooth_l1_gpu(Batch * Inputs, state.Input, state.Truth, DeltaGpu, OutputGpu);
            }
            else
            {
                Blas.l2_gpu(Batch * Inputs, state.Input, state.Truth, DeltaGpu, OutputGpu);
            }

            if (Ratio != 0)
            {
                Array.Copy(DeltaGpu, Delta, Batch * Inputs);
                Delta = Delta.OrderBy(f => Math.Abs(f)).ToArray();
                int n = (int)((1 - Ratio) * Batch * Inputs);
                float thresh = Delta[n];
                thresh = 0;
                Console.Write($"%f\n", thresh);
                Blas.supp_ongpu(Batch * Inputs, thresh, DeltaGpu, 1);
            }

            Array.Copy(OutputGpu, Output, Batch * Inputs);
            Cost[0] = Output.Sum();
        }
        
        public void backward_cost_layer_gpu(NetworkState state)
        {
            Blas.axpy_ongpu(Batch * Inputs, Scale, DeltaGpu, 1, state.Delta, 1);
        }
        
        public static void increment_layer(int steps)
        {
            int num = Outputs * Batch * steps;
            Output += num;
            Delta += num;
            X += num;
            XNorm += num;


            OutputGpu += num;
            DeltaGpu += num;
            XGpu += num;
            XNormGpu += num;

        }
        
        public static Layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, Activation activation, int batch_normalize)
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

            l.State = new float[l.Hidden * batch * (steps + 1)];

            l.InputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputLayer) = make_convolutional_layer(batch * steps, h, w, c, hidden_filters, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
            l.InputLayer.Batch = batch;

            l.InputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputLayer) = make_convolutional_layer(batch * steps, h, w, hidden_filters, hidden_filters, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
            l.InputLayer.Batch = batch;

            l.OutputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.OutputLayer) = make_convolutional_layer(batch * steps, h, w, hidden_filters, output_filters, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
            l.OutputLayer.Batch = batch;

            l.Output = l.OutputLayer.Output;
            l.Delta = l.OutputLayer.Delta;

            l.Forward = forward_crnn_layer;
            l.Backward = backward_crnn_layer;
            l.Update = update_crnn_layer;


            l.ForwardGpu = forward_crnn_layer_gpu;
            l.BackwardGpu = backward_crnn_layer_gpu;
            l.UpdateGpu = update_crnn_layer_gpu;

            l.StateGpu = (float[])l.State.Clone();
            l.OutputGpu = l.OutputLayer.OutputGpu;
            l.DeltaGpu = l.OutputLayer.DeltaGpu;


            return l;
        }
        
        public void update_crnn_layer(int batch, float learning_rate, float momentum, float decay)
        {
            InputLayer.update_convolutional_layer(batch, learning_rate, momentum, decay);
            InputLayer.update_convolutional_layer(batch, learning_rate, momentum, decay);
            OutputLayer.update_convolutional_layer(batch, learning_rate, momentum, decay);
        }
        
        public void forward_crnn_layer(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (InputLayer);
            Layer self_layer = (InputLayer);
            Layer output_layer = (OutputLayer);

            Blas.Fill_cpu(Outputs * Batch * Steps, 0, output_layer.Delta, 1);
            Blas.Fill_cpu(Hidden * Batch * Steps, 0, self_layer.Delta, 1);
            Blas.Fill_cpu(Hidden * Batch * Steps, 0, input_layer.Delta, 1);
            if (state.Train) Blas.Fill_cpu(Hidden * Batch, 0, State, 1);

            for (i = 0; i < Steps; ++i)
            {
                s.Input = state.Input;
                input_layer.forward_convolutional_layer(s);

                s.Input = State;
                self_layer.forward_convolutional_layer(s);

                float[] old_state = State;
                if (state.Train) State += Hidden * Batch;
                if (shortcut)
                {
                    Blas.Copy_cpu(Hidden * Batch, old_state, 1, State, 1);
                }
                else
                {
                    Blas.Fill_cpu(Hidden * Batch, 0, State, 1);
                }
                Blas.Axpy_cpu(Hidden * Batch, 1, input_layer.Output, State);
                Blas.Axpy_cpu(Hidden * Batch, 1, self_layer.Output, State);

                s.Input = State;
                output_layer.forward_convolutional_layer(s);

                state.Input += Inputs * Batch;
                input_layer.increment_layer(1);
                self_layer.increment_layer(1);
                output_layer.increment_layer(1);
            }
        }
        
        public void backward_crnn_layer(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (InputLayer);
            Layer self_layer = (InputLayer);
            Layer output_layer = OutputLayer;

            input_layer.increment_layer(Steps - 1);
            self_layer.increment_layer(Steps - 1);
            output_layer.increment_layer(Steps - 1);

            State += Hidden * Batch * Steps;
            for (i = Steps - 1; i >= 0; --i)
            {
                Blas.Copy_cpu(Hidden * Batch, input_layer.Output, 1, State, 1);
                Blas.Axpy_cpu(Hidden * Batch, 1, self_layer.Output, State);

                s.Input = State;
                s.Delta = self_layer.Delta;
                output_layer.backward_convolutional_layer(s);

                State -= Hidden * Batch;

                s.Input = State;
                s.Delta = self_layer.Delta - Hidden * Batch;
                if (i == 0) s.Delta = 0;
                self_layer.backward_convolutional_layer(s);

                Blas.Copy_cpu(Hidden * Batch, self_layer.Delta, 1, input_layer.Delta, 1);
                if (i > 0 && shortcut) Blas.Axpy_cpu(Hidden * Batch, 1, self_layer.Delta, self_layer.Delta - Hidden * Batch);
                s.Input = state.Input + i * Inputs * Batch;
                if (state.Delta) s.Delta = state.Delta + i * Inputs * Batch;
                else s.Delta = 0;
                input_layer.backward_convolutional_layer(s);

                input_layer.increment_layer(-1);
                self_layer.increment_layer(-1);
                output_layer.increment_layer(-1);
            }
        }
        
        public void pull_crnn_layer()
        {
            InputLayer.pull_convolutional_layer();
            InputLayer.pull_convolutional_layer();
            OutputLayer.pull_convolutional_layer();
        }
        
        public void push_crnn_layer()
        {
            InputLayer.push_convolutional_layer();
            InputLayer.push_convolutional_layer();
            OutputLayer.push_convolutional_layer();
        }
        
        public void update_crnn_layer_gpu(int batch, float learning_rate, float momentum, float decay)
        {
            update_convolutional_layer_gpu((InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer_gpu((InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer_gpu((OutputLayer), batch, learning_rate, momentum, decay);
        }
        
        public void forward_crnn_layer_gpu(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (InputLayer);
            Layer self_layer = (InputLayer);
            Layer output_layer = (OutputLayer);

            Blas.fill_ongpu(Outputs * Batch * Steps, 0, output_layer.DeltaGpu, 1);
            Blas.fill_ongpu(Hidden * Batch * Steps, 0, self_layer.DeltaGpu, 1);
            Blas.fill_ongpu(Hidden * Batch * Steps, 0, input_layer.DeltaGpu, 1);
            if (state.Train) Blas.fill_ongpu(Hidden * Batch, 0, StateGpu, 1);

            for (i = 0; i < Steps; ++i)
            {
                s.Input = state.Input;
                forward_convolutional_layer_gpu(input_layer, s);

                s.Input = StateGpu;
                forward_convolutional_layer_gpu(self_layer, s);

                float[] old_state = StateGpu;
                if (state.Train) StateGpu += Hidden * Batch;
                if (shortcut)
                {
                    Blas.copy_ongpu(Hidden * Batch, old_state, 1, StateGpu, 1);
                }
                else
                {
                    Blas.fill_ongpu(Hidden * Batch, 0, StateGpu, 1);
                }
                Blas.axpy_ongpu(Hidden * Batch, 1, input_layer.OutputGpu, 1, StateGpu, 1);
                Blas.axpy_ongpu(Hidden * Batch, 1, self_layer.OutputGpu, 1, StateGpu, 1);

                s.Input = StateGpu;
                forward_convolutional_layer_gpu(output_layer, s);

                state.Input += Inputs * Batch;
                input_layer.increment_layer(1);
                self_layer.increment_layer(1);
                output_layer.increment_layer(1);
            }
        }
        
        public void backward_crnn_layer_gpu(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (InputLayer);
            Layer self_layer = (InputLayer);
            Layer output_layer = (OutputLayer);
            input_layer.increment_layer(Steps - 1);
            self_layer.increment_layer(Steps - 1);
            output_layer.increment_layer(Steps - 1);
            StateGpu += Hidden * Batch * Steps;
            for (i = Steps - 1; i >= 0; --i)
            {
                Blas.copy_ongpu(Hidden * Batch, input_layer.OutputGpu, 1, StateGpu, 1);
                Blas.axpy_ongpu(Hidden * Batch, 1, self_layer.OutputGpu, 1, StateGpu, 1);

                s.Input = StateGpu;
                s.Delta = self_layer.DeltaGpu;
                output_layer.backward_convolutional_layer_gpu(s);

                StateGpu -= Hidden * Batch;

                s.Input = StateGpu;
                s.Delta = self_layer.DeltaGpu - Hidden * Batch;
                if (i == 0) s.Delta = 0;
                backward_convolutional_layer_gpu(self_layer, s);

                Blas.copy_ongpu(Hidden * Batch, self_layer.DeltaGpu, 1, input_layer.DeltaGpu, 1);
                if (i > 0 && shortcut) Blas.axpy_ongpu(Hidden * Batch, 1, self_layer.DeltaGpu, 1, self_layer.DeltaGpu - Hidden * Batch, 1);
                s.Input = state.Input + i * Inputs * Batch;
                if (state.Delta.Any()) s.Delta = state.Delta + i * Inputs * Batch;
                else s.Delta = new float[0];
                backward_convolutional_layer_gpu(input_layer, s);

                input_layer.increment_layer(-1);
                self_layer.increment_layer(-1);
                output_layer.increment_layer(-1);
            }
        }
        
        public Image get_crop_image()
        {
            int h = OutH;
            int w = OutW;
            int c = OutC;
            return new Image(w, h, c, Output);
        }
        
        public void backward_crop_layer(NetworkState state) { }
        
        public  void backward_crop_layer_gpu(NetworkState state) { }
        
        public static Layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, bool flip, float angle, float saturation, float exposure)
        {
            Console.Error.Write($"Crop Layer: {h} x {w} . {crop_height} x {crop_width} x {c} Image\n");
            Layer l = new Layer();
            l.LayerType = LayerType.Crop;
            l.Batch = batch;
            l.H = h;
            l.W = w;
            l.C = c;
            l.Scale = (float)crop_height / h;
            l.Flip = flip;
            l.Angle = angle;
            l.Saturation = saturation;
            l.Exposure = exposure;
            l.OutW = crop_width;
            l.OutH = crop_height;
            l.OutC = c;
            l.Inputs = l.W * l.H * l.C;
            l.Outputs = l.OutW * l.OutH * l.OutC;
            l.Output = new float[l.Outputs * batch];
            l.Forward = forward_crop_layer;
            l.Backward = backward_crop_layer;


            l.ForwardGpu = forward_crop_layer_gpu;
            l.BackwardGpu = backward_crop_layer_gpu;
            l.OutputGpu = (float[])l.Output.Clone();
            l.RandGpu = new float[l.Batch * 8];

            return l;
        }
        
        public void resize_crop_layer(int w, int h)
        {
            W = w;
            H = h;

            OutW = (int)(Scale * w);
            OutH = (int)(Scale * h);

            Inputs = W * H * C;
            Outputs = OutH * OutW * OutC;

            Array.Resize(ref Output, Batch * Outputs);

            OutputGpu = (float[])Output.Clone();

        }
        
        public void forward_crop_layer(NetworkState state)
        {
            int i, j, c, b, row, col;
            int index;
            int count = 0;
            bool flip = (Flip && Utils.Rand.Next() % 2 != 0);
            int dh = Utils.Rand.Next() % (H - OutH + 1);
            int dw = Utils.Rand.Next() % (W - OutW + 1);
            float scale = 2;
            float trans = -1;
            if (Noadjust)
            {
                scale = 1;
                trans = 0;
            }
            if (!state.Train)
            {
                flip = false;
                dh = (H - OutH) / 2;
                dw = (W - OutW) / 2;
            }
            for (b = 0; b < Batch; ++b)
            {
                for (c = 0; c < C; ++c)
                {
                    for (i = 0; i < OutH; ++i)
                    {
                        for (j = 0; j < OutW; ++j)
                        {
                            if (flip)
                            {
                                col = W - dw - j - 1;
                            }
                            else
                            {
                                col = j + dw;
                            }
                            row = i + dh;
                            index = col + W * (row + H * (c + C * b));
                            Output[count++] = state.Input[index] * scale + trans;
                        }
                    }
                }
            }
        }
        
        public int deconvolutional_out_height()
        {
            int h = Stride * (H - 1) + Size;
            return h;
        }
        
        public int deconvolutional_out_width()
        {
            int w = Stride * (W - 1) + Size;
            return w;
        }
        
        public int deconvolutional_out_size()
        {
            return deconvolutional_out_height(l) * deconvolutional_out_width(l);
        }
        
        public Image get_deconvolutional_image()
        {
            int h, w, c;
            h = deconvolutional_out_height(l);
            w = deconvolutional_out_width(l);
            c = N;
            return new Image(w, h, c, Output);
        }
        
        public Image get_deconvolutional_delta()
        {
            int h, w, c;
            h = deconvolutional_out_height(l);
            w = deconvolutional_out_width(l);
            c = N;
            return new Image(w, h, c, Delta);
        }
        
        public static Layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, Activation activation)
        {
            int i;
            Layer l = new Layer();
            l.LayerType = LayerType.Deconvolutional;

            l.H = h;
            l.W = w;
            l.C = c;
            l.N = n;
            l.Batch = batch;
            l.Stride = stride;
            l.Size = size;

            l.Weights = new float[c * n * size * size];
            l.WeightUpdates = new float[c * n * size * size];

            l.Biases = new float[n];
            l.BiasUpdates = new float[n];
            float scale = 1.0f / (float)Math.Sqrt(size * size * c);
            for (i = 0; i < c * n * size * size; ++i) l.Weights[i] = scale * Utils.rand_normal();
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

            l.ColImage = new float[h * w * size * size * n];
            l.Output = new float[l.Batch * out_h * out_w * n];
            l.Delta = new float[l.Batch * out_h * out_w * n];

            l.Forward = forward_deconvolutional_layer;
            l.Backward = backward_deconvolutional_layer;
            l.Update = update_deconvolutional_layer;


            l.WeightsGpu = (float[])l.Weights.Clone();
            l.WeightUpdatesGpu = (float[])l.WeightUpdates.Clone();

            l.BiasesGpu = (float[])l.Biases.Clone();
            l.BiasUpdatesGpu = (float[])l.BiasUpdates.Clone();

            l.ColImageGpu = (float[])l.ColImage.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            l.OutputGpu = (float[])l.Output.Clone();


            l.Activation = activation;

            Console.Error.Write($"Deconvolutional Layer: %d x %d x %d Image, %d filters . %d x %d x %d Image\n", h, w, c, n, out_h, out_w, n);

            return l;
        }
        
        public void resize_deconvolutional_layer(int h, int w)
        {
            H = h;
            W = w;
            int out_h = deconvolutional_out_height(this);
            int out_w = deconvolutional_out_width(this);

            Array.Resize(ref ColImage, out_h * out_w * Size * Size * C);
            Array.Resize(ref Output, Batch * out_h * out_w * N);
            Array.Resize(ref Delta, Batch * out_h * out_w * N);

            ColImageGpu = (float[])ColImage.Clone();
            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

        }
        
        public void forward_deconvolutional_layer(NetworkState state)
        {
            int i;
            int out_h = deconvolutional_out_height(this);
            int out_w = deconvolutional_out_width(this);
            int size = out_h * out_w;

            int m = Size * Size * N;
            int n = H * W;
            int k = C;

            Blas.Fill_cpu(Outputs * Batch, 0, Output, 1);

            for (i = 0; i < Batch; ++i)
            {
                float[] a = Weights;
                float[] b = state.Input + i * C * H * W;
                float[] c = ColImage;

                Gemm.gemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

                Im2Col.col2im_cpu(c, N, out_h, out_w, Size, Stride, 0, Output + i * N * size);
            }
            add_bias(Output, Biases, Batch, N, size);
            ActivationsHelper.Activate_array(Output, Batch * N * size, Activation);
        }
        
        public void backward_deconvolutional_layer(NetworkState state)
        {
            float alpha = 1.0f / Batch;
            int out_h = deconvolutional_out_height(this);
            int out_w = deconvolutional_out_width(this);
            int size = out_h * out_w;
            int i;

            ActivationsHelper.Gradient_array(Output, size * N * Batch, Activation, Delta);
            backward_bias(BiasUpdates, Delta, Batch, N, size);

            for (i = 0; i < Batch; ++i)
            {
                int m = C;
                int n = Size * Size * N;
                int k = H * W;

                float[] a = state.Input + i * m * n;
                float[] b = ColImage;
                float[] c = WeightUpdates;

                Im2Col.im2col_cpu(Delta + i * N * size, N, out_h, out_w,
                        Size, Stride, 0, b);
                Gemm.gemm(0, 1, m, n, k, alpha, a, k, b, k, 1, c, n);

                if (state.Delta)
                {
                    int m2 = C;
                    int n2 = H * W;
                    int k2 = Size * Size * N;

                    float[] a2 = Weights;
                    float[] b2 = ColImage;
                    float[] c2 = state.Delta + i * n2 * m2;

                    Gemm.gemm(0, 0, m2, n2, k2, 1, a2, k2, b2, n2, 1, c2, n2);
                }
            }
        }
        
        public void update_deconvolutional_layer(int n, float learning_rate, float momentum, float decay)
        {
            int size = Size * Size * C * N;
            Blas.Axpy_cpu(N, learning_rate, BiasUpdates, Biases);
            Blas.Scal_cpu(N, momentum, BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay, Weights, WeightUpdates);
            Blas.Axpy_cpu(size, learning_rate, WeightUpdates, Weights);
            Blas.Scal_cpu(size, momentum, WeightUpdates, 1);
        }
        
        public static Layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
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
            l.Cost = 0;
            l.Outputs = l.Inputs;
            l.Truths = l.Side * l.Side * (1 + l.Coords + l.Classes);
            l.Output = new float[batch * l.Outputs];
            l.Delta = new float[batch * l.Outputs];

            l.Forward = forward_detection_layer;
            l.Backward = backward_detection_layer;

            l.ForwardGpu = forward_detection_layer_gpu;
            l.BackwardGpu = backward_detection_layer_gpu;
            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();

            Console.Error.Write($"Detection Layer\n");

            return l;
        }
        
        public void forward_detection_layer(NetworkState state)
        {
            int locations = Side * Side;
            int i, j;
            Array.Copy(state.Input, 0, Output, 0, Outputs * Batch);
            //if(reorg) reorg(Output, W*H, size*N, Batch, 1);
            int b;
            if (Softmax != 0)
            {
                for (b = 0; b < Batch; ++b)
                {
                    int index = b * Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int offset = i * Classes;
                        softmax(Output + index + offset, Classes, 1,
                                Output + index + offset);
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
                (Cost) = null;
                int size = Inputs * Batch;
                Delta = new float[size];
                for (b = 0; b < Batch; ++b)
                {
                    int index = b * Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int truth_index = (b * locations + i) * (1 + Coords + Classes);
                        bool is_obj = state.Truth[truth_index] != 0;
                        for (j = 0; j < N; ++j)
                        {
                            int pn_index = index + locations * Classes + i * N + j;
                            Delta[pn_index] = NoobjectScale * (0 - Output[pn_index]);
                            (Cost) += NoobjectScale * (float)Math.Pow(Output[pn_index], 2);
                            avg_anyobj += Output[pn_index];
                        }

                        int best_index = -1;
                        float best_iou = 0;
                        float best_rmse = 20;

                        if (!is_obj)
                        {
                            continue;
                        }

                        int class_index = index + i * Classes;
                        for (j = 0; j < Classes; ++j)
                        {
                            Delta[class_index + j] = ClassScale * (state.Truth[truth_index + 1 + j] - Output[class_index + j]);
                            (Cost) += ClassScale * (float)Math.Pow(state.Truth[truth_index + 1 + j] - Output[class_index + j], 2);
                            if (state.Truth[truth_index + 1 + j]) avg_cat += Output[class_index + j];
                            avg_allcat += Output[class_index + j];
                        }

                        Box truth = float_to_box(state.Truth + truth_index + 1 + Classes);
                        truth.X /= Side;
                        truth.Y /= Side;

                        for (j = 0; j < N; ++j)
                        {
                            int box_index = index + locations * (Classes + N) + (i * N + j) * Coords;
                            Box outputout = float_to_box(Output + box_index);
                            outputout.X /= Side;
                            outputout.Y /= Side;

                            if (Sqrt)
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

                        if (Forced != 0)
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
                        if (Random != 0 && state.Net.Seen < 64000)
                        {
                            best_index = Utils.Rand.Next() % N;
                        }

                        int box_index = index + locations * (Classes + N) + (i * N + best_index) * Coords;
                        int tbox_index = truth_index + 1 + Classes;

                        Box outputout = float_to_box(Output + box_index);
                        outputout.X /= Side;
                        outputout.Y /= Side;
                        if (Sqrt)
                        {
                            outputout.W = outputout.W * outputout.W;
                            outputout.H = outputout.H * outputout.H;
                        }
                        float iou = box_iou(outputout, truth);

                        //Console.Write($"%d,", best_index);
                        int p_index = index + locations * Classes + i * N + best_index;
                        (Cost) -= NoobjectScale * (float)Math.Pow(Output[p_index], 2);
                        (Cost) += ObjectScale * (float)Math.Pow(1 - Output[p_index], 2);
                        avg_obj += Output[p_index];
                        Delta[p_index] = ObjectScale * (1.0f - Output[p_index]);

                        if (Rescore)
                        {
                            Delta[p_index] = ObjectScale * (iou - Output[p_index]);
                        }

                        Delta[box_index + 0] = CoordScale * (state.Truth[tbox_index + 0] - Output[box_index + 0]);
                        Delta[box_index + 1] = CoordScale * (state.Truth[tbox_index + 1] - Output[box_index + 1]);
                        Delta[box_index + 2] = CoordScale * (state.Truth[tbox_index + 2] - Output[box_index + 2]);
                        Delta[box_index + 3] = CoordScale * (state.Truth[tbox_index + 3] - Output[box_index + 3]);
                        if (Sqrt)
                        {
                            Delta[box_index + 2] = CoordScale * ((float)Math.Sqrt(state.Truth[tbox_index + 2]) - Output[box_index + 2]);
                            Delta[box_index + 3] = CoordScale * ((float)Math.Sqrt(state.Truth[tbox_index + 3]) - Output[box_index + 3]);
                        }

                        (Cost) += (float)Math.Pow(1 - iou, 2);
                        avg_iou += iou;
                        ++count;
                    }
                }
                (Cost) = (float)Math.Pow(Utils.mag_array(Delta, Outputs * Batch), 2);


                Console.Write($"Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou / count, avg_cat / count, avg_allcat / (count * Classes), avg_obj / count, avg_anyobj / (Batch * locations * N), count);
                //if(reorg) reorg(Delta, W*H, size*N, Batch, 0);
            }
        }
        
        public void backward_detection_layer(NetworkState state)
        {
            Blas.Axpy_cpu(Batch * Inputs, 1, Delta, state.Delta);
        }
        
        public static void get_detection_boxes(int w, int h, float thresh, float[][] probs, Box[] boxes, bool only_objectness)
        {
            int i, j, n;
            float[] predictions = Output;
            //int per_cell = 5*num+classes;
            for (i = 0; i < Side * Side; ++i)
            {
                int row = i / Side;
                int col = i % Side;
                for (n = 0; n < N; ++n)
                {
                    int index = i * N + n;
                    int p_index = Side * Side * Classes + i * N + n;
                    float scale = predictions[p_index];
                    int box_index = Side * Side * (Classes + N) + (i * N + n) * 4;
                    boxes[index].X = (predictions[box_index + 0] + col) / Side * w;
                    boxes[index].Y = (predictions[box_index + 1] + row) / Side * h;
                    boxes[index].W = (float)Math.Pow(predictions[box_index + 2], (Sqrt != 0 ? 2 : 1)) * w;
                    boxes[index].H = (float)Math.Pow(predictions[box_index + 3], (Sqrt != 0 ? 2 : 1)) * h;
                    for (j = 0; j < Classes; ++j)
                    {
                        int class_index = i * Classes;
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
        
        public void forward_detection_layer_gpu(NetworkState state)
        {
            if (!state.Train)
            {
                Blas.copy_ongpu(Batch * Inputs, state.Input, 1, OutputGpu, 1);
                return;
            }

            float[] in_cpu = new float[Batch * Inputs];
            float[] truth_cpu = null;
            if (state.Truth.Any())
            {
                int num_truth = Batch * Side * Side * (1 + Coords + Classes);
                truth_cpu = new float[num_truth];
                Array.Copy(state.Truth, truth_cpu, num_truth);
            }
            Array.Copy(state.Input, in_cpu, Batch * Inputs);
            NetworkState cpu_state = state;
            cpu_state.Train = state.Train;
            cpu_state.Truth = truth_cpu;
            cpu_state.Input = in_cpu;
            forward_detection_layer(cpu_state);
            Array.Copy(Output, OutputGpu, Batch * Outputs);
            Array.Copy(Delta, DeltaGpu, Batch * Inputs);
        }
        
        public void backward_detection_layer_gpu(NetworkState state)
        {
            Blas.axpy_ongpu(Batch * Inputs, 1, DeltaGpu, 1, state.Delta, 1);
        }
        
        public static Layer make_dropout_layer(int batch, int inputs, float probability)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Dropout;
            l.Probability = probability;
            l.Inputs = inputs;
            l.Outputs = inputs;
            l.Batch = batch;
            l.Rand = new float[inputs * batch];
            l.Scale = 1.0f / (1.0f - probability);
            l.Forward = forward_dropout_layer;
            l.Backward = backward_dropout_layer;

            l.ForwardGpu = forward_dropout_layer_gpu;
            l.BackwardGpu = backward_dropout_layer_gpu;
            l.RandGpu = (float[])l.Rand.Clone();

            Console.Error.Write($"dropout       p = %.2f               %4d  .  %4d\n", probability, inputs, inputs);
            return l;
        }
        
        public void resize_dropout_layer(int inputs)
        {
            Array.Resize(ref Rand, Inputs * Batch);

            RandGpu = (float[])Rand.Clone();
        }
        
        public void forward_dropout_layer(NetworkState state)
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
        
        public void backward_dropout_layer(NetworkState state)
        {
            int i;
            if (!state.Delta) return;
            for (i = 0; i < Batch * Inputs; ++i)
            {
                float r = Rand[i];
                if (r < Probability) state.Delta[i] = 0;
                else state.Delta[i] *= Scale;
            }
        }
        
        public static Layer make_gru_layer(int batch, int inputs, int outputs, int steps, bool batch_normalize)
        {
            Console.Error.Write($"GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
            batch = batch / steps;
            Layer l = new Layer();
            l.Batch = batch;
            l.LayerType = LayerType.Gru;
            l.Steps = steps;
            l.Inputs = inputs;

            l.InputZLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputZLayer) = make_connected_layer(batch * steps, inputs, outputs, Activation.Linear, batch_normalize);
            l.InputZLayer.Batch = batch;

            l.StateZLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.StateZLayer) = make_connected_layer(batch * steps, outputs, outputs, Activation.Linear, batch_normalize);
            l.StateZLayer.Batch = batch;



            l.InputRLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputRLayer) = make_connected_layer(batch * steps, inputs, outputs, Activation.Linear, batch_normalize);
            l.InputRLayer.Batch = batch;

            l.StateRLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.StateRLayer) = make_connected_layer(batch * steps, outputs, outputs, Activation.Linear, batch_normalize);
            l.StateRLayer.Batch = batch;



            l.InputHLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputHLayer) = make_connected_layer(batch * steps, inputs, outputs, Activation.Linear, batch_normalize);
            l.InputHLayer.Batch = batch;

            l.StateHLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.StateHLayer) = make_connected_layer(batch * steps, outputs, outputs, Activation.Linear, batch_normalize);
            l.StateHLayer.Batch = batch;

            l.BatchNormalize = batch_normalize;


            l.Outputs = outputs;
            l.Output = new float[outputs * batch * steps];
            l.Delta = new float[outputs * batch * steps];
            l.State = new float[outputs * batch];
            l.PrevState = new float[outputs * batch];
            l.ForgotState = new float[outputs * batch];
            l.ForgotDelta = new float[outputs * batch];

            l.RCpu = new float[outputs * batch];
            l.ZCpu = new float[outputs * batch];
            l.HCpu = new float[outputs * batch];

            l.Forward = forward_gru_layer;
            l.Backward = backward_gru_layer;
            l.Update = update_gru_layer;


            l.ForwardGpu = forward_gru_layer_gpu;
            l.BackwardGpu = backward_gru_layer_gpu;
            l.UpdateGpu = update_gru_layer_gpu;

            l.ForgotStateGpu = (float[])l.Output.Clone();
            l.ForgotDeltaGpu = (float[])l.Output.Clone();
            l.PrevStateGpu = (float[])l.Output.Clone();
            l.StateGpu = (float[])l.Output.Clone();
            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            l.RGpu = (float[])l.OutputGpu.Clone();
            l.ZGpu = (float[])l.OutputGpu.Clone();
            l.HGpu = (float[])l.OutputGpu.Clone();

            return l;
        }
        
        public void update_gru_layer(int batch, float learning_rate, float momentum, float decay)
        {
            InputLayer.update_connected_layer(batch, learning_rate, momentum, decay);
            InputLayer.update_connected_layer(batch, learning_rate, momentum, decay);
            OutputLayer.update_connected_layer(batch, learning_rate, momentum, decay);
        }
        
        public void forward_gru_layer(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_z_layer = (InputZLayer);
            Layer input_r_layer = (InputRLayer);
            Layer input_h_layer = (InputHLayer);

            Layer state_z_layer = (StateZLayer);
            Layer state_r_layer = (StateRLayer);
            Layer state_h_layer = (StateHLayer);

            Blas.Fill_cpu(Outputs * Batch * Steps, 0, input_z_layer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, input_r_layer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, input_h_layer.Delta, 1);

            Blas.Fill_cpu(Outputs * Batch * Steps, 0, state_z_layer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, state_r_layer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, state_h_layer.Delta, 1);
            if (state.Train)
            {
                Blas.Fill_cpu(Outputs * Batch * Steps, 0, Delta, 1);
                Blas.Copy_cpu(Outputs * Batch, State, 1, PrevState, 1);
            }

            for (i = 0; i < Steps; ++i)
            {
                s.Input = State;
                state_z_layer.forward_connected_layer(s);
                state_r_layer.forward_connected_layer(s);

                s.Input = state.Input;
                input_z_layer.forward_connected_layer(s);
                input_r_layer.forward_connected_layer(s);
                input_h_layer.forward_connected_layer(s);


                Blas.Copy_cpu(Outputs * Batch, input_z_layer.Output, 1, ZCpu, 1);
                Blas.Axpy_cpu(Outputs * Batch, 1, state_z_layer.Output, 1, ZCpu, 1);

                Blas.Copy_cpu(Outputs * Batch, input_r_layer.Output, 1, RCpu, 1);
                Blas.Axpy_cpu(Outputs * Batch, 1, state_r_layer.Output, 1, RCpu, 1);

                ActivationsHelper.Activate_array(ZCpu, Outputs * Batch, Activation.Logistic);
                ActivationsHelper.Activate_array(RCpu, Outputs * Batch, Activation.Logistic);

                Blas.Copy_cpu(Outputs * Batch, State, 1, ForgotState, 1);
                Blas.Mul_cpu(Outputs * Batch, RCpu, 1, ForgotState, 1);

                s.Input = ForgotState;
                state_h_layer.forward_connected_layer(s);

                Blas.Copy_cpu(Outputs * Batch, input_h_layer.Output, 1, HCpu, 1);
                Blas.Axpy_cpu(Outputs * Batch, 1, state_h_layer.Output, 1, HCpu, 1);

                // USET ActivationsHelper.Activate_array(HCpu, Outputs * Batch, TANH);
                ActivationsHelper.Activate_array(HCpu, Outputs * Batch, Activation.Logistic);


                Blas.Weighted_sum_cpu(State, HCpu, ZCpu, Outputs * Batch, Output);

                Blas.Copy_cpu(Outputs * Batch, Output, 1, State, 1);

                state.Input += Inputs * Batch;
                Output += Outputs * Batch;
                input_z_layer.increment_layer(1);
                input_r_layer.increment_layer(1);
                input_h_layer.increment_layer(1);

                state_z_layer.increment_layer(1);
                state_r_layer.increment_layer(1);
                state_h_layer.increment_layer(1);
            }
        }
        
        public void backward_gru_layer(NetworkState state)
        {
        }
        
        public static void pull_gru_layer()
        {
        }
        
        public static void push_gru_layer()
        {
        }
        
        public void update_gru_layer_gpu(int batch, float learning_rate, float momentum, float decay)
        {
            InputRLayer.update_connected_layer_gpu(batch, learning_rate, momentum, decay);
            InputZLayer.update_connected_layer_gpu(batch, learning_rate, momentum, decay);
            InputHLayer.update_connected_layer_gpu(batch, learning_rate, momentum, decay);
            StateRLayer.update_connected_layer_gpu(batch, learning_rate, momentum, decay);
            StateZLayer.update_connected_layer_gpu(batch, learning_rate, momentum, decay);
            StateHLayer.update_connected_layer_gpu(batch, learning_rate, momentum, decay);
        }
        
        public void forward_gru_layer_gpu(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_z_layer = (InputZLayer);
            Layer input_r_layer = (InputRLayer);
            Layer input_h_layer = (InputHLayer);

            Layer state_z_layer = (StateZLayer);
            Layer state_r_layer = (StateRLayer);
            Layer state_h_layer = (StateHLayer);

            Blas.fill_ongpu(Outputs * Batch * Steps, 0, input_z_layer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, input_r_layer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, input_h_layer.DeltaGpu, 1);

            Blas.fill_ongpu(Outputs * Batch * Steps, 0, state_z_layer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, state_r_layer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, state_h_layer.DeltaGpu, 1);
            if (state.Train)
            {
                Blas.fill_ongpu(Outputs * Batch * Steps, 0, DeltaGpu, 1);
                Blas.copy_ongpu(Outputs * Batch, StateGpu, 1, PrevStateGpu, 1);
            }

            for (i = 0; i < Steps; ++i)
            {
                s.Input = StateGpu;
                state_z_layer.forward_connected_layer_gpu(s);
                state_r_layer.forward_connected_layer_gpu(s);

                s.Input = state.Input;
                input_z_layer.forward_connected_layer_gpu(s);
                input_r_layer.forward_connected_layer_gpu(s);
                input_h_layer.forward_connected_layer_gpu(s);


                Blas.copy_ongpu(Outputs * Batch, input_z_layer.OutputGpu, 1, ZGpu, 1);
                Blas.axpy_ongpu(Outputs * Batch, 1, state_z_layer.OutputGpu, 1, ZGpu, 1);

                Blas.copy_ongpu(Outputs * Batch, input_r_layer.OutputGpu, 1, RGpu, 1);
                Blas.axpy_ongpu(Outputs * Batch, 1, state_r_layer.OutputGpu, 1, RGpu, 1);

                ActivationsHelper.activate_array_ongpu(ZGpu, Outputs * Batch, Activation.Logistic);
                ActivationsHelper.activate_array_ongpu(RGpu, Outputs * Batch, Activation.Logistic);

                Blas.copy_ongpu(Outputs * Batch, StateGpu, 1, ForgotStateGpu, 1);
                Blas.mul_ongpu(Outputs * Batch, RGpu, 1, ForgotStateGpu, 1);

                s.Input = ForgotStateGpu;
                state_h_layer.forward_connected_layer_gpu(s);

                Blas.copy_ongpu(Outputs * Batch, input_h_layer.OutputGpu, 1, HGpu, 1);
                Blas.axpy_ongpu(Outputs * Batch, 1, state_h_layer.OutputGpu, 1, HGpu, 1);

                // USET ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, TANH);
                ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, Activation.Logistic);


                Blas.weighted_sum_gpu(StateGpu, HGpu, ZGpu, Outputs * Batch, OutputGpu);

                Blas.copy_ongpu(Outputs * Batch, OutputGpu, 1, StateGpu, 1);

                state.Input += Inputs * Batch;
                OutputGpu += Outputs * Batch;
                input_z_layer.increment_layer(1);
                input_r_layer.increment_layer(1);
                input_h_layer.increment_layer(1);

                state_z_layer.increment_layer(1);
                state_r_layer.increment_layer(1);
                state_h_layer.increment_layer(1);
            }
        }
        
        public void backward_gru_layer_gpu(NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_z_layer = (InputZLayer);
            Layer input_r_layer = (InputRLayer);
            Layer input_h_layer = (InputHLayer);

            Layer state_z_layer = (StateZLayer);
            Layer state_r_layer = (StateRLayer);
            Layer state_h_layer = (StateHLayer);

            input_z_layer.increment_layer(Steps - 1);
            input_r_layer.increment_layer(Steps - 1);
            input_h_layer.increment_layer(Steps - 1);

            state_z_layer.increment_layer(Steps - 1);
            state_r_layer.increment_layer(Steps - 1);
            state_h_layer.increment_layer(Steps - 1);

            state.Input += Inputs * Batch * (Steps - 1);
            if (state.Delta) state.Delta += Inputs * Batch * (Steps - 1);
            OutputGpu += Outputs * Batch * (Steps - 1);
            DeltaGpu += Outputs * Batch * (Steps - 1);
            for (i = Steps - 1; i >= 0; --i)
            {
                if (i != 0) Blas.copy_ongpu(Outputs * Batch, OutputGpu - Outputs * Batch, 1, PrevStateGpu, 1);
                float[] prev_delta_gpu = (i == 0) ? 0 : DeltaGpu - Outputs * Batch;

                Blas.copy_ongpu(Outputs * Batch, input_z_layer.OutputGpu, 1, ZGpu, 1);
                Blas.axpy_ongpu(Outputs * Batch, 1, state_z_layer.OutputGpu, 1, ZGpu, 1);

                Blas.copy_ongpu(Outputs * Batch, input_r_layer.OutputGpu, 1, RGpu, 1);
                Blas.axpy_ongpu(Outputs * Batch, 1, state_r_layer.OutputGpu, 1, RGpu, 1);

                ActivationsHelper.activate_array_ongpu(ZGpu, Outputs * Batch, Activation.Logistic);
                ActivationsHelper.activate_array_ongpu(RGpu, Outputs * Batch, Activation.Logistic);

                Blas.copy_ongpu(Outputs * Batch, input_h_layer.OutputGpu, 1, HGpu, 1);
                Blas.axpy_ongpu(Outputs * Batch, 1, state_h_layer.OutputGpu, 1, HGpu, 1);

                // USET ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, TANH);
                ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, Activation.Logistic);


                Blas.weighted_delta_gpu(PrevStateGpu, HGpu, ZGpu, prev_delta_gpu, input_h_layer.DeltaGpu, input_z_layer.DeltaGpu, Outputs * Batch, DeltaGpu);

                // USET ActivationsHelper.gradient_array_ongpu(HGpu, Outputs * Batch, TANH, input_h_layer.DeltaGpu);
                ActivationsHelper.gradient_array_ongpu(HGpu, Outputs * Batch, Activation.Logistic, input_h_layer.DeltaGpu);


                Blas.copy_ongpu(Outputs * Batch, input_h_layer.DeltaGpu, 1, state_h_layer.DeltaGpu, 1);

                Blas.copy_ongpu(Outputs * Batch, PrevStateGpu, 1, ForgotStateGpu, 1);
                Blas.mul_ongpu(Outputs * Batch, RGpu, 1, ForgotStateGpu, 1);
                Blas.fill_ongpu(Outputs * Batch, 0, ForgotDeltaGpu, 1);

                s.Input = ForgotStateGpu;
                s.Delta = ForgotDeltaGpu;

                state_h_layer.backward_connected_layer_gpu(s);
                if (prev_delta_gpu)
                {
                    Blas.mult_add_into_gpu(Outputs * Batch, ForgotDeltaGpu, RGpu, prev_delta_gpu);
                }
                Blas.mult_add_into_gpu(Outputs * Batch, ForgotDeltaGpu, PrevStateGpu, input_r_layer.DeltaGpu);

                ActivationsHelper.gradient_array_ongpu(RGpu, Outputs * Batch, Activation.Logistic, input_r_layer.DeltaGpu);
                Blas.copy_ongpu(Outputs * Batch, input_r_layer.DeltaGpu, 1, state_r_layer.DeltaGpu, 1);

                ActivationsHelper.gradient_array_ongpu(ZGpu, Outputs * Batch, Activation.Logistic, input_z_layer.DeltaGpu);
                Blas.copy_ongpu(Outputs * Batch, input_z_layer.DeltaGpu, 1, state_z_layer.DeltaGpu, 1);

                s.Input = PrevStateGpu;
                s.Delta = prev_delta_gpu;

                state_r_layer.backward_connected_layer_gpu(s);
                state_z_layer.backward_connected_layer_gpu(s);

                s.Input = state.Input;
                s.Delta = state.Delta;

                input_h_layer.backward_connected_layer_gpu(s);
                input_r_layer.backward_connected_layer_gpu(s);
                input_z_layer.backward_connected_layer_gpu(s);


                state.Input -= Inputs * Batch;
                if (state.Delta) state.Delta -= Inputs * Batch;
                OutputGpu -= Outputs * Batch;
                DeltaGpu -= Outputs * Batch;
                input_z_layer.increment_layer(-1);
                input_r_layer.increment_layer(-1);
                input_h_layer.increment_layer(-1);

                state_z_layer.increment_layer(-1);
                state_r_layer.increment_layer(-1);
                state_h_layer.increment_layer(-1);
            }
        }
        
        public static Layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, Activation activation)
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

            l.Weights = new float[c * n * size * size * locations];
            l.WeightUpdates = new float[c * n * size * size * locations];

            l.Biases = new float[l.Outputs];
            l.BiasUpdates = new float[l.Outputs];

            // float scale = 1./(float)Math.Sqrt(size*size*c);
            float scale = (float)Math.Sqrt(2.0 / (size * size * c));
            for (i = 0; i < c * n * size * size; ++i) l.Weights[i] = scale * Utils.rand_uniform(-1, 1);

            l.ColImage = new float[out_h * out_w * size * size * c];
            l.Output = new float[l.Batch * out_h * out_w * n];
            l.Delta = new float[l.Batch * out_h * out_w * n];

            l.Forward = forward_local_layer;
            l.Backward = backward_local_layer;
            l.Update = update_local_layer;


            l.ForwardGpu = forward_local_layer_gpu;
            l.BackwardGpu = backward_local_layer_gpu;
            l.UpdateGpu = update_local_layer_gpu;

            l.WeightsGpu = (float[])l.Weights.Clone();
            l.WeightUpdatesGpu = (float[])l.WeightUpdates.Clone();

            l.BiasesGpu = (float[])l.Biases.Clone();
            l.BiasUpdatesGpu = (float[])l.BiasUpdates.Clone();

            l.ColImageGpu = (float[])l.ColImage.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            l.OutputGpu = (float[])l.Output.Clone();


            l.Activation = activation;

            Console.Error.Write($"Local Layer: %d x %d x %d Image, %d filters . %d x %d x %d Image\n", h, w, c, n, out_h, out_w, n);

            return l;
        }
        
        public void pull_local_layer()
        {
            int locations = OutW * OutH;
            int size = Size * Size * C * N * locations;
            Array.Copy(WeightsGpu, Weights, size);
            Array.Copy(BiasesGpu, Biases, Outputs);
        }
        
        public void push_local_layer()
        {
            int locations = OutW * OutH;
            int size = Size * Size * C * N * locations;
            Array.Copy(Weights, WeightsGpu, size);
            Array.Copy(Biases, BiasesGpu, Outputs);
        }
        
        public Image get_maxpool_image()
        {
            int h = OutH;
            int w = OutW;
            int c = C;
            return new Image(w, h, c, Output);
        }
        
        public Image get_maxpool_delta()
        {
            int h = OutH;
            int w = OutW;
            int c = C;
            return new Image(w, h, c, Delta);
        }
        
        public static Layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Maxpool;
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
            l.Indexes = new int[output_size];
            l.Output = new float[output_size];
            l.Delta = new float[output_size];
            l.Forward = forward_maxpool_layer;
            l.Backward = backward_maxpool_layer;

            l.ForwardGpu = forward_maxpool_layer_gpu;
            l.BackwardGpu = backward_maxpool_layer_gpu;
            l.IndexesGpu = cuda_make_int_array(output_size);
            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();

            Console.Error.Write($"max          %d x %d / %d  %4d x%4d x%4d   .  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.OutW, l.OutH, l.OutC);
            return l;
        }
        
        public void resize_maxpool_layer(int w, int h)
        {
            H = h;
            W = w;
            Inputs = h * w * C;

            OutW = (w + 2 * Pad) / Stride;
            OutH = (h + 2 * Pad) / Stride;
            Outputs = OutW * OutH * C;
            int output_size = Outputs * Batch;

            Array.Resize(ref Indexes, output_size);
            Array.Resize(ref Output, output_size);
            Array.Resize(ref Delta, output_size);

            IndexesGpu = cuda_make_int_array(output_size);
            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();

        }
        
        public void forward_maxpool_layer(NetworkState state)
        {
            int b, i, j, k, m, n;
            int w_offset = -Pad;
            int h_offset = -Pad;

            int h = OutH;
            int w = OutW;
            int c = C;

            for (b = 0; b < Batch; ++b)
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
                            for (n = 0; n < Size; ++n)
                            {
                                for (m = 0; m < Size; ++m)
                                {
                                    int cur_h = h_offset + i * Stride + n;
                                    int cur_w = w_offset + j * Stride + m;
                                    int index = cur_w + W * (cur_h + H * (k + b * C));
                                    int valid = (cur_h >= 0 && cur_h < H &&
                                                 cur_w >= 0 && cur_w < W);
                                    float val = (valid != 0) ? state.Input[index] : float.MinValue;
                                    max_i = (val > max) ? index : max_i;
                                    max = (val > max) ? val : max;
                                }
                            }
                            Output[out_index] = max;
                            Indexes[out_index] = max_i;
                        }
                    }
                }
            }
        }
        
        public void backward_maxpool_layer(NetworkState state)
        {
            int i;
            int h = OutH;
            int w = OutW;
            int c = C;
            for (i = 0; i < h * w * c * Batch; ++i)
            {
                int index = Indexes[i];
                state.Delta[index] += Delta[i];
            }
        }
        
        public static Layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
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
            layer.Output = new float[h * w * c * batch];
            layer.Delta = new float[h * w * c * batch];
            layer.Squared = new float[h * w * c * batch];
            layer.Norms = new float[h * w * c * batch];
            layer.Inputs = w * h * c;
            layer.Outputs = layer.Inputs;

            layer.Forward = forward_normalization_layer;
            layer.Backward = backward_normalization_layer;

            layer.ForwardGpu = forward_normalization_layer_gpu;
            layer.BackwardGpu = backward_normalization_layer_gpu;

            layer.OutputGpu = (float[])layer.Output.Clone();
            layer.DeltaGpu = (float[])layer.Delta.Clone();
            layer.SquaredGpu = (float[])layer.Squared.Clone();
            layer.NormsGpu = (float[])layer.Norms.Clone();

            return layer;
        }
        
        public void resize_normalization_layer(int w, int h)
        {
            int c = C;
            int batch = Batch;
            H = h;
            W = w;
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
        
        public void forward_normalization_layer(NetworkState state)
        {
            int k, b;
            int w = W;
            int h = H;
            int c = C;
            Blas.Scal_cpu(w * h * c * Batch, 0, Squared, 1);

            for (b = 0; b < Batch; ++b)
            {
                float[] squared = Squared + w * h * c * b;
                float[] norms = Norms + w * h * c * b;
                float[] input = state.Input + w * h * c * b;
                Blas.Pow_cpu(w * h * c, 2, input, 1, squared, 1);

                Blas.Const_cpu(w * h, Kappa, norms, 1);
                for (k = 0; k < Size / 2; ++k)
                {
                    Blas.Axpy_cpu(w * h, Alpha, squared + w * h * k, 1, norms, 1);
                }

                for (k = 1; k < C; ++k)
                {
                    Blas.Copy_cpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
                    int prev = k - ((Size - 1) / 2) - 1;
                    int next = k + (Size / 2);
                    if (prev >= 0) Blas.Axpy_cpu(w * h, -Alpha, squared + w * h * prev, 1, norms + w * h * k, 1);
                    if (next < C) Blas.Axpy_cpu(w * h, Alpha, squared + w * h * next, 1, norms + w * h * k, 1);
                }
            }
            Blas.Pow_cpu(w * h * c * Batch, -Beta, Norms, 1, Output, 1);
            Blas.Mul_cpu(w * h * c * Batch, state.Input, 1, Output, 1);
        }
        
        public void backward_normalization_layer(NetworkState state)
        {
            // TODO This is approximate ;-)
            // Also this should add in to delta instead of overwritting.

            int w = W;
            int h = H;
            int c = C;
            Blas.Pow_cpu(w * h * c * Batch, -Beta, Norms, 1, state.Delta, 1);
            Blas.Mul_cpu(w * h * c * Batch, Delta, 1, state.Delta, 1);
        }
        
        public void forward_normalization_layer_gpu(NetworkState state)
        {
            int k, b;
            int w = W;
            int h = H;
            int c = C;
            Blas.scal_ongpu(w * h * c * Batch, 0, SquaredGpu, 1);

            for (b = 0; b < Batch; ++b)
            {
                float[] squared = SquaredGpu + w * h * c * b;
                float[] norms = NormsGpu + w * h * c * b;
                float[] input = state.Input + w * h * c * b;
                Blas.pow_ongpu(w * h * c, 2, input, 1, squared, 1);

                Blas.const_ongpu(w * h, Kappa, norms, 1);
                for (k = 0; k < Size / 2; ++k)
                {
                    Blas.axpy_ongpu(w * h, Alpha, squared + w * h * k, 1, norms, 1);
                }

                for (k = 1; k < C; ++k)
                {
                    Blas.copy_ongpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
                    int prev = k - ((Size - 1) / 2) - 1;
                    int next = k + (Size / 2);
                    if (prev >= 0) Blas.axpy_ongpu(w * h, -Alpha, squared + w * h * prev, 1, norms + w * h * k, 1);
                    if (next < C) Blas.axpy_ongpu(w * h, Alpha, squared + w * h * next, 1, norms + w * h * k, 1);
                }
            }
            Blas.pow_ongpu(w * h * c * Batch, -Beta, NormsGpu, 1, OutputGpu, 1);
            Blas.mul_ongpu(w * h * c * Batch, state.Input, 1, OutputGpu, 1);
        }
        
        public void backward_normalization_layer_gpu(NetworkState state)
        {
            // TODO This is approximate ;-)

            int w = W;
            int h = H;
            int c = C;
            Blas.pow_ongpu(w * h * c * Batch, -Beta, NormsGpu, 1, state.Delta, 1);
            Blas.mul_ongpu(w * h * c * Batch, DeltaGpu, 1, state.Delta, 1);
        }

        private static void binarize_kernel(float[] x, int n, float[] binary)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            binary[i] = (x[i] >= 0) ? 1 : -1;
        }

        [GpuManaged]
        public void binarize_gpu(float[] x, int n, float[] binary)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(binarize_kernel, lp, x, n, binary);
        }

        private static void binarize_input_kernel(float[] input, int n, int size, float[] binary)
        {
            int s = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (s >= size) return;
            int i = 0;
            float mean = 0;
            for (i = 0; i < n; ++i)
            {
                mean += Math.Abs(input[i * size + s]);
            }
            mean = mean / n;
            for (i = 0; i < n; ++i)
            {
                binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
            }
        }

        [GpuManaged]
        public void binarize_input_gpu(float[] input, int n, int size, float[] binary)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(binarize_input_kernel, lp, input, n, size, binary);
        }
        
        private static void binarize_weights_kernel(float[] weights, int n, int size, float[] binary)
        {
            int f = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (f >= n) return;
            int i = 0;
            float mean = 0;
            for (i = 0; i < size; ++i)
            {
                mean += Math.Abs(weights[f * size + i]);
            }
            mean = mean / size;
            for (i = 0; i < size; ++i)
            {
                binary[f * size + i] = (weights[f * size + i] > 0) ? mean : -mean;
            }
        }

        [GpuManaged]
        public void binarize_weights_gpu(float[] weights, int n, int size, float[] binary)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(binarize_weights_kernel, lp, weights, n, size, binary);
        }
        
        public void forward_convolutional_layer_gpu( NetworkState state)
        {
            Blas.fill_ongpu(Outputs * Batch, 0, OutputGpu, 1);
            if (Binary)
            {
                binarize_weights_gpu(WeightsGpu, N, C * Size * Size, BinaryWeightsGpu);
                swap_binary();
            }

            if (Xnor)
            {
                binarize_weights_gpu(WeightsGpu, N, C * Size * Size, BinaryWeightsGpu);
                swap_binary();
                binarize_gpu(state.Input, C * H * W * Batch, BinaryInputGpu);
                state.Input = BinaryInputGpu;
            }

            float one = 1;
            cudnnConvolutionForward(cudnn_handle(),
                        one,
                        SrcTensorDesc,
                        state.Input,
                        WeightDesc,
                        WeightsGpu,
                        ConvDesc,
                        FwAlgo,
                        state.Workspace,
                        WorkspaceSize,
                        one,
                        DstTensorDesc,
                        OutputGpu);
            
            if (BatchNormalize)
            {
                forward_batchnorm_layer_gpu(state);
            }
            Blas.add_bias_gpu(OutputGpu, BiasesGpu, Batch, N, OutW * OutH);

            ActivationsHelper.activate_array_ongpu(OutputGpu, Outputs * Batch, Activation);
            if (Binary || Xnor) swap_binary();
        }

        public void backward_convolutional_layer_gpu( NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, DeltaGpu);

            Blas.backward_bias_gpu(BiasUpdatesGpu, DeltaGpu, Batch, N, OutW * OutH);

            if (BatchNormalize)
            {
                backward_batchnorm_layer_gpu( state);
            }
            float[] original_input = state.Input;

            if (Xnor) state.Input = BinaryInputGpu;
            float one = 1;
            CuDnn.cudnnConvolutionBackwardFilter(CudaUtils.cudnn_handle(),
                    one,
                    SrcTensorDesc,
                    state.Input,
                    DdstTensorDesc,
                    DeltaGpu,
                    ConvDesc,
                    BfAlgo,
                    state.Workspace,
                    WorkspaceSize,
                    one,
                    DweightDesc,
                    WeightUpdatesGpu);

            if (state.Delta)
            {
                if (Binary || Xnor) swap_binary();
                CuDnn.cudnnConvolutionBackwardData(cudnn_handle(),
                        one,
                        WeightDesc,
                        WeightsGpu,
                        DdstTensorDesc,
                        DeltaGpu,
                        ConvDesc,
                        BdAlgo,
                        state.Workspace,
                        WorkspaceSize,
                        one,
                        DsrcTensorDesc,
                        state.Delta);
                if (Binary || Xnor) swap_binary();
                if (Xnor) ActivationsHelper.gradient_array_ongpu(original_input, Batch * C * H * W, Activation.Hardtan, state.Delta);
            }
        }

        public void pull_convolutional_layer()
        {
            Array.Copy(WeightsGpu, Weights, C * N * Size * Size);
            Array.Copy(BiasesGpu, Biases, N);
            Array.Copy(WeightUpdatesGpu, WeightUpdates, C * N * Size * Size);
            Array.Copy(BiasUpdatesGpu, BiasUpdates, N);
            if (BatchNormalize)
            {
                Array.Copy(ScalesGpu, Scales, N);
                Array.Copy(RollingMeanGpu, RollingMean, N);
                Array.Copy(RollingVarianceGpu, RollingVariance, N);
            }
            if (Adam)
            {
                Array.Copy(MGpu, M, C * N * Size * Size);
                Array.Copy(VGpu, V, C * N * Size * Size);
            }
        }

        public void push_convolutional_layer()
        {
            Array.Copy(Weights, WeightsGpu, C * N * Size * Size);
            Array.Copy(Biases, BiasesGpu, N);
            Array.Copy(WeightUpdates, WeightUpdatesGpu, C * N * Size * Size);
            Array.Copy(BiasUpdates, BiasUpdatesGpu, N);
            if (BatchNormalize)
            {
                Array.Copy(Scales, ScalesGpu, N);
                Array.Copy(RollingMean, RollingMeanGpu, N);
                Array.Copy(RollingVariance, RollingVarianceGpu, N);
            }
            if (Adam)
            {
                Array.Copy(M, MGpu, C * N * Size * Size);
                Array.Copy(V, VGpu, C * N * Size * Size);
            }
        }

        public void update_convolutional_layer_gpu( int batch, float learning_rate, float momentum, float decay)
        {
            int size = Size * Size * C * N;
            Blas.axpy_ongpu(N, learning_rate / batch, BiasUpdatesGpu, 1, BiasesGpu, 1);
            Blas.scal_ongpu(N, momentum, BiasUpdatesGpu, 1);

            if (ScalesGpu.Any())
            {
                Blas.axpy_ongpu(N, learning_rate / batch, ScaleUpdatesGpu, 1, ScalesGpu, 1);
                Blas.scal_ongpu(N, momentum, ScaleUpdatesGpu, 1);
            }

            if (Adam)
            {
                Blas.scal_ongpu(size, B1, MGpu, 1);
                Blas.scal_ongpu(size, B2, VGpu, 1);

                Blas.axpy_ongpu(size, -decay * batch, WeightsGpu, 1, WeightUpdatesGpu, 1);

                Blas.axpy_ongpu(size, -(1 - B1), WeightUpdatesGpu, 1, MGpu, 1);
                Blas.mul_ongpu(size, WeightUpdatesGpu, 1, WeightUpdatesGpu, 1);
                Blas.axpy_ongpu(size, (1 - B2), WeightUpdatesGpu, 1, VGpu, 1);

                Blas.adam_gpu(size, WeightsGpu, MGpu, VGpu, B1, B2, learning_rate / batch, Eps, T + 1);
                Blas.fill_ongpu(size, 0, WeightUpdatesGpu, 1);
            }
            else
            {
                Blas.axpy_ongpu(size, -decay * batch, WeightsGpu, 1, WeightUpdatesGpu, 1);
                Blas.axpy_ongpu(size, learning_rate / batch, WeightUpdatesGpu, 1, WeightsGpu, 1);
                Blas.scal_ongpu(size, momentum, WeightUpdatesGpu, 1);
            }
        }

    }
}