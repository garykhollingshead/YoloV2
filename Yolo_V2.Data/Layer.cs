using System;
using System.Linq;
using System.Runtime.InteropServices;
using Alea;
using Alea.CSharp;
using Alea.CudaDnn;
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
        public bool Shortcut;
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
        public bool Reverse;
        public int Pad;
        public bool Sqrt;
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
        public bool Softmax;
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
        public bool BiasMatch;
        public int Random;
        public float Thresh;
        public int Classfix;
        public int Absolute;

        public bool Dontload;
        public bool Dontloadscales;

        public float Temperature;
        public float Probability;
        public float Scale;

        public int[] Indexes;
        public float[] Rand;
        public float? Cost;
        public string Cweights;
        public int StateIndex;
        public float[] State;
        public float[] StateBackup;
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
        public int[] InputLayers;
        public int[] InputSizes;
        public int DeltaIndex;
        public float[] DeltaBackup;
        public float[] Delta;
        public int OutputIndex;
        public float[] OutputBackup;
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

        public int XIndex;
        public float[] XBackup;
        public float[] X;
        public int XNormIndex;
        public float[] XNormBackup;
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

        public int[] IndexesGpu;
        public float[] PrevStateGpu;
        public float[] ForgotStateGpu;
        public float[] ForgotDeltaGpu;
        public int StateGpuIndex;
        public float[] StateGpuBackup;
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

        public int XGpuIndex;
        public float[] XGpuBackup;
        public float[] XGpu;
        public int XNormGpuIndex;
        public float[] XNormGpuBackup;
        public float[] XNormGpu;
        public float[] WeightsGpu;
        public float[] WeightUpdatesGpu;

        public float[] BiasesGpu;
        public float[] BiasUpdatesGpu;

        public float[] ScalesGpu;
        public float[] ScaleUpdatesGpu;

        public int OutputGpuIndex;
        public float[] OutputGpuBackup;
        public float[] OutputGpu;
        public int DeltaGpuIndex;
        public float[] DeltaGpuBackup;
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

        private int DOABS = 1;

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
            Layer l = new Layer();
            l.LayerType = LayerType.Batchnorm;
            l.Batch = batch;
            l.H = l.OutH = h;
            l.W = l.OutW = w;
            l.C = l.OutC = c;
            l.Output = new float[h * w * c * batch];
            l.Delta = new float[h * w * c * batch];
            l.Inputs = w * h * c;
            l.Outputs = l.Inputs;

            l.Scales = new float[c];
            l.ScaleUpdates = new float[c];
            int i;
            for (i = 0; i < c; ++i)
            {
                l.Scales[i] = 1;
            }

            l.Mean = new float[c];
            l.Variance = new float[c];

            l.RollingMean = new float[c];
            l.RollingVariance = new float[c];

            l.Forward = forward_batchnorm_layer;
            l.Backward = backward_batchnorm_layer;
            l.ForwardGpu = forward_batchnorm_layer_gpu;
            l.BackwardGpu = backward_batchnorm_layer_gpu;

            l.OutputGpu = new float[h * w * c * batch];
            l.DeltaGpu = new float[h * w * c * batch];

            l.ScalesGpu = new float[c];
            l.ScaleUpdatesGpu = new float[c];

            l.MeanGpu = new float[c];
            l.VarianceGpu = new float[c];

            l.RollingMeanGpu = new float[c];
            l.RollingVarianceGpu = new float[c];

            l.MeanDeltaGpu = new float[c];
            l.VarianceDeltaGpu = new float[c];

            l.XGpu = new float[l.Batch * l.Outputs];
            l.XNormGpu = new float[l.Batch * l.Outputs];
            return l;
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

        public static void forward_batchnorm_layer(Layer l, NetworkState state)
        {
            if (l.LayerType == LayerType.Batchnorm)
            {
                Blas.Copy_cpu(l.Outputs * l.Batch, state.Input, l.Output);
            }
            if (l.LayerType == LayerType.Connected)
            {
                l.OutC = l.Outputs;
                l.OutH = l.OutW = 1;
            }
            if (state.Train)
            {
                Blas.Mean_cpu(l.Output, l.Batch, l.OutC, l.OutH * l.OutW, l.Mean);
                Blas.Variance_cpu(l.Output, l.Mean, l.Batch, l.OutC, l.OutH * l.OutW, l.Variance);

                Blas.Scal_cpu(l.OutC, .9f, l.RollingMean, 1);
                Blas.Axpy_cpu(l.OutC, .1f, l.Mean, l.RollingMean);
                Blas.Scal_cpu(l.OutC, .9f, l.RollingVariance, 1);
                Blas.Axpy_cpu(l.OutC, .1f, l.Variance, l.RollingVariance);

                Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, l.X);
                Blas.Normalize_cpu(l.Output, l.Mean, l.Variance, l.Batch, l.OutC, l.OutH * l.OutW);
                Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, l.XNorm);
            }
            else
            {
                Blas.Normalize_cpu(l.Output, l.RollingMean, l.RollingVariance, l.Batch, l.OutC, l.OutH * l.OutW);
            }
            Blas.Scale_bias(l.Output, l.Scales, l.Batch, l.OutC, l.OutH * l.OutW);
        }

        public static void backward_batchnorm_layer(Layer l, NetworkState state)
        {
            backward_scale_cpu(l.XNorm, l.Delta, l.Batch, l.OutC, l.OutW * l.OutH, l.ScaleUpdates);

            Blas.Scale_bias(l.Delta, l.Scales, l.Batch, l.OutC, l.OutH * l.OutW);

            mean_delta_cpu(l.Delta, l.Variance, l.Batch, l.OutC, l.OutW * l.OutH, l.MeanDelta);
            variance_delta_cpu(l.X, l.Delta, l.Mean, l.Variance, l.Batch, l.OutC, l.OutW * l.OutH, l.VarianceDelta);
            normalize_delta_cpu(l.X, l.Mean, l.Variance, l.MeanDelta, l.VarianceDelta, l.Batch, l.OutC, l.OutW * l.OutH, l.Delta);
            if (l.LayerType == LayerType.Batchnorm) Blas.Copy_cpu(l.Outputs * l.Batch, l.Delta, state.Delta);
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

        public static void forward_batchnorm_layer_gpu(Layer l, NetworkState state)
        {
            if (l.LayerType == LayerType.Batchnorm)
            {
                Blas.copy_ongpu(l.Outputs * l.Batch, state.Input, l.OutputGpu);
            }
            if (l.LayerType == LayerType.Connected)
            {
                l.OutC = l.Outputs;
                l.OutH = l.OutW = 1;
            }
            if (state.Train)
            {
                Blas.fast_mean_gpu(l.OutputGpu, l.Batch, l.OutC, l.OutH * l.OutW, l.MeanGpu);
                Blas.fast_variance_gpu(l.OutputGpu, l.MeanGpu, l.Batch, l.OutC, l.OutH * l.OutW, l.VarianceGpu);

                Blas.scal_ongpu(l.OutC, .99f, l.RollingMeanGpu, 1);
                Blas.axpy_ongpu(l.OutC, .01f, l.MeanGpu, l.RollingMeanGpu);
                Blas.scal_ongpu(l.OutC, .99f, l.RollingVarianceGpu, 1);
                Blas.axpy_ongpu(l.OutC, .01f, l.VarianceGpu, l.RollingVarianceGpu);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.OutputGpu, l.XGpu);
                Blas.normalize_gpu(l.OutputGpu, l.MeanGpu, l.VarianceGpu, l.Batch, l.OutC, l.OutH * l.OutW);
                Blas.copy_ongpu(l.Outputs * l.Batch, l.OutputGpu, l.XNormGpu);
            }
            else
            {
                Blas.normalize_gpu(l.OutputGpu, l.RollingMeanGpu, l.RollingVarianceGpu, l.Batch, l.OutC, l.OutH * l.OutW);
            }

            Blas.scale_bias_gpu(l.OutputGpu, l.ScalesGpu, l.Batch, l.OutC, l.OutH * l.OutW);
        }

        public static void backward_batchnorm_layer_gpu(Layer l, NetworkState state)
        {
            Blas.backward_scale_gpu(l.XNormGpu, l.DeltaGpu, l.Batch, l.OutC, l.OutW * l.OutH, l.ScaleUpdatesGpu);

            Blas.scale_bias_gpu(l.DeltaGpu, l.ScalesGpu, l.Batch, l.OutC, l.OutH * l.OutW);

            Blas.fast_mean_delta_gpu(l.DeltaGpu, l.VarianceGpu, l.Batch, l.OutC, l.OutW * l.OutH, l.MeanDeltaGpu);
            Blas.fast_variance_delta_gpu(l.XGpu, l.DeltaGpu, l.MeanGpu, l.VarianceGpu, l.Batch, l.OutC, l.OutW * l.OutH, l.VarianceDeltaGpu);
            Blas.normalize_delta_gpu(l.XGpu, l.MeanGpu, l.VarianceGpu, l.MeanDeltaGpu, l.VarianceDeltaGpu, l.Batch, l.OutC, l.OutW * l.OutH, l.DeltaGpu);
            if (l.LayerType == LayerType.Batchnorm) Blas.copy_ongpu(l.Outputs * l.Batch, l.DeltaGpu, state.Delta);
        }

        private static void forward_local_layer(Layer l, NetworkState state)
        {
            int out_h = l.local_out_height();
            int out_w = l.local_out_width();
            int i, j;
            int locations = out_h * out_w;

            for (i = 0; i < l.Batch; ++i)
            {
                var index = i * l.Outputs;
                var output = new float[l.Outputs];
                Blas.Copy_cpu(l.Outputs, l.Biases, output, 0, index);
                CombineLists(l.Output, index, output);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                var index = i * l.W * l.H * l.C;
                Im2Col.im2col_cpu(state.Input, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, l.ColImage, index);

                index = i * l.Outputs;
                float[] output = new float[l.Output.Length - index];
                Array.Copy(l.Output, index, output, 0, output.Length);

                for (j = 0; j < locations; ++j)
                {
                    index = j * l.Size * l.Size * l.C * l.N;
                    float[] a = new float[l.Weights.Length - index];
                    float[] b = new float[l.ColImage.Length - j];
                    float[] c = new float[output.Length - j];
                    Array.Copy(l.Weights, index, a, 0, a.Length);
                    Array.Copy(l.ColImage, j, b, 0, b.Length);
                    Array.Copy(output, j, c, 0, c.Length);

                    int m = l.N;
                    int n = 1;
                    int k = l.Size * l.Size * l.C;

                    Gemm.gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                    Array.Copy(c, 0, output, j, c.Length);
                }
                Array.Copy(output, 0, l.Output, index, output.Length);
            }
            ActivationsHelper.Activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        private static void backward_local_layer(Layer l, NetworkState state)
        {
            int i, j;
            int locations = l.OutW * l.OutH;

            ActivationsHelper.Gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);

            for (i = 0; i < l.Batch; ++i)
            {
                var index = i * l.Outputs;
                Blas.Axpy_cpu(l.Outputs, 1, l.Delta.Skip(index).ToArray(), l.BiasUpdates);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                var index = i * l.W * l.H * l.C;
                float[] input = state.Input.Skip(index).ToArray();
                Im2Col.im2col_cpu(input, l.C, l.H, l.W,
                    l.Size, l.Stride, l.Pad, l.ColImage);

                for (j = 0; j < locations; ++j)
                {
                    var indexA = i * l.Outputs + j;
                    float[] a = l.Delta.Skip(indexA).ToArray();
                    float[] b = l.ColImage.Skip(j).ToArray();
                    var indexC = j * l.Size * l.Size * l.C * l.N;
                    float[] c = l.WeightUpdates.Skip(indexC).ToArray();
                    int m = l.N;
                    int n = l.Size * l.Size * l.C;
                    int k = 1;

                    Gemm.gemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
                    CombineLists(l.Delta, indexA, a);
                    CombineLists(l.ColImage, j, b);
                    CombineLists(l.WeightUpdates, indexC, c);
                }

                if (state.Delta.Any())
                {
                    for (j = 0; j < locations; ++j)
                    {
                        var indexA = j * l.Size * l.Size * l.C * l.N;
                        var indexB = i * l.Outputs + j;
                        float[] a = l.Weights.Skip(indexA).ToArray();
                        float[] b = l.Delta.Skip(indexB).ToArray();
                        float[] c = l.ColImage.Skip(j).ToArray();

                        int m = l.Size * l.Size * l.C;
                        int n = 1;
                        int k = l.N;

                        Gemm.gemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                        CombineLists(l.Weights, indexA, a);
                        CombineLists(l.Delta, indexB, b);
                        CombineLists(l.ColImage, j, c);
                    }

                    index = i * l.C * l.H * l.W;
                    var output = state.Delta.Skip(index).ToArray();
                    Im2Col.col2im_cpu(l.ColImage, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, output);
                    CombineLists(state.Delta, index, output);
                }
            }
        }

        private static void update_local_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            Blas.Axpy_cpu(l.Outputs, learning_rate / batch, l.BiasUpdates, l.Biases);
            Blas.Scal_cpu(l.Outputs, momentum, l.BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay * batch, l.Weights, l.WeightUpdates);
            Blas.Axpy_cpu(size, learning_rate / batch, l.WeightUpdates, l.Weights);
            Blas.Scal_cpu(size, momentum, l.WeightUpdates, 1);
        }

        private static void forward_local_layer_gpu(Layer l, NetworkState state)
        {
            int out_h = l.local_out_height();
            int out_w = l.local_out_width();
            int i, j;
            int locations = out_h * out_w;

            //for (i = 0; i < Batch; ++i)
            //{
            //    Blas.copy_ongpu(Outputs, BiasesGpu, 1, Output.Gpu + i * Outputs, 1);
            //}

            for (i = 0; i < l.Batch; ++i)
            {
                var inIndex = i * l.W * l.H * l.C;
                float[] input = state.Input.Skip(inIndex).ToArray();
                Im2Col.im2col_ongpu(input, l.C, l.H, l.W,
                    l.Size, l.Stride, l.Pad, l.ColImageGpu);
                var outIndex = i * l.Outputs;
                float[] output = l.OutputGpu.Skip(outIndex).ToArray();
                for (j = 0; j < locations; ++j)
                {
                    var aIndex = j * l.Size * l.Size * l.C * l.N;
                    float[] a = l.WeightsGpu.Skip(aIndex).ToArray();
                    float[] b = l.ColImageGpu.Skip(j).ToArray();
                    float[] c = output.Skip(j).ToArray();

                    int m = l.N;
                    int n = 1;
                    int k = l.Size * l.Size * l.C;

                    Gemm.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                    CombineLists(l.WeightsGpu, aIndex, a);
                    CombineLists(l.ColImageGpu, j, b);
                    CombineLists(output, j, c);
                }
                CombineLists(state.Input, inIndex, input);
                CombineLists(l.OutputGpu, outIndex, output);
            }
            ActivationsHelper.activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }

        private static void backward_local_layer_gpu(Layer l, NetworkState state)
        {
            int i, j;
            int locations = l.OutW * l.OutH;

            ActivationsHelper.gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            for (i = 0; i < l.Batch; ++i)
            {
                var tmp = l.DeltaGpu.Skip(i * l.Outputs).ToArray();
                Blas.axpy_ongpu(l.Outputs, 1, tmp, l.BiasUpdatesGpu);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                int index = i * l.W * l.H * l.C;
                float[] input = state.Input.Skip(index).ToArray();
                Im2Col.im2col_ongpu(input, l.C, l.H, l.W,
                    l.Size, l.Stride, l.Pad, l.ColImageGpu);
                CombineLists(state.Input, index, input);
                for (j = 0; j < locations; ++j)
                {
                    int aIndex = i * l.Outputs + j;
                    int cIndex = j * l.Size * l.Size * l.C * l.N;
                    float[] a = l.DeltaGpu.Skip(aIndex).ToArray();
                    float[] b = l.ColImageGpu.Skip(j).ToArray();
                    float[] c = l.WeightUpdatesGpu.Skip(cIndex).ToArray();
                    int m = l.N;
                    int n = l.Size * l.Size * l.C;
                    int k = 1;

                    Gemm.gemm_ongpu(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
                    CombineLists(l.DeltaGpu, aIndex, a);
                    CombineLists(l.ColImageGpu, j, b);
                    CombineLists(l.WeightUpdatesGpu, cIndex, c);
                }

                if (state.Delta.Any())
                {
                    for (j = 0; j < locations; ++j)
                    {
                        int aIndex = j * l.Size * l.Size * l.C * l.N;
                        int bIndex = i * l.Outputs + j;
                        float[] a = l.WeightsGpu.Skip(aIndex).ToArray();
                        float[] b = l.DeltaGpu.Skip(bIndex).ToArray();
                        float[] c = l.ColImageGpu.Skip(j).ToArray();

                        int m = l.Size * l.Size * l.C;
                        int n = 1;
                        int k = l.N;

                        Gemm.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                        CombineLists(l.WeightsGpu, aIndex, a);
                        CombineLists(l.DeltaGpu, bIndex, b);
                        CombineLists(l.ColImageGpu, j, c);
                    }

                    var dIndex = i * l.C * l.H * l.W;
                    var delta = state.Delta.Skip(dIndex).ToArray();
                    Im2Col.col2im_ongpu(l.ColImageGpu, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, delta);
                    CombineLists(state.Delta, dIndex, delta);
                }
            }
        }

        private static void update_local_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            Blas.axpy_ongpu(l.Outputs, learning_rate / batch, l.BiasUpdatesGpu, l.BiasesGpu);
            Blas.scal_ongpu(l.Outputs, momentum, l.BiasUpdatesGpu, 1);

            Blas.axpy_ongpu(size, -decay * batch, l.WeightsGpu, l.WeightUpdatesGpu);
            Blas.axpy_ongpu(size, learning_rate / batch, l.WeightUpdatesGpu, l.WeightsGpu);
            Blas.scal_ongpu(size, momentum, l.WeightUpdatesGpu, 1);
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
            if (adam)
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
                if (adam)
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
                l.cudnn_convolutional_setup();
            }
            l.WorkspaceSize = l.get_workspace_size();
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
            Layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, Activation.Leaky, true, false, false, false);
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
            forward_convolutional_layer(l, state);
        }

        public void resize_convolutional_layer(int w, int h)
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
                forward_batchnorm_layer(l, state);
            }
            add_bias(l.Output, l.Biases, l.Batch, l.N, out_h * out_w);

            ActivationsHelper.Activate_array(l.Output, m * n * l.Batch, l.Activation);
            if (l.Binary || l.Xnor) l.swap_binary();
        }

        public static void backward_convolutional_layer(Layer l, NetworkState state)
        {
            int i;
            int m = l.N;
            int n = l.Size * l.Size * l.C;
            int k = l.convolutional_out_height() *
                    l.convolutional_out_width();

            ActivationsHelper.Gradient_array(l.Output, m * k * l.Batch, l.Activation, l.Delta);
            backward_bias(l.BiasUpdates, l.Delta, l.Batch, l.N, k);

            if (l.BatchNormalize)
            {
                backward_batchnorm_layer(l, state);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] a = l.Delta.Skip(i * m * k).ToArray();
                float[] b = state.Workspace;
                float[] c = l.WeightUpdates;

                float[] im = state.Input.Skip(i * l.C * l.H * l.W).ToArray();

                Im2Col.im2col_cpu(im, l.C, l.H, l.W,
                    l.Size, l.Stride, l.Pad, b);
                Gemm.gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

                Array.Copy(a, 0, l.Delta, i * m * k, a.Length);
                Array.Copy(im, 0, state.Input, i * l.C * l.H * l.W, im.Length);
                if (state.Delta.Any())
                {
                    a = l.Weights;
                    b = l.Delta.Skip(i * m * k).ToArray();
                    Array.Copy(l.Delta, i * m * k, b, 0, b.Length);
                    c = state.Workspace;
                    var d = state.Delta.Skip(i * l.C * l.H * l.W).ToArray();

                    Gemm.gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

                    Im2Col.col2im_cpu(state.Workspace, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, d);
                    Array.Copy(b, 0, l.Delta, i * m * k, b.Length);
                    Array.Copy(d, 0, state.Delta, i * l.C * l.H * l.W, d.Length);

                }
            }
        }

        public static void update_convolutional_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int size = l.Size * l.Size * l.C * l.N;
            Blas.Axpy_cpu(l.N, learning_rate / batch, l.BiasUpdates, l.Biases);
            Blas.Scal_cpu(l.N, momentum, l.BiasUpdates, 1);

            if (l.Scales.Any())
            {
                Blas.Axpy_cpu(l.N, learning_rate / batch, l.ScaleUpdates, l.Scales);
                Blas.Scal_cpu(l.N, momentum, l.ScaleUpdates, 1);
            }

            Blas.Axpy_cpu(size, -decay * batch, l.Weights, l.WeightUpdates);
            Blas.Axpy_cpu(size, learning_rate / batch, l.WeightUpdates, l.Weights);
            Blas.Scal_cpu(size, momentum, l.WeightUpdates, 1);
        }

        public Image get_convolutional_weight(int i)
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
                Image im = get_convolutional_weight(i);
                if (im.C == 3)
                {
                    LoadArgs.rgbgr_image(im);
                }
            }
        }

        public void rescale_weights(float scale, float trans)
        {
            int i;
            for (i = 0; i < N; ++i)
            {
                Image im = get_convolutional_weight(i);
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
                weights[i] = new Image(get_convolutional_weight(i));
            }
            return weights;
        }

        public Image[] visualize_convolutional_layer(string window)
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

        public static void forward_activation_layer(Layer l, NetworkState state)
        {
            Blas.Copy_cpu(l.Outputs * l.Batch, state.Input, l.Output);
            ActivationsHelper.Activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        public static void backward_activation_layer(Layer l, NetworkState state)
        {
            ActivationsHelper.Gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);
            Blas.Copy_cpu(l.Outputs * l.Batch, l.Delta, state.Delta);
        }

        public static void forward_activation_layer_gpu(Layer l, NetworkState state)
        {
            Blas.copy_ongpu(l.Outputs * l.Batch, state.Input, l.OutputGpu);
            ActivationsHelper.activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }

        public static void backward_activation_layer_gpu(Layer l, NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            Blas.copy_ongpu(l.Outputs * l.Batch, l.DeltaGpu, state.Delta);
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

        public static void forward_avgpool_layer(Layer l, NetworkState state)
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

        public static void backward_avgpool_layer(Layer l, NetworkState state)
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

            l.Weights = new float[outputs * inputs];
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

        public static void update_connected_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            Blas.Axpy_cpu(l.Outputs, learning_rate / batch, l.BiasUpdates, l.Biases);
            Blas.Scal_cpu(l.Outputs, momentum, l.BiasUpdates, 1);

            if (l.BatchNormalize)
            {
                Blas.Axpy_cpu(l.Outputs, learning_rate / batch, l.ScaleUpdates, l.Scales);
                Blas.Scal_cpu(l.Outputs, momentum, l.ScaleUpdates, 1);
            }

            Blas.Axpy_cpu(l.Inputs * l.Outputs, -decay * batch, l.Weights, l.WeightUpdates);
            Blas.Axpy_cpu(l.Inputs * l.Outputs, learning_rate / batch, l.WeightUpdates, l.Weights);
            Blas.Scal_cpu(l.Inputs * l.Outputs, momentum, l.WeightUpdates, 1);
        }

        public static void forward_connected_layer(Layer l, NetworkState state)
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
                    Blas.Mean_cpu(l.Output, l.Batch, l.Outputs, 1, l.Mean);
                    Blas.Variance_cpu(l.Output, l.Mean, l.Batch, l.Outputs, 1, l.Variance);

                    Blas.Scal_cpu(l.Outputs, .95f, l.RollingMean, 1);
                    Blas.Axpy_cpu(l.Outputs, .05f, l.Mean, l.RollingMean);
                    Blas.Scal_cpu(l.Outputs, .95f, l.RollingVariance, 1);
                    Blas.Axpy_cpu(l.Outputs, .05f, l.Variance, l.RollingVariance);

                    Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, l.X);
                    Blas.Normalize_cpu(l.Output, l.Mean, l.Variance, l.Batch, l.Outputs, 1);
                    Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, l.XNorm);
                }
                else
                {
                    Blas.Normalize_cpu(l.Output, l.RollingMean, l.RollingVariance, l.Batch, l.Outputs, 1);
                }
                scale_bias(l.Output, l.Scales, l.Batch, l.Outputs, 1);
            }
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.Biases, l.Output, 0, i * l.Outputs);
            }
            ActivationsHelper.Activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        public static void backward_connected_layer(Layer l, NetworkState state)
        {
            int i;
            ActivationsHelper.Gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.Delta, l.BiasUpdates, i * l.Outputs);
            }
            if (l.BatchNormalize)
            {
                backward_scale_cpu(l.XNorm, l.Delta, l.Batch, l.Outputs, 1, l.ScaleUpdates);

                scale_bias(l.Delta, l.Scales, l.Batch, l.Outputs, 1);

                mean_delta_cpu(l.Delta, l.Variance, l.Batch, l.Outputs, 1, l.MeanDelta);
                variance_delta_cpu(l.X, l.Delta, l.Mean, l.Variance, l.Batch, l.Outputs, 1, l.VarianceDelta);
                normalize_delta_cpu(l.X, l.Mean, l.Variance, l.MeanDelta, l.VarianceDelta, l.Batch, l.Outputs, 1, l.Delta);
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

            if (c.Any()) Gemm.gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
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

        public static void update_connected_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            Blas.axpy_ongpu(l.Outputs, learning_rate / batch, l.BiasUpdatesGpu, l.BiasesGpu);
            Blas.scal_ongpu(l.Outputs, momentum, l.BiasUpdatesGpu, 1);

            if (l.BatchNormalize)
            {
                Blas.axpy_ongpu(l.Outputs, learning_rate / batch, l.ScaleUpdatesGpu, l.ScalesGpu);
                Blas.scal_ongpu(l.Outputs, momentum, l.ScaleUpdatesGpu, 1);
            }

            Blas.axpy_ongpu(l.Inputs * l.Outputs, -decay * batch, l.WeightsGpu, l.WeightUpdatesGpu);
            Blas.axpy_ongpu(l.Inputs * l.Outputs, learning_rate / batch, l.WeightUpdatesGpu, l.WeightsGpu);
            Blas.scal_ongpu(l.Inputs * l.Outputs, momentum, l.WeightUpdatesGpu, 1);
        }

        public static void forward_connected_layer_gpu(Layer l, NetworkState state)
        {
            int i;
            Blas.fill_ongpu(l.Outputs * l.Batch, 0, l.OutputGpu, 1);

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
                Blas.axpy_ongpu(l.Outputs, 1, l.BiasesGpu, l.OutputGpu, 0, i * l.Outputs);
            }
            ActivationsHelper.activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }

        public static void backward_connected_layer_gpu(Layer l, NetworkState state)
        {
            int i;
            Blas.constrain_ongpu(l.Outputs * l.Batch, 1, l.DeltaGpu, 1);
            ActivationsHelper.gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            for (i = 0; i < l.Batch; ++i)
            {
                Blas.axpy_ongpu(l.Outputs, 1, l.DeltaGpu, l.BiasUpdatesGpu, i * l.Outputs);
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

        public void resize_cost_layer(int inputs)
        {
            Inputs = inputs;
            Outputs = inputs;
            Array.Resize(ref Delta, inputs * Batch);
            Array.Resize(ref Output, inputs * Batch);

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

        }

        public static void forward_cost_layer(Layer l, NetworkState state)
        {
            if (!state.Truth.Any()) return;
            if (l.CostType == CostType.Masked)
            {
                int i;
                for (i = 0; i < l.Batch * l.Inputs; ++i)
                {
                    if (state.Truth[i] == Utils.SecretNum) state.Input[i] = Utils.SecretNum;
                }
            }
            if (l.CostType == CostType.Smooth)
            {
                Blas.Smooth_l1_cpu(l.Batch * l.Inputs, state.Input, state.Truth, l.Delta, l.Output);
            }
            else
            {
                Blas.L2_cpu(l.Batch * l.Inputs, state.Input, state.Truth, l.Delta, l.Output);
            }
            l.Cost = l.Output.Sum();
        }

        public static void backward_cost_layer(Layer l, NetworkState state)
        {
            Blas.Axpy_cpu(l.Batch * l.Inputs, l.Scale, l.Delta, state.Delta);
        }

        public static void pull_cost_layer(Layer l)
        {
            Array.Copy(l.DeltaGpu, l.Delta, l.Batch * l.Inputs);
        }

        public static void push_cost_layer(Layer l)
        {
            Array.Copy(l.Delta, l.DeltaGpu, l.Batch * l.Inputs);
        }

        public static void forward_cost_layer_gpu(Layer l, NetworkState state)
        {
            if (!state.Truth.Any()) return;
            if (l.CostType == CostType.Masked)
            {
                Blas.mask_ongpu(l.Batch * l.Inputs, state.Input, Utils.SecretNum, state.Truth);
            }

            if (l.CostType == CostType.Smooth)
            {
                Blas.smooth_l1_gpu(l.Batch * l.Inputs, state.Input, state.Truth, l.DeltaGpu, l.OutputGpu);
            }
            else
            {
                Blas.l2_gpu(l.Batch * l.Inputs, state.Input, state.Truth, l.DeltaGpu, l.OutputGpu);
            }

            if (l.Ratio != 0)
            {
                Array.Copy(l.DeltaGpu, l.Delta, l.Batch * l.Inputs);
                l.Delta = l.Delta.OrderBy(f => Math.Abs(f)).ToArray();
                int n = (int)((1 - l.Ratio) * l.Batch * l.Inputs);
                float thresh = l.Delta[n];
                thresh = 0;
                Console.Write($"{thresh}\n");
                Blas.supp_ongpu(l.Batch * l.Inputs, thresh, l.DeltaGpu, 1);
            }

            Array.Copy(l.OutputGpu, l.Output, l.Batch * l.Inputs);
            l.Cost = l.Output.Sum();
        }

        public static void backward_cost_layer_gpu(Layer l, NetworkState state)
        {
            Blas.axpy_ongpu(l.Batch * l.Inputs, l.Scale, l.DeltaGpu, state.Delta);
        }

        public static void increment_layer(Layer l, int steps)
        {
            int num = l.Outputs * l.Batch * steps;
            Utils.IncArray(ref l.Output, ref l.OutputBackup, l.OutputIndex, l.OutputIndex += num);
            Utils.IncArray(ref l.Delta, ref l.DeltaBackup, l.DeltaIndex, l.DeltaIndex += num);
            Utils.IncArray(ref l.X, ref l.XBackup, l.XIndex, l.XIndex += num);
            Utils.IncArray(ref l.XNorm, ref l.XNormBackup, l.XNormIndex, l.XNormIndex += num);

            Utils.IncArray(ref l.OutputGpu, ref l.OutputGpuBackup, l.OutputGpuIndex, l.OutputGpuIndex += num);
            Utils.IncArray(ref l.DeltaGpu, ref l.DeltaGpuBackup, l.DeltaGpuIndex, l.DeltaGpuIndex += num);
            Utils.IncArray(ref l.XGpu, ref l.XGpuBackup, l.XGpuIndex, l.XGpuIndex += num);
            Utils.IncArray(ref l.XNormGpu, ref l.XNormGpuBackup, l.XNormGpuIndex, l.XNormGpuIndex += num);
        }

        public static Layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, Activation activation, bool batch_normalize)
        {
            Console.Error.Write($"LayerType.Crnn Layer: {h} x {w} x {c} Image, {output_filters} filters\n");
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
            (l.InputLayer) = make_convolutional_layer(batch * steps, h, w, c, hidden_filters, 3, 1, 1, activation, batch_normalize, false, false, false);
            l.InputLayer.Batch = batch;

            l.InputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputLayer) = make_convolutional_layer(batch * steps, h, w, hidden_filters, hidden_filters, 3, 1, 1, activation, batch_normalize, false, false, false);
            l.InputLayer.Batch = batch;

            l.OutputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.OutputLayer) = make_convolutional_layer(batch * steps, h, w, hidden_filters, output_filters, 3, 1, 1, activation, batch_normalize, false, false, false);
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

        public static void update_crnn_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_convolutional_layer(l.InputLayer, batch, learning_rate, momentum, decay);
            update_convolutional_layer(l.InputLayer, batch, learning_rate, momentum, decay);
            update_convolutional_layer(l.OutputLayer, batch, learning_rate, momentum, decay);
        }

        public static void forward_crnn_layer(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.InputLayer);
            Layer output_layer = (l.OutputLayer);

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
                if (state.Train)
                {
                    Utils.IncArray(ref l.State, ref l.StateBackup, l.StateIndex, l.StateIndex += l.Hidden * l.Batch);
                }
                if (l.Shortcut)
                {
                    Blas.Copy_cpu(l.Hidden * l.Batch, old_state, l.State);
                }
                else
                {
                    Blas.Fill_cpu(l.Hidden * l.Batch, 0, l.State, 1);
                }
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, input_layer.Output, l.State);
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output, l.State);

                s.Input = l.State;
                forward_convolutional_layer(output_layer, s);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch);
                increment_layer(input_layer, 1);
                increment_layer(self_layer, 1);
                increment_layer(output_layer, 1);
            }
        }

        public static void backward_crnn_layer(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.InputLayer);
            Layer output_layer = l.OutputLayer;

            increment_layer(input_layer, l.Steps - 1);
            increment_layer(self_layer, l.Steps - 1);
            increment_layer(output_layer, l.Steps - 1);

            Utils.IncArray(ref l.State, ref l.StateBackup, l.StateIndex, l.StateIndex += l.Hidden * l.Batch * l.Steps);
            for (i = l.Steps - 1; i >= 0; --i)
            {
                Blas.Copy_cpu(l.Hidden * l.Batch, input_layer.Output, l.State);
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output, l.State);

                s.Input = l.State;
                s.Delta = self_layer.Delta;
                backward_convolutional_layer(output_layer, s);

                Utils.DecArray(ref l.State, ref l.StateBackup, l.StateIndex, l.StateIndex -= l.Hidden * l.Batch);

                s.Input = l.State;
                if (i == 0)
                {
                    s.Delta = new float[0];
                }
                else
                {

                }
                backward_convolutional_layer(self_layer, s);

                Blas.Copy_cpu(l.Hidden * l.Batch, self_layer.Delta, input_layer.Delta);
                var tempFloats = new float[self_layer.Delta.Length - l.Hidden * l.Batch];
                Array.Copy(self_layer.DeltaBackup, self_layer.DeltaIndex - l.Hidden * l.Batch, tempFloats, 0, tempFloats.Length);
                if (i > 0 && l.Shortcut) Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Delta, tempFloats);
                Array.Copy(tempFloats, 0, self_layer.DeltaBackup, self_layer.DeltaIndex - l.Hidden * l.Batch, tempFloats.Length);
                Array.Copy(state.Input, i * l.Inputs * l.Batch, s.Input, 0, state.Input.Length);
                if (state.Delta.Length != 0)
                {
                    Array.Copy(state.Delta, i * l.Inputs * l.Batch, s.Delta, 0, state.Delta.Length);
                }
                else s.Delta = new float[0];
                backward_convolutional_layer(input_layer, s);

                increment_layer(input_layer, -1);
                increment_layer(self_layer, -1);
                increment_layer(output_layer, -1);
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

        public static void update_crnn_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_convolutional_layer_gpu((l.InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer_gpu((l.InputLayer), batch, learning_rate, momentum, decay);
            update_convolutional_layer_gpu((l.OutputLayer), batch, learning_rate, momentum, decay);
        }

        public static void forward_crnn_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.InputLayer);
            Layer output_layer = (l.OutputLayer);

            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, output_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Hidden * l.Batch * l.Steps, 0, self_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Hidden * l.Batch * l.Steps, 0, input_layer.DeltaGpu, 1);
            if (state.Train) Blas.fill_ongpu(l.Hidden * l.Batch, 0, l.StateGpu, 1);

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = state.Input;
                forward_convolutional_layer_gpu(input_layer, s);

                s.Input = l.StateGpu;
                forward_convolutional_layer_gpu(self_layer, s);

                float[] old_state = l.StateGpu;
                if (state.Train)
                {
                    Utils.IncArray(ref l.StateGpu, ref l.StateGpuBackup, l.StateGpuIndex, l.StateGpuIndex += l.Hidden * l.Batch);
                }
                if (l.Shortcut)
                {
                    Blas.copy_ongpu(l.Hidden * l.Batch, old_state, l.StateGpu);
                }
                else
                {
                    Blas.fill_ongpu(l.Hidden * l.Batch, 0, l.StateGpu, 1);
                }
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, input_layer.OutputGpu, l.StateGpu);
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.OutputGpu, l.StateGpu);

                s.Input = l.StateGpu;
                forward_convolutional_layer_gpu(output_layer, s);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch);
                increment_layer(input_layer, 1);
                increment_layer(self_layer, 1);
                increment_layer(output_layer, 1);
            }
        }

        public static void backward_crnn_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.InputLayer);
            Layer output_layer = (l.OutputLayer);
            increment_layer(input_layer, l.Steps - 1);
            increment_layer(self_layer, l.Steps - 1);
            increment_layer(output_layer, l.Steps - 1);
            Utils.IncArray(ref l.StateGpu, ref l.StateGpuBackup, l.StateGpuIndex, l.StateGpuIndex += l.Hidden * l.Batch * l.Steps);
            for (i = l.Steps - 1; i >= 0; --i)
            {
                Blas.copy_ongpu(l.Hidden * l.Batch, input_layer.OutputGpu, l.StateGpu);
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.OutputGpu, l.StateGpu);

                s.Input = l.StateGpu;
                s.Delta = self_layer.DeltaGpu;
                backward_convolutional_layer_gpu(output_layer, s);

                Utils.DecArray(ref l.StateGpu, ref l.StateGpuBackup, l.StateGpuIndex, l.StateGpuIndex -= l.Hidden * l.Batch);

                s.Input = l.StateGpu;
                Array.Copy(self_layer.DeltaGpuBackup, self_layer.DeltaGpuIndex - l.Hidden * l.Batch, s.Delta, 0, self_layer.DeltaGpu.Length + l.Hidden * l.Batch);
                if (i == 0) s.Delta = new float[0];
                backward_convolutional_layer_gpu(self_layer, s);

                Blas.copy_ongpu(l.Hidden * l.Batch, self_layer.DeltaGpu, input_layer.DeltaGpu);

                if (i > 0 && l.Shortcut)
                {
                    var tempFloat = new float[self_layer.DeltaGpu.Length + l.Hidden * l.Batch];
                    Array.Copy(self_layer.DeltaGpuBackup, self_layer.DeltaGpuIndex + l.Hidden * l.Batch, tempFloat, 0, tempFloat.Length);
                    Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.DeltaGpu, tempFloat);
                    Array.Copy(tempFloat, tempFloat.Length, self_layer.DeltaGpuBackup, self_layer.DeltaGpu.Length + l.Hidden * l.Batch, tempFloat.Length);
                }
                Array.Copy(state.Input, state.Input.Length + i * l.Inputs * l.Batch, s.Input, 0, i * l.Inputs * l.Batch);
                if (state.Delta.Length != 0)
                {
                    Array.Copy(state.Delta, state.Delta.Length + i * l.Inputs * l.Batch, s.Delta, 0, i * l.Inputs * l.Batch);
                }
                else s.Delta = new float[0];
                backward_convolutional_layer_gpu(input_layer, s);

                increment_layer(input_layer, -1);
                increment_layer(self_layer, -1);
                increment_layer(output_layer, -1);
            }
        }

        public Image get_crop_image()
        {
            int h = OutH;
            int w = OutW;
            int c = OutC;
            return new Image(w, h, c, Output);
        }

        public static void backward_crop_layer(Layer l, NetworkState state) { }

        public static void backward_crop_layer_gpu(Layer l, NetworkState state) { }

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

        public static void forward_crop_layer(Layer l, NetworkState state)
        {
            int i, j, c, b, row, col;
            int index;
            int count = 0;
            bool flip = (l.Flip && Utils.Rand.Next() % 2 != 0);
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
                flip = false;
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
            return deconvolutional_out_height() * deconvolutional_out_width();
        }

        public Image get_deconvolutional_image()
        {
            int h, w, c;
            h = deconvolutional_out_height();
            w = deconvolutional_out_width();
            c = N;
            return new Image(w, h, c, Output);
        }

        public Image get_deconvolutional_delta()
        {
            int h, w, c;
            h = deconvolutional_out_height();
            w = deconvolutional_out_width();
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
            int out_h = l.deconvolutional_out_height();
            int out_w = l.deconvolutional_out_width();

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
            int out_h = deconvolutional_out_height();
            int out_w = deconvolutional_out_width();

            Array.Resize(ref ColImage, out_h * out_w * Size * Size * C);
            Array.Resize(ref Output, Batch * out_h * out_w * N);
            Array.Resize(ref Delta, Batch * out_h * out_w * N);

            ColImageGpu = (float[])ColImage.Clone();
            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

        }

        public static void forward_deconvolutional_layer(Layer l, NetworkState state)
        {
            int i;
            int out_h = l.deconvolutional_out_height();
            int out_w = l.deconvolutional_out_width();
            int size = out_h * out_w;

            int m = l.Size * l.Size * l.N;
            int n = l.H * l.W;
            int k = l.C;

            Blas.Fill_cpu(l.Outputs * l.Batch, 0, l.Output, 1);

            for (i = 0; i < l.Batch; ++i)
            {
                float[] a = l.Weights;
                float[] b = new float[state.Input.Length + i * l.C * l.H * l.W];
                Array.Copy(state.Input, i * l.C * l.H * l.W, b, 0, b.Length);
                float[] c = l.ColImage;

                Gemm.gemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

                Im2Col.col2im_cpu(c, l.N, out_h, out_w, l.Size, l.Stride, 0, l.Output, i * l.N * size);
            }
            add_bias(l.Output, l.Biases, l.Batch, l.N, size);
            ActivationsHelper.Activate_array(l.Output, l.Batch * l.N * size, l.Activation);
        }

        public static void backward_deconvolutional_layer(Layer l, NetworkState state)
        {
            float alpha = 1.0f / l.Batch;
            int out_h = l.deconvolutional_out_height();
            int out_w = l.deconvolutional_out_width();
            int size = out_h * out_w;
            int i;

            ActivationsHelper.Gradient_array(l.Output, size * l.N * l.Batch, l.Activation, l.Delta);
            backward_bias(l.BiasUpdates, l.Delta, l.Batch, l.N, size);

            for (i = 0; i < l.Batch; ++i)
            {
                int m = l.C;
                int n = l.Size * l.Size * l.N;
                int k = l.H * l.W;

                float[] a = new float[state.Input.Length - i * m * n];
                Array.Copy(state.Input, i * m * n, a, 0, a.Length);
                float[] b = l.ColImage;
                float[] c = l.WeightUpdates;

                Im2Col.im2col_cpu(l.Delta, l.N, out_h, out_w,
                    l.Size, l.Stride, 0, b, i * l.N * size);
                Gemm.gemm(0, 1, m, n, k, alpha, a, k, b, k, 1, c, n);

                if (state.Delta.Length != 0)
                {
                    int m2 = l.C;
                    int n2 = l.H * l.W;
                    int k2 = l.Size * l.Size * l.N;

                    float[] a2 = l.Weights;
                    float[] b2 = l.ColImage;
                    float[] c2 = new float[state.Delta.Length - i * n2 * m2];
                    Array.Copy(state.Delta, i * n2 * m2, c2, 0, c2.Length);
                    Gemm.gemm(0, 0, m2, n2, k2, 1, a2, k2, b2, n2, 1, c2, n2);
                    Array.Copy(c2, 0, state.Delta, i * n2 * m2, c2.Length);
                }
            }
        }

        public static void update_deconvolutional_layer(Layer l, int n, float learning_rate, float momentum, float decay)
        {
            int size = l.Size * l.Size * l.C * l.N;
            Blas.Axpy_cpu(l.N, learning_rate, l.BiasUpdates, l.Biases);
            Blas.Scal_cpu(l.N, momentum, l.BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay, l.Weights, l.WeightUpdates);
            Blas.Axpy_cpu(size, learning_rate, l.WeightUpdates, l.Weights);
            Blas.Scal_cpu(size, momentum, l.WeightUpdates, 1);
        }

        public static Layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, bool rescore)
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

        public static void forward_detection_layer(Layer l, NetworkState state)
        {
            int locations = l.Side * l.Side;
            int i, j;
            Array.Copy(state.Input, 0, l.Output, 0, l.Outputs * l.Batch);
            int b;
            if (l.Softmax)
            {
                for (b = 0; b < l.Batch; ++b)
                {
                    int index = b * l.Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int offset = i * l.Classes;
                        Blas.Softmax(l.Output, l.Classes, 1,
                            l.Output, index + offset, index + offset);
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
                (l.Cost) = null;
                int size = l.Inputs * l.Batch;
                l.Delta = new float[size];
                for (b = 0; b < l.Batch; ++b)
                {
                    int index = b * l.Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int truth_index = (b * locations + i) * (1 + l.Coords + l.Classes);
                        bool is_obj = state.Truth[truth_index] != 0;
                        for (j = 0; j < l.N; ++j)
                        {
                            int pn_index = index + locations * l.Classes + i * l.N + j;
                            l.Delta[pn_index] = l.NoobjectScale * (0 - l.Output[pn_index]);
                            (l.Cost) += l.NoobjectScale * (float)Math.Pow(l.Output[pn_index], 2);
                            avg_anyobj += l.Output[pn_index];
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
                            l.Delta[class_index + j] = l.ClassScale * (state.Truth[truth_index + 1 + j] - l.Output[class_index + j]);
                            (l.Cost) += l.ClassScale * (float)Math.Pow(state.Truth[truth_index + 1 + j] - l.Output[class_index + j], 2);
                            if (state.Truth[truth_index + 1 + j] != 0) avg_cat += l.Output[class_index + j];
                            avg_allcat += l.Output[class_index + j];
                        }

                        Box truth = new Box(state.Truth, truth_index + 1 + l.Classes);
                        truth.X /= l.Side;
                        truth.Y /= l.Side;

                        for (j = 0; j < l.N; ++j)
                        {
                            int box_index = index + locations * (l.Classes + l.N) + (i * l.N + j) * l.Coords;
                            Box outputout = new Box(l.Output, box_index);
                            outputout.X /= l.Side;
                            outputout.Y /= l.Side;

                            if (l.Sqrt)
                            {
                                outputout.W = outputout.W * outputout.W;
                                outputout.H = outputout.H * outputout.H;
                            }

                            float iou = Box.box_iou(outputout, truth);
                            float rmse = Box.box_rmse(outputout, truth);
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

                        if (l.Forced != 0)
                        {
                            best_index = truth.W * truth.H < .1 ? 1 : 0;
                        }
                        if (l.Random != 0 && state.Net.Seen < 64000)
                        {
                            best_index = Utils.Rand.Next() % l.N;
                        }

                        int box_index2 = index + locations * (l.Classes + l.N) + (i * l.N + best_index) * l.Coords;
                        int tbox_index = truth_index + 1 + l.Classes;

                        Box outputout2 = new Box(l.Output, box_index2);
                        outputout2.X /= l.Side;
                        outputout2.Y /= l.Side;
                        if (l.Sqrt)
                        {
                            outputout2.W = outputout2.W * outputout2.W;
                            outputout2.H = outputout2.H * outputout2.H;
                        }
                        float iou2 = Box.box_iou(outputout2, truth);

                        int p_index = index + locations * l.Classes + i * l.N + best_index;
                        (l.Cost) -= l.NoobjectScale * (float)Math.Pow(l.Output[p_index], 2);
                        (l.Cost) += l.ObjectScale * (float)Math.Pow(1 - l.Output[p_index], 2);
                        avg_obj += l.Output[p_index];
                        l.Delta[p_index] = l.ObjectScale * (1.0f - l.Output[p_index]);

                        if (l.Rescore)
                        {
                            l.Delta[p_index] = l.ObjectScale * (iou2 - l.Output[p_index]);
                        }

                        l.Delta[box_index2 + 0] = l.CoordScale * (state.Truth[tbox_index + 0] - l.Output[box_index2 + 0]);
                        l.Delta[box_index2 + 1] = l.CoordScale * (state.Truth[tbox_index + 1] - l.Output[box_index2 + 1]);
                        l.Delta[box_index2 + 2] = l.CoordScale * (state.Truth[tbox_index + 2] - l.Output[box_index2 + 2]);
                        l.Delta[box_index2 + 3] = l.CoordScale * (state.Truth[tbox_index + 3] - l.Output[box_index2 + 3]);
                        if (l.Sqrt)
                        {
                            l.Delta[box_index2 + 2] = l.CoordScale * ((float)Math.Sqrt(state.Truth[tbox_index + 2]) - l.Output[box_index2 + 2]);
                            l.Delta[box_index2 + 3] = l.CoordScale * ((float)Math.Sqrt(state.Truth[tbox_index + 3]) - l.Output[box_index2 + 3]);
                        }

                        (l.Cost) += (float)Math.Pow(1 - iou2, 2);
                        avg_iou += iou2;
                        ++count;
                    }
                }
                (l.Cost) = (float)Math.Pow(Utils.mag_array(l.Delta, l.Outputs * l.Batch), 2);


                Console.Write(
                    $"Detection Avg IOU: {avg_iou / count}, Pos Cat: {avg_cat / count}, All Cat: {avg_allcat / (count * l.Classes)}, Pos Obj: {avg_obj / count}, Any Obj: {avg_anyobj / (l.Batch * locations * l.N)}, count: {count}\n");
            }
        }

        public static void backward_detection_layer(Layer l, NetworkState state)
        {
            Blas.Axpy_cpu(l.Batch * l.Inputs, 1, l.Delta, state.Delta);
        }

        public void get_detection_boxes(int w, int h, float thresh, float[][] probs, Box[] boxes, bool only_objectness)
        {
            int i, j, n;
            float[] predictions = Output;
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
                    boxes[index].W = (float)Math.Pow(predictions[box_index + 2], (Sqrt ? 2 : 1)) * w;
                    boxes[index].H = (float)Math.Pow(predictions[box_index + 3], (Sqrt ? 2 : 1)) * h;
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

        public static void forward_detection_layer_gpu(Layer l, NetworkState state)
        {
            if (!state.Train)
            {
                Blas.copy_ongpu(l.Batch * l.Inputs, state.Input, l.OutputGpu);
                return;
            }

            float[] in_cpu = new float[l.Batch * l.Inputs];
            float[] truth_cpu = null;
            if (state.Truth.Any())
            {
                int num_truth = l.Batch * l.Side * l.Side * (1 + l.Coords + l.Classes);
                truth_cpu = new float[num_truth];
                Array.Copy(state.Truth, truth_cpu, num_truth);
            }
            Array.Copy(state.Input, in_cpu, l.Batch * l.Inputs);
            NetworkState cpu_state = state;
            cpu_state.Train = state.Train;
            cpu_state.Truth = truth_cpu;
            cpu_state.Input = in_cpu;
            forward_detection_layer(l, cpu_state);
            Array.Copy(l.Output, l.OutputGpu, l.Batch * l.Outputs);
            Array.Copy(l.Delta, l.DeltaGpu, l.Batch * l.Inputs);
        }

        public static void backward_detection_layer_gpu(Layer l, NetworkState state)
        {
            Blas.axpy_ongpu(l.Batch * l.Inputs, 1, l.DeltaGpu, state.Delta);
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

        public static void forward_dropout_layer(Layer l, NetworkState state)
        {
            int i;
            if (!state.Train) return;
            for (i = 0; i < l.Batch * l.Inputs; ++i)
            {
                float r = Utils.rand_uniform(0, 1);
                l.Rand[i] = r;
                if (r < l.Probability) state.Input[i] = 0;
                else state.Input[i] *= l.Scale;
            }
        }

        public static void backward_dropout_layer(Layer l, NetworkState state)
        {
            int i;
            if (state.Delta.Length == 0) return;
            for (i = 0; i < l.Batch * l.Inputs; ++i)
            {
                float r = l.Rand[i];
                if (r < l.Probability) state.Delta[i] = 0;
                else state.Delta[i] *= l.Scale;
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

        public static void update_gru_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_connected_layer(l.InputLayer, batch, learning_rate, momentum, decay);
            update_connected_layer(l.InputLayer, batch, learning_rate, momentum, decay);
            update_connected_layer(l.OutputLayer, batch, learning_rate, momentum, decay);
        }

        public static void forward_gru_layer(Layer l, NetworkState state)
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
                Blas.Copy_cpu(l.Outputs * l.Batch, l.State, l.PrevState);
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


                Blas.Copy_cpu(l.Outputs * l.Batch, input_z_layer.Output, l.ZCpu);
                Blas.Axpy_cpu(l.Outputs * l.Batch, 1, state_z_layer.Output, l.ZCpu);

                Blas.Copy_cpu(l.Outputs * l.Batch, input_r_layer.Output, l.RCpu);
                Blas.Axpy_cpu(l.Outputs * l.Batch, 1, state_r_layer.Output, l.RCpu);

                ActivationsHelper.Activate_array(l.ZCpu, l.Outputs * l.Batch, Activation.Logistic);
                ActivationsHelper.Activate_array(l.RCpu, l.Outputs * l.Batch, Activation.Logistic);

                Blas.Copy_cpu(l.Outputs * l.Batch, l.State, l.ForgotState);
                Blas.Mul_cpu(l.Outputs * l.Batch, l.RCpu, 1, l.ForgotState, 1);

                s.Input = l.ForgotState;
                forward_connected_layer(state_h_layer, s);

                Blas.Copy_cpu(l.Outputs * l.Batch, input_h_layer.Output, l.HCpu);
                Blas.Axpy_cpu(l.Outputs * l.Batch, 1, state_h_layer.Output, l.HCpu);

                // USET ActivationsHelper.Activate_array(HCpu, Outputs * Batch, TANH);
                ActivationsHelper.Activate_array(l.HCpu, l.Outputs * l.Batch, Activation.Logistic);


                Blas.Weighted_sum_cpu(l.State, l.HCpu, l.ZCpu, l.Outputs * l.Batch, l.Output);

                Blas.Copy_cpu(l.Outputs * l.Batch, l.Output, l.State);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch);
                Utils.IncArray(ref l.Output, ref l.OutputBackup, l.OutputIndex, l.OutputIndex += l.Outputs * l.Batch);
                increment_layer(input_z_layer, 1);
                increment_layer(input_r_layer, 1);
                increment_layer(input_h_layer, 1);

                increment_layer(state_z_layer, 1);
                increment_layer(state_r_layer, 1);
                increment_layer(state_h_layer, 1);
            }
        }

        public static void backward_gru_layer(Layer l, NetworkState state)
        {
        }

        public static void pull_gru_layer()
        {
        }

        public static void push_gru_layer()
        {
        }

        public static void update_gru_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_connected_layer_gpu(l.InputRLayer, batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(l.InputZLayer, batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(l.InputHLayer, batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(l.StateRLayer, batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(l.StateZLayer, batch, learning_rate, momentum, decay);
            update_connected_layer_gpu(l.StateHLayer, batch, learning_rate, momentum, decay);
        }

        public static void forward_gru_layer_gpu(Layer l, NetworkState state)
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

            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, input_z_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, input_r_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, input_h_layer.DeltaGpu, 1);

            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, state_z_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, state_r_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, state_h_layer.DeltaGpu, 1);
            if (state.Train)
            {
                Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, l.DeltaGpu, 1);
                Blas.copy_ongpu(l.Outputs * l.Batch, l.StateGpu, l.PrevStateGpu);
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


                Blas.copy_ongpu(l.Outputs * l.Batch, input_z_layer.OutputGpu, l.ZGpu);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_z_layer.OutputGpu, l.ZGpu);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_r_layer.OutputGpu, l.RGpu);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_r_layer.OutputGpu, l.RGpu);

                ActivationsHelper.activate_array_ongpu(l.ZGpu, l.Outputs * l.Batch, Activation.Logistic);
                ActivationsHelper.activate_array_ongpu(l.RGpu, l.Outputs * l.Batch, Activation.Logistic);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.StateGpu, l.ForgotStateGpu);
                Blas.mul_ongpu(l.Outputs * l.Batch, l.RGpu, 1, l.ForgotStateGpu, 1);

                s.Input = l.ForgotStateGpu;
                forward_connected_layer_gpu(state_h_layer, s);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_h_layer.OutputGpu, l.HGpu);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_h_layer.OutputGpu, l.HGpu);

                // USET ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, TANH);
                ActivationsHelper.activate_array_ongpu(l.HGpu, l.Outputs * l.Batch, Activation.Logistic);


                Blas.weighted_sum_gpu(l.StateGpu, l.HGpu, l.ZGpu, l.Outputs * l.Batch, l.OutputGpu);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.OutputGpu, l.StateGpu);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch);
                Utils.IncArray(ref l.OutputGpu, ref l.OutputGpuBackup, l.OutputGpuIndex, l.OutputGpuIndex += l.Outputs * l.Batch);
                increment_layer(input_z_layer, 1);
                increment_layer(input_r_layer, 1);
                increment_layer(input_h_layer, 1);

                increment_layer(state_z_layer, 1);
                increment_layer(state_r_layer, 1);
                increment_layer(state_h_layer, 1);
            }
        }

        public static void backward_gru_layer_gpu(Layer l, NetworkState state)
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

            increment_layer(input_z_layer, l.Steps - 1);
            increment_layer(input_r_layer, l.Steps - 1);
            increment_layer(input_h_layer, l.Steps - 1);

            increment_layer(state_z_layer, l.Steps - 1);
            increment_layer(state_r_layer, l.Steps - 1);
            increment_layer(state_h_layer, l.Steps - 1);

            Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch * (l.Steps - 1));
            if (state.Delta.Length != 0)
            {
                Utils.IncArray(ref state.Delta, ref state.DeltaBackup, state.DeltaIndex, state.DeltaIndex += l.Inputs * l.Batch * (l.Steps - 1));
            }
            Utils.IncArray(ref l.OutputGpu, ref l.OutputGpuBackup, l.OutputGpuIndex, l.OutputGpuIndex += l.Outputs * l.Batch * (l.Steps - 1));
            Utils.IncArray(ref l.DeltaGpu, ref l.DeltaGpuBackup, l.DeltaGpuIndex, l.DeltaGpuIndex += l.Outputs * l.Batch * (l.Steps - 1));
            for (i = l.Steps - 1; i >= 0; --i)
            {
                if (i != 0)
                {
                    Blas.copy_ongpu(l.Outputs * l.Batch, l.OutputGpuBackup, l.PrevStateGpu, l.OutputGpuIndex - l.Outputs * l.Batch);
                }
                float[] prev_delta_gpu;
                if (i == 0)
                {
                    prev_delta_gpu = new float[0];
                }
                else
                {
                    prev_delta_gpu = new float[l.DeltaGpu.Length + l.Outputs * l.Batch];
                    Array.Copy(l.DeltaGpuBackup, l.DeltaGpuIndex - l.Outputs * l.Batch, prev_delta_gpu, 0, l.Outputs * l.Batch);
                    Array.Copy(l.DeltaGpu, 0, prev_delta_gpu, l.Outputs * l.Batch, l.DeltaGpu.Length);
                }

                Blas.copy_ongpu(l.Outputs * l.Batch, input_z_layer.OutputGpu, l.ZGpu);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_z_layer.OutputGpu, l.ZGpu);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_r_layer.OutputGpu, l.RGpu);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_r_layer.OutputGpu, l.RGpu);

                ActivationsHelper.activate_array_ongpu(l.ZGpu, l.Outputs * l.Batch, Activation.Logistic);
                ActivationsHelper.activate_array_ongpu(l.RGpu, l.Outputs * l.Batch, Activation.Logistic);

                Blas.copy_ongpu(l.Outputs * l.Batch, input_h_layer.OutputGpu, l.HGpu);
                Blas.axpy_ongpu(l.Outputs * l.Batch, 1, state_h_layer.OutputGpu, l.HGpu);

                // USET ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, TANH);
                ActivationsHelper.activate_array_ongpu(l.HGpu, l.Outputs * l.Batch, Activation.Logistic);


                Blas.weighted_delta_gpu(l.PrevStateGpu, l.HGpu, l.ZGpu, prev_delta_gpu, input_h_layer.DeltaGpu, input_z_layer.DeltaGpu, l.Outputs * l.Batch, l.DeltaGpu);

                // USET ActivationsHelper.gradient_array_ongpu(HGpu, Outputs * Batch, TANH, input_h_layer.DeltaGpu);
                ActivationsHelper.gradient_array_ongpu(l.HGpu, l.Outputs * l.Batch, Activation.Logistic, input_h_layer.DeltaGpu);


                Blas.copy_ongpu(l.Outputs * l.Batch, input_h_layer.DeltaGpu, state_h_layer.DeltaGpu);

                Blas.copy_ongpu(l.Outputs * l.Batch, l.PrevStateGpu, l.ForgotStateGpu);
                Blas.mul_ongpu(l.Outputs * l.Batch, l.RGpu, 1, l.ForgotStateGpu, 1);
                Blas.fill_ongpu(l.Outputs * l.Batch, 0, l.ForgotDeltaGpu, 1);

                s.Input = l.ForgotStateGpu;
                s.Delta = l.ForgotDeltaGpu;

                backward_connected_layer_gpu(state_h_layer, s);
                if (prev_delta_gpu.Length != 0)
                {
                    Blas.mult_add_into_gpu(l.Outputs * l.Batch, l.ForgotDeltaGpu, l.RGpu, prev_delta_gpu);
                }
                Blas.mult_add_into_gpu(l.Outputs * l.Batch, l.ForgotDeltaGpu, l.PrevStateGpu, input_r_layer.DeltaGpu);

                ActivationsHelper.gradient_array_ongpu(l.RGpu, l.Outputs * l.Batch, Activation.Logistic, input_r_layer.DeltaGpu);
                Blas.copy_ongpu(l.Outputs * l.Batch, input_r_layer.DeltaGpu, state_r_layer.DeltaGpu);

                ActivationsHelper.gradient_array_ongpu(l.ZGpu, l.Outputs * l.Batch, Activation.Logistic, input_z_layer.DeltaGpu);
                Blas.copy_ongpu(l.Outputs * l.Batch, input_z_layer.DeltaGpu, state_z_layer.DeltaGpu);

                s.Input = l.PrevStateGpu;
                s.Delta = prev_delta_gpu;

                backward_connected_layer_gpu(state_r_layer, s);
                backward_connected_layer_gpu(state_z_layer, s);

                s.Input = state.Input;
                s.Delta = state.Delta;

                backward_connected_layer_gpu(input_h_layer, s);
                backward_connected_layer_gpu(input_r_layer, s);
                backward_connected_layer_gpu(input_z_layer, s);

                Utils.DecArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex -= l.Inputs * l.Batch);
                if (state.Delta.Length != 0)
                {
                    Utils.DecArray(ref state.Delta, ref state.DeltaBackup, state.DeltaIndex, state.DeltaIndex -= l.Inputs * l.Batch);
                }
                Utils.DecArray(ref l.OutputGpu, ref l.OutputGpuBackup, l.OutputGpuIndex, l.OutputGpuIndex -= l.Outputs * l.Batch);
                Utils.DecArray(ref l.DeltaGpu, ref l.DeltaGpuBackup, l.DeltaGpuIndex, l.DeltaGpuIndex -= l.Outputs * l.Batch);
                increment_layer(input_z_layer, -1);
                increment_layer(input_r_layer, -1);
                increment_layer(input_h_layer, -1);

                increment_layer(state_z_layer, -1);
                increment_layer(state_r_layer, -1);
                increment_layer(state_h_layer, -1);
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

            int out_h = l.local_out_height();
            int out_w = l.local_out_width();
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
            l.IndexesGpu = new int[output_size];
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

            IndexesGpu = new int[output_size];
            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();

        }

        public static void forward_maxpool_layer(Layer l, NetworkState state)
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
                                    bool valid = (cur_h >= 0 && cur_h < l.H &&
                                                 cur_w >= 0 && cur_w < l.W);
                                    float val = (valid) ? state.Input[index] : float.MinValue;
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

        public static void backward_maxpool_layer(Layer l, NetworkState state)
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

        public static Layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
        {
            Console.Error.Write($"Local Response Normalization Layer: %d x %d x %d Image, %d size\n", w, h, c, size);
            Layer l = new Layer();
            l.LayerType = LayerType.Normalization;
            l.Batch = batch;
            l.H = l.OutH = h;
            l.W = l.OutW = w;
            l.C = l.OutC = c;
            l.Kappa = kappa;
            l.Size = size;
            l.Alpha = alpha;
            l.Beta = beta;
            l.Output = new float[h * w * c * batch];
            l.Delta = new float[h * w * c * batch];
            l.Squared = new float[h * w * c * batch];
            l.Norms = new float[h * w * c * batch];
            l.Inputs = w * h * c;
            l.Outputs = l.Inputs;

            l.Forward = forward_normalization_layer;
            l.Backward = backward_normalization_layer;

            l.ForwardGpu = forward_normalization_layer_gpu;
            l.BackwardGpu = backward_normalization_layer_gpu;

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            l.SquaredGpu = (float[])l.Squared.Clone();
            l.NormsGpu = (float[])l.Norms.Clone();

            return l;
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

        public static void forward_normalization_layer(Layer l, NetworkState state)
        {
            int k, b;
            int w = l.W;
            int h = l.H;
            int c = l.C;
            Blas.Scal_cpu(w * h * c * l.Batch, 0, l.Squared, 1);

            for (b = 0; b < l.Batch; ++b)
            {
                var index = w * h * c * b;
                float[] squared = new float[l.Squared.Length - index];
                float[] norms = new float[l.Norms.Length - index];
                float[] input = new float[state.Input.Length - index];
                Array.Copy(l.Squared, index, squared, 0, squared.Length);
                Array.Copy(l.Norms, index, norms, 0, norms.Length);
                Array.Copy(state.Input, index, input, 0, input.Length);
                Blas.Pow_cpu(w * h * c, 2, input, squared);

                Blas.Const_cpu(w * h, l.Kappa, norms, 1);
                for (k = 0; k < l.Size / 2; ++k)
                {
                    Blas.Axpy_cpu(w * h, l.Alpha, squared, norms, 0, w * h * k);
                }

                for (k = 1; k < l.C; ++k)
                {
                    Blas.Copy_cpu(w * h, norms, norms, w * h * (k - 1), w * h * k);
                    int prev = k - ((l.Size - 1) / 2) - 1;
                    int next = k + (l.Size / 2);
                    if (prev >= 0) Blas.Axpy_cpu(w * h, -l.Alpha, squared, norms, w * h * prev, w * h * k);
                    if (next < l.C) Blas.Axpy_cpu(w * h, l.Alpha, squared, norms, w * h * next, w * h * k);
                }
                Array.Copy(squared, 0, l.Squared, index, squared.Length);
                Array.Copy(norms, 0, l.Norms, index, norms.Length);
            }
            Blas.Pow_cpu(w * h * c * l.Batch, -l.Beta, l.Norms, l.Output);
            Blas.Mul_cpu(w * h * c * l.Batch, state.Input, 1, l.Output, 1);
        }

        public static void backward_normalization_layer(Layer l, NetworkState state)
        {
            // TODO This is approximate ;-)
            // Also this should add in to delta instead of overwritting.

            int w = l.W;
            int h = l.H;
            int c = l.C;
            Blas.Pow_cpu(w * h * c * l.Batch, -l.Beta, l.Norms, state.Delta);
            Blas.Mul_cpu(w * h * c * l.Batch, l.Delta, 1, state.Delta, 1);
        }

        public static void forward_normalization_layer_gpu(Layer l, NetworkState state)
        {
            int k, b;
            int w = l.W;
            int h = l.H;
            int c = l.C;
            Blas.scal_ongpu(w * h * c * l.Batch, 0, l.SquaredGpu, 1);

            for (b = 0; b < l.Batch; ++b)
            {
                var index = w * h * c * b;
                float[] squared = new float[l.SquaredGpu.Length - index];
                float[] norms = new float[l.NormsGpu.Length - index];
                float[] input = new float[state.Input.Length - index];
                Array.Copy(l.Squared, index, squared, 0, squared.Length);
                Array.Copy(l.Norms, index, norms, 0, norms.Length);
                Array.Copy(state.Input, index, input, 0, input.Length);

                Blas.pow_ongpu(w * h * c, 2, input, squared);

                Blas.const_ongpu(w * h, l.Kappa, norms, 1);
                for (k = 0; k < l.Size / 2; ++k)
                {
                    Blas.axpy_ongpu(w * h, l.Alpha, squared, norms, w * h * k);
                }

                for (k = 1; k < l.C; ++k)
                {
                    Blas.copy_ongpu(w * h, norms, norms, w * h * (k - 1), w * h * k);
                    int prev = k - ((l.Size - 1) / 2) - 1;
                    int next = k + (l.Size / 2);
                    if (prev >= 0) Blas.axpy_ongpu(w * h, -l.Alpha, squared, norms, w * h * prev, w * h * k);
                    if (next < l.C) Blas.axpy_ongpu(w * h, l.Alpha, squared, norms, w * h * next, w * h * k);
                }
                Array.Copy(squared, 0, l.Squared, index, squared.Length);
                Array.Copy(norms, 0, l.Norms, index, norms.Length);
            }
            Blas.pow_ongpu(w * h * c * l.Batch, -l.Beta, l.NormsGpu, l.OutputGpu);
            Blas.mul_ongpu(w * h * c * l.Batch, state.Input, 1, l.OutputGpu, 1);
        }

        public static void backward_normalization_layer_gpu(Layer l, NetworkState state)
        {
            // TODO This is approximate ;-)

            int w = l.W;
            int h = l.H;
            int c = l.C;
            Blas.pow_ongpu(w * h * c * l.Batch, -l.Beta, l.NormsGpu, state.Delta);
            Blas.mul_ongpu(w * h * c * l.Batch, l.DeltaGpu, 1, state.Delta, 1);
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

        public static void forward_convolutional_layer_gpu(Layer l, NetworkState state)
        {
            Blas.fill_ongpu(l.Outputs * l.Batch, 0, l.OutputGpu, 1);
            if (l.Binary)
            {
                l.binarize_weights_gpu(l.WeightsGpu, l.N, l.C * l.Size * l.Size, l.BinaryWeightsGpu);
                l.swap_binary();
            }

            if (l.Xnor)
            {
                l.binarize_weights_gpu(l.WeightsGpu, l.N, l.C * l.Size * l.Size, l.BinaryWeightsGpu);
                l.swap_binary();
                l.binarize_gpu(state.Input, l.C * l.H * l.W * l.Batch, l.BinaryInputGpu);
                state.Input = l.BinaryInputGpu;
            }

            unsafe
            {
                var gpuOne = System.Runtime.InteropServices.Marshal.AllocCoTaskMem(sizeof(float));
                Marshal.WriteInt32(gpuOne, 1);

                int size = Marshal.SizeOf(state.Input[0]) * state.Input.Length;
                IntPtr Input = Marshal.AllocHGlobal(size);
                Marshal.Copy(state.Input, 0, Input, state.Input.Length);

                size = Marshal.SizeOf(l.WeightsGpu[0]) * l.WeightsGpu.Length;
                IntPtr WeightsGpu = Marshal.AllocHGlobal(size);
                Marshal.Copy(l.WeightsGpu, 0, WeightsGpu, l.WeightsGpu.Length);

                size = Marshal.SizeOf(state.Workspace[0]) * state.Workspace.Length;
                IntPtr Workspace = Marshal.AllocHGlobal(size);
                Marshal.Copy(state.Workspace, 0, Workspace, state.Workspace.Length);

                size = Marshal.SizeOf(l.OutputGpu[0]) * l.OutputGpu.Length;
                IntPtr OutputGpu = Marshal.AllocHGlobal(size);
                Marshal.Copy(l.OutputGpu, 0, OutputGpu, l.OutputGpu.Length);

                using (var SrcTensorDesc = Alea.Interop.Marshal.Align(l.SrcTensorDesc))
                using (var WeightDesc = Alea.Interop.Marshal.Align(l.WeightDesc))
                using (var ConvDesc = Alea.Interop.Marshal.Align(l.ConvDesc))
                using (var DstTensorDesc = Alea.Interop.Marshal.Align(l.DstTensorDesc))
                {
                    CuDnn.cudnnConvolutionForward(CudaUtils.cudnn_handle(),
                        gpuOne,
                        (cudnnTensorStruct*)SrcTensorDesc.Handle,
                        Input,
                        (cudnnFilterStruct*)WeightDesc.Handle,
                        WeightsGpu,
                        (cudnnConvolutionStruct*)ConvDesc.Handle,
                        l.FwAlgo,
                        Workspace,
                        l.WorkspaceSize,
                        gpuOne,
                        (cudnnTensorStruct*)DstTensorDesc.Handle,
                        OutputGpu);
                }

                Marshal.Copy(Input, state.Input, 0, state.Input.Length);
                Marshal.Copy(WeightsGpu, l.WeightsGpu, 0, l.WeightsGpu.Length);
                Marshal.Copy(Workspace, state.Workspace, 0, state.Workspace.Length);
                Marshal.Copy(OutputGpu, l.OutputGpu, 0, l.OutputGpu.Length);

                Marshal.FreeCoTaskMem(gpuOne);
                Marshal.FreeHGlobal(Input);
                Marshal.FreeHGlobal(WeightsGpu);
                Marshal.FreeHGlobal(Workspace);
                Marshal.FreeHGlobal(OutputGpu);
            }

            if (l.BatchNormalize)
            {
                forward_batchnorm_layer_gpu(l, state);
            }
            Blas.add_bias_gpu(l.OutputGpu, l.BiasesGpu, l.Batch, l.N, l.OutW * l.OutH);

            ActivationsHelper.activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
            if (l.Binary || l.Xnor) l.swap_binary();
        }

        public static void backward_convolutional_layer_gpu(Layer l, NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);

            Blas.backward_bias_gpu(l.BiasUpdatesGpu, l.DeltaGpu, l.Batch, l.N, l.OutW * l.OutH);

            if (l.BatchNormalize)
            {
                backward_batchnorm_layer_gpu(l, state);
            }
            float[] original_input = state.Input;

            if (l.Xnor) state.Input = l.BinaryInputGpu;

            unsafe
            {
                var gpuOne = Marshal.AllocCoTaskMem(sizeof(float));
                Marshal.WriteInt32(gpuOne, 1);

                int size = Marshal.SizeOf(state.Input[0]) * state.Input.Length;
                IntPtr Input = Marshal.AllocHGlobal(size);
                Marshal.Copy(state.Input, 0, Input, state.Input.Length);

                size = Marshal.SizeOf(l.DeltaGpu[0]) * l.DeltaGpu.Length;
                IntPtr DeltaGpu = Marshal.AllocHGlobal(size);
                Marshal.Copy(l.DeltaGpu, 0, DeltaGpu, l.DeltaGpu.Length);

                size = Marshal.SizeOf(state.Workspace[0]) * state.Workspace.Length;
                IntPtr Workspace = Marshal.AllocHGlobal(size);
                Marshal.Copy(state.Workspace, 0, Workspace, state.Workspace.Length);

                size = Marshal.SizeOf(l.WeightUpdatesGpu[0]) * l.WeightUpdatesGpu.Length;
                IntPtr WeightUpdatesGpu = Marshal.AllocHGlobal(size);
                Marshal.Copy(l.WeightUpdatesGpu, 0, WeightUpdatesGpu, l.WeightUpdatesGpu.Length);

                using (var SrcTensorDesc = Alea.Interop.Marshal.Align(l.SrcTensorDesc))
                using (var DdstTensorDesc = Alea.Interop.Marshal.Align(l.DdstTensorDesc))
                using (var ConvDesc = Alea.Interop.Marshal.Align(l.ConvDesc))
                using (var DweightDesc = Alea.Interop.Marshal.Align(l.DweightDesc))
                {
                    CuDnn.cudnnConvolutionBackwardFilter(CudaUtils.cudnn_handle(),
                        gpuOne,
                        (cudnnTensorStruct*)SrcTensorDesc.Handle,
                        Input,
                        (cudnnTensorStruct*)DdstTensorDesc.Handle,
                        DeltaGpu,
                        (cudnnConvolutionStruct*)ConvDesc.Handle,
                        l.BfAlgo,
                        Workspace,
                        l.WorkspaceSize,
                        gpuOne,
                        (cudnnFilterStruct*)DweightDesc.Handle,
                        WeightUpdatesGpu);
                }

                Marshal.Copy(Input, state.Input, 0, state.Input.Length);
                Marshal.Copy(DeltaGpu, l.DeltaGpu, 0, l.DeltaGpu.Length);
                Marshal.Copy(Workspace, state.Workspace, 0, state.Workspace.Length);
                Marshal.Copy(WeightUpdatesGpu, l.WeightUpdatesGpu, 0, l.WeightUpdatesGpu.Length);

                Marshal.FreeCoTaskMem(gpuOne);
                Marshal.FreeHGlobal(Input);
                Marshal.FreeHGlobal(DeltaGpu);
                Marshal.FreeHGlobal(Workspace);
                Marshal.FreeHGlobal(WeightUpdatesGpu);
            }

            if (state.Delta.Length != 0)
            {
                if (l.Binary || l.Xnor) l.swap_binary();

                unsafe
                {
                    var gpuOne = Marshal.AllocCoTaskMem(sizeof(float));
                    Marshal.WriteInt32(gpuOne, 1);

                    int size = Marshal.SizeOf(l.WeightsGpu[0]) * l.WeightsGpu.Length;
                    IntPtr WeightsGpu = Marshal.AllocHGlobal(size);
                    Marshal.Copy(l.WeightsGpu, 0, WeightsGpu, l.WeightsGpu.Length);

                    size = Marshal.SizeOf(l.DeltaGpu[0]) * l.DeltaGpu.Length;
                    IntPtr DeltaGpu = Marshal.AllocHGlobal(size);
                    Marshal.Copy(l.DeltaGpu, 0, DeltaGpu, l.DeltaGpu.Length);

                    size = Marshal.SizeOf(state.Workspace[0]) * state.Workspace.Length;
                    IntPtr Workspace = Marshal.AllocHGlobal(size);
                    Marshal.Copy(state.Workspace, 0, Workspace, state.Workspace.Length);

                    size = Marshal.SizeOf(state.Delta[0]) * state.Delta.Length;
                    IntPtr Delta = Marshal.AllocHGlobal(size);
                    Marshal.Copy(state.Delta, 0, Delta, state.Delta.Length);

                    using (var WeightDesc = Alea.Interop.Marshal.Align(l.WeightDesc))
                    using (var DdstTensorDesc = Alea.Interop.Marshal.Align(l.DdstTensorDesc))
                    using (var ConvDesc = Alea.Interop.Marshal.Align(l.ConvDesc))
                    using (var DsrcTensorDesc = Alea.Interop.Marshal.Align(l.DsrcTensorDesc))
                    {
                        CuDnn.cudnnConvolutionBackwardData(CudaUtils.cudnn_handle(),
                        gpuOne,
                        (cudnnFilterStruct*)WeightDesc.Handle,
                        WeightsGpu,
                        (cudnnTensorStruct*)DdstTensorDesc.Handle,
                        DeltaGpu,
                        (cudnnConvolutionStruct*)ConvDesc.Handle,
                        l.BdAlgo,
                        Workspace,
                        l.WorkspaceSize,
                        gpuOne,
                        (cudnnTensorStruct*)DsrcTensorDesc.Handle,
                        Delta);
                    }

                    Marshal.Copy(WeightsGpu, l.WeightsGpu, 0, l.WeightsGpu.Length);
                    Marshal.Copy(DeltaGpu, l.DeltaGpu, 0, l.DeltaGpu.Length);
                    Marshal.Copy(Workspace, state.Workspace, 0, state.Workspace.Length);
                    Marshal.Copy(Delta, state.Delta, 0, state.Delta.Length);

                    Marshal.FreeCoTaskMem(gpuOne);
                    Marshal.FreeHGlobal(WeightsGpu);
                    Marshal.FreeHGlobal(DeltaGpu);
                    Marshal.FreeHGlobal(Workspace);
                    Marshal.FreeHGlobal(Delta);
                }

                if (l.Binary || l.Xnor) l.swap_binary();
                if (l.Xnor) ActivationsHelper.gradient_array_ongpu(original_input, l.Batch * l.C * l.H * l.W, Activation.Hardtan, state.Delta);
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

        public static void update_convolutional_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int size = l.Size * l.Size * l.C * l.N;
            Blas.axpy_ongpu(l.N, learning_rate / batch, l.BiasUpdatesGpu, l.BiasesGpu);
            Blas.scal_ongpu(l.N, momentum, l.BiasUpdatesGpu, 1);

            if (l.ScalesGpu.Any())
            {
                Blas.axpy_ongpu(l.N, learning_rate / batch, l.ScaleUpdatesGpu, l.ScalesGpu);
                Blas.scal_ongpu(l.N, momentum, l.ScaleUpdatesGpu, 1);
            }

            if (l.Adam)
            {
                Blas.scal_ongpu(size, l.B1, l.MGpu, 1);
                Blas.scal_ongpu(size, l.B2, l.VGpu, 1);

                Blas.axpy_ongpu(size, -decay * batch, l.WeightsGpu, l.WeightUpdatesGpu);

                Blas.axpy_ongpu(size, -(1 - l.B1), l.WeightUpdatesGpu, l.MGpu);
                Blas.mul_ongpu(size, l.WeightUpdatesGpu, 1, l.WeightUpdatesGpu, 1);
                Blas.axpy_ongpu(size, (1 - l.B2), l.WeightUpdatesGpu, l.VGpu);

                Blas.adam_gpu(size, l.WeightsGpu, l.MGpu, l.VGpu, l.B1, l.B2, learning_rate / batch, l.Eps, l.T + 1);
                Blas.fill_ongpu(size, 0, l.WeightUpdatesGpu, 1);
            }
            else
            {
                Blas.axpy_ongpu(size, -decay * batch, l.WeightsGpu, l.WeightUpdatesGpu);
                Blas.axpy_ongpu(size, learning_rate / batch, l.WeightUpdatesGpu, l.WeightsGpu);
                Blas.scal_ongpu(size, momentum, l.WeightUpdatesGpu, 1);
            }
        }

        private static void forward_avgpool_layer_kernel(int n, int w, int h, int c, float[] input, float[] output)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int k = id % c;
            id /= c;
            int b = id;

            int i;
            int out_index = (k + c * b);
            output[out_index] = 0;
            for (i = 0; i < w * h; ++i)
            {
                int in_index = i + h * w * (k + b * c);
                output[out_index] += input[in_index];
            }
            output[out_index] /= w * h;
        }

        private static void backward_avgpool_layer_kernel(int n, int w, int h, int c, float[] in_delta, float[] out_delta)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int k = id % c;
            id /= c;
            int b = id;

            int i;
            int out_index = (k + c * b);
            for (i = 0; i < w * h; ++i)
            {
                int in_index = i + h * w * (k + b * c);
                in_delta[in_index] += out_delta[out_index] / (w * h);
            }
        }

        [GpuManaged]
        public static void forward_avgpool_layer_gpu(Layer l, NetworkState state)
        {
            int n = l.C * l.Batch;

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(forward_avgpool_layer_kernel, lp, n, l.W, l.H, l.C, state.Input, l.OutputGpu);
        }

        [GpuManaged]
        public static void backward_avgpool_layer_gpu(Layer l, NetworkState state)
        {
            int n = l.C * l.Batch;

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(backward_avgpool_layer_kernel, lp, n, l.W, l.H, l.C, state.Delta, l.DeltaGpu);
        }

        private static float get_pixel_kernel(float[] image, int w, int h, int x, int y, int c, int imageStart = 0)
        {
            if (x < 0 || x >= w || y < 0 || y >= h) return 0;
            return image[imageStart + x + w * (y + c * h)];
        }

        private static Alea.float3 rgb_to_hsv_kernel(Alea.float3 rgb)
        {
            float r = rgb.x;
            float g = rgb.y;
            float b = rgb.z;

            float h, s, v;
            float max = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
            float min = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
            float delta = max - min;
            v = max;
            if (max == 0)
            {
                s = 0;
                h = -1;
            }
            else
            {
                s = delta / max;
                if (r == max)
                {
                    h = (g - b) / delta;
                }
                else if (g == max)
                {
                    h = 2 + (b - r) / delta;
                }
                else
                {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
            }
            return new Alea.float3(h, s, v);
        }

        private static Alea.float3 hsv_to_rgb_kernel(Alea.float3 hsv)
        {
            float h = hsv.x;
            float s = hsv.y;
            float v = hsv.z;

            float r, g, b;
            float f, p, q, t;

            if (s == 0)
            {
                r = g = b = v;
            }
            else
            {
                int index = (int)Math.Floor(h);
                f = h - index;
                p = v * (1 - s);
                q = v * (1 - s * f);
                t = v * (1 - s * (1 - f));
                if (index == 0)
                {
                    r = v; g = t; b = p;
                }
                else if (index == 1)
                {
                    r = q; g = v; b = p;
                }
                else if (index == 2)
                {
                    r = p; g = v; b = t;
                }
                else if (index == 3)
                {
                    r = p; g = q; b = v;
                }
                else if (index == 4)
                {
                    r = t; g = p; b = v;
                }
                else
                {
                    r = v; g = p; b = q;
                }
            }
            r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
            g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
            b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
            return new Alea.float3(r, g, b);
        }

        private static float bilinear_interpolate_kernel(float[] image, int w, int h, float x, float y, int c, int imageStart = 0)
        {
            int ix = (int)Math.Floor(x);
            int iy = (int)Math.Floor(y);

            float dx = x - ix;
            float dy = y - iy;

            float val = (1 - dy) * (1 - dx) * get_pixel_kernel(image, w, h, ix, iy, c, imageStart) +
                dy * (1 - dx) * get_pixel_kernel(image, w, h, ix, iy + 1, c, imageStart) +
                (1 - dy) * dx * get_pixel_kernel(image, w, h, ix + 1, iy, c, imageStart) +
                dy * dx * get_pixel_kernel(image, w, h, ix + 1, iy + 1, c, imageStart);
            return val;
        }

        private static void levels_image_kernel(float[] image, float[] rand, int batch, int w, int h, bool train, float saturation, float exposure, float translate, float scale, float shift)
        {
            int size = batch * w * h;
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= size) return;
            int x = id % w;
            id /= w;
            int y = id % h;
            id /= h;
            float rshift = rand[0];
            float gshift = rand[1];
            float bshift = rand[2];
            float r0 = rand[8 * id + 0];
            float r1 = rand[8 * id + 1];
            float r2 = rand[8 * id + 2];
            float r3 = rand[8 * id + 3];

            saturation = r0 * (saturation - 1) + 1;
            saturation = (r1 > .5) ? 1.0f / saturation : saturation;
            exposure = r2 * (exposure - 1) + 1;
            exposure = (r3 > .5) ? 1.0f / exposure : exposure;

            var offset = id * h * w * 3;

            float r = image[offset + x + w * (y + h * 0)];
            float g = image[offset + x + w * (y + h * 1)];
            float b = image[offset + x + w * (y + h * 2)];
            Alea.float3 rgb = new Alea.float3(r, g, b);
            if (train)
            {
                Alea.float3 hsv = rgb_to_hsv_kernel(rgb);
                hsv.y *= saturation;
                hsv.z *= exposure;
                rgb = hsv_to_rgb_kernel(hsv);
            }
            else
            {
                shift = 0;
            }
            image[offset + x + w * (y + h * 0)] = (float)(rgb.x * scale + translate + (rshift - .5) * shift);
            image[offset + x + w * (y + h * 1)] = (float)(rgb.y * scale + translate + (gshift - .5) * shift);
            image[offset + x + w * (y + h * 2)] = (float)(rgb.z * scale + translate + (bshift - .5) * shift);
        }

        private static void forward_crop_layer_kernel(float[] input, float[] rand, int size, int c, int h, int w, int crop_height, int crop_width, bool train, bool flip, float angle, float[] output)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= size) return;

            float cx = w / 2.0f;
            float cy = h / 2.0f;

            int count = id;
            int j = id % crop_width;
            id /= crop_width;
            int i = id % crop_height;
            id /= crop_height;
            int k = id % c;
            id /= c;
            int b = id;

            float r4 = rand[8 * b + 4];
            float r5 = rand[8 * b + 5];
            float r6 = rand[8 * b + 6];
            float r7 = rand[8 * b + 7];

            float dw = (w - crop_width) * r4;
            float dh = (h - crop_height) * r5;
            flip = (flip && (r6 > .5));
            angle = 2 * angle * r7 - angle;
            if (!train)
            {
                dw = (w - crop_width) / 2.0f;
                dh = (h - crop_height) / 2.0f;
                flip = false;
                angle = 0;
            }

            float x = (flip) ? w - dw - j - 1 : j + dw;
            float y = i + dh;

            float rx = (float)(Math.Cos(angle) * (x - cx) - Math.Sin(angle) * (y - cy) + cx);
            float ry = (float)(Math.Sin(angle) * (x - cx) + Math.Cos(angle) * (y - cy) + cy);

            output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k, w * h * c * b);
        }

        [GpuManaged]
        public static void forward_crop_layer_gpu(Layer l, NetworkState state)
        {
            CudaUtils.cuda_random(l.RandGpu, (ulong)l.Batch * 8);

            float radians = l.Angle * 3.14159265f / 180.0f;

            float scale = 2;
            float translate = -1;
            if (l.Noadjust)
            {
                scale = 1;
                translate = 0;
            }

            int size = l.Batch * l.W * l.H;

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(levels_image_kernel, lp, state.Input, l.RandGpu, l.Batch, l.W, l.H, state.Train, l.Saturation, l.Exposure, translate, scale, l.Shift);

            size = l.Batch * l.C * l.OutW * l.OutH;

            lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(forward_crop_layer_kernel, lp, state.Input, l.RandGpu, size, l.C, l.H, l.W, l.OutH, l.OutW, state.Train, l.Flip, radians, l.OutputGpu);
        }

        private static void yoloswag420blazeit360noscope(float[] input, int size, float[] rand, float prob, float scale)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id < size) input[id] = (rand[id] < prob) ? 0 : input[id] * scale;
        }

        [GpuManaged]
        public static void forward_dropout_layer_gpu(Layer l, NetworkState state)
        {
            if (!state.Train) return;
            int size = l.Inputs * l.Batch;
            CudaUtils.cuda_random(l.RandGpu, (ulong)size);

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(yoloswag420blazeit360noscope, lp, state.Input, size, l.RandGpu, l.Probability, l.Scale);
        }

        [GpuManaged]
        public static void backward_dropout_layer_gpu(Layer l, NetworkState state)
        {
            if (state.Delta.Length == 0) return;
            int size = l.Inputs * l.Batch;

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(size), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(yoloswag420blazeit360noscope, lp, state.Delta, size, l.RandGpu, l.Probability, l.Scale);
        }

        private static void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float[] input, float[] output, int[] indexes)
        {
            int h = (in_h + 2 * pad) / stride;
            int w = (in_w + 2 * pad) / stride;
            int c = in_c;

            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int j = id % w;
            id /= w;
            int i = id % h;
            id /= h;
            int k = id % c;
            id /= c;
            int b = id;

            int w_offset = -pad;
            int h_offset = -pad;

            int out_index = j + w * (i + h * (k + c * b));
            float max = float.NegativeInfinity;
            int max_i = -1;
            int l, m;
            for (l = 0; l < size; ++l)
            {
                for (m = 0; m < size; ++m)
                {
                    int cur_h = h_offset + i * stride + l;
                    int cur_w = w_offset + j * stride + m;
                    int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
                    bool valid = (cur_h >= 0 && cur_h < in_h &&
                            cur_w >= 0 && cur_w < in_w);
                    float val = (valid) ? input[index] : float.NegativeInfinity;
                    max_i = (val > max) ? index : max_i;
                    max = (val > max) ? val : max;
                }
            }
            output[out_index] = max;
            indexes[out_index] = max_i;
        }

        private static void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float[] delta, float[] prev_delta, int[] indexes)
        {
            int h = (in_h + 2 * pad) / stride;
            int w = (in_w + 2 * pad) / stride;
            int c = in_c;
            int area = (size - 1) / stride;

            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= n) return;

            int index = id;
            int j = id % in_w;
            id /= in_w;
            int i = id % in_h;
            id /= in_h;
            int k = id % in_c;
            id /= in_c;
            int b = id;

            int w_offset = -pad;
            int h_offset = -pad;

            float d = 0;
            int l, m;
            for (l = -area; l < area + 1; ++l)
            {
                for (m = -area; m < area + 1; ++m)
                {
                    int out_w = (j - w_offset) / stride + m;
                    int out_h = (i - h_offset) / stride + l;
                    int out_index = out_w + w * (out_h + h * (k + c * b));
                    bool valid = (out_w >= 0 && out_w < w &&
                             out_h >= 0 && out_h < h);
                    d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
                }
            }
            prev_delta[index] += d;
        }

        [GpuManaged]
        public static void forward_maxpool_layer_gpu(Layer l, NetworkState state)
        {
            int h = l.OutH;
            int w = l.OutH;
            int c = l.C;

            var n = h * w * c * l.Batch;

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(forward_maxpool_layer_kernel, lp, n, l.H, l.W, l.C, l.Stride, l.Size, l.Pad, state.Input, l.OutputGpu, l.IndexesGpu);
        }

        [GpuManaged]
        public static void backward_maxpool_layer_gpu(Layer l, NetworkState state)
        {
            var n = l.H * l.W * l.C * l.Batch;

            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(backward_maxpool_layer_kernel, lp, n, l.H, l.W, l.C, l.Stride, l.Size, l.Pad, l.DeltaGpu, state.Delta, l.IndexesGpu);
        }

        public static Layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Region;

            l.N = n;
            l.Batch = batch;
            l.H = h;
            l.W = w;
            l.Classes = classes;
            l.Coords = coords;
            l.Cost = 0;
            l.Biases = new float[n * 2];
            l.BiasUpdates = new float[n * 2];
            l.Outputs = h * w * n * (classes + coords + 1);
            l.Inputs = l.Outputs;
            l.Truths = 30 * (5);
            l.Delta = new float[batch * l.Outputs];
            l.Output = new float[batch * l.Outputs];
            int i;
            for (i = 0; i < n * 2; ++i)
            {
                l.Biases[i] = .5f;
            }

            l.Forward = forward_region_layer;
            l.Backward = backward_region_layer;
            l.ForwardGpu = forward_region_layer_gpu;
            l.BackwardGpu = backward_region_layer_gpu;
            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();

            Console.Error.Write($"detection\n");

            return l;
        }

        public void resize_region_layer(int w, int h)
        {
            W = w;
            H = h;

            Outputs = h * w * N * (Classes + Coords + 1);
            Inputs = Outputs;

            Array.Resize(ref Output, Batch * Outputs * sizeof(float));
            Array.Resize(ref Delta, Batch * Outputs * sizeof(float));

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();
        }

        public static Box get_region_box(float[] x, float[] biases, int n, int index, int i, int j, int w, int h)
        {
            Box b = new Box();
            b.X = (i + ActivationsHelper.Logistic_activate(x[index + 0])) / w;
            b.Y = (j + ActivationsHelper.Logistic_activate(x[index + 1])) / h;
            b.W = (float)Math.Exp(x[index + 2]) * biases[2 * n];
            b.H = (float)Math.Exp(x[index + 3]) * biases[2 * n + 1];

            b.W = (float)Math.Exp(x[index + 2]) * biases[2 * n] / w;
            b.H = (float)Math.Exp(x[index + 3]) * biases[2 * n + 1] / h;

            return b;
        }

        public static float delta_region_box(Box truth, float[] x, float[] biases, int n, int index, int i, int j, int w, int h, float[] delta, float scale)
        {
            Box pred = get_region_box(x, biases, n, index, i, j, w, h);
            float iou = Box.box_iou(pred, truth);

            float tx = (truth.X * w - i);
            float ty = (truth.Y * h - j);
            float tw = (float)Math.Log(truth.W / biases[2 * n]);
            float th = (float)Math.Log(truth.H / biases[2 * n + 1]);

            tw = (float)Math.Log(truth.W * w / biases[2 * n]);
            th = (float)Math.Log(truth.H * h / biases[2 * n + 1]);

            delta[index + 0] = scale * (tx - ActivationsHelper.Logistic_activate(x[index + 0])) * ActivationsHelper.Logistic_gradient(ActivationsHelper.Logistic_activate(x[index + 0]));
            delta[index + 1] = scale * (ty - ActivationsHelper.Logistic_activate(x[index + 1])) * ActivationsHelper.Logistic_gradient(ActivationsHelper.Logistic_activate(x[index + 1]));
            delta[index + 2] = scale * (tw - x[index + 2]);
            delta[index + 3] = scale * (th - x[index + 3]);
            return iou;
        }

        public static void delta_region_class(float[] output, float[] delta, int index, int sclass, int classes, Tree hier, float scale, ref float avg_cat)
        {
            int i, n;
            if (hier != null)
            {
                float pred = 1;
                while (sclass >= 0)
                {
                    pred *= output[index + sclass];
                    int g = hier.Group[sclass];
                    int offset = hier.GroupOffset[g];
                    for (i = 0; i < hier.GroupSize[g]; ++i)
                    {
                        delta[index + offset + i] = scale * (0 - output[index + offset + i]);
                    }
                    delta[index + sclass] = scale * (1 - output[index + sclass]);

                    sclass = hier.Parent[sclass];
                }
                avg_cat += pred;
            }
            else
            {
                for (n = 0; n < classes; ++n)
                {
                    delta[index + n] = scale * (((n == sclass) ? 1 : 0) - output[index + n]);
                    if (n == sclass) avg_cat += output[index + n];
                }
            }
        }

        public static float logit(float x)
        {
            return (float)Math.Log(x / (1.0f - x));
        }

        public static bool tisnan(float x)
        {
            return (x != x);
        }

        public static void forward_region_layer(Layer l, NetworkState state)
        {
            int i, j, b, t, n;
            int size = l.Coords + l.Classes + 1;
            Array.Copy(l.Output, state.Input, l.Outputs * l.Batch);
            for (b = 0; b < l.Batch; ++b)
            {
                for (i = 0; i < l.H * l.W * l.N; ++i)
                {
                    int index = size * i + b * l.Outputs;
                    l.Output[index + 4] = ActivationsHelper.Logistic_activate(l.Output[index + 4]);
                }
            }

            if (!state.Train) return;
            l.Delta = new float[l.Outputs * l.Batch];
            float avg_iou = 0;
            float recall = 0;
            float avg_cat = 0;
            float avg_obj = 0;
            float avg_anyobj = 0;
            int count = 0;
            int class_count = 0;
            (l.Cost) = 0;
            for (b = 0; b < l.Batch; ++b)
            {
                if (l.SoftmaxTree != null)
                {
                    bool onlyclass = false;
                    for (t = 0; t < 30; ++t)
                    {
                        Box truth = new Box(state.Truth, t * 5 + b * l.Truths);
                        if (truth.X == 0) break;
                        int sclass = (int)state.Truth[t * 5 + b * l.Truths + 4];
                        float maxp = 0;
                        int maxi = 0;
                        if (truth.X > 100000 && truth.Y > 100000)
                        {
                            for (n = 0; n < l.N * l.W * l.H; ++n)
                            {
                                int index = size * n + b * l.Outputs + 5;
                                float scale = l.Output[index - 1];
                                float p = scale * l.SoftmaxTree.Get_hierarchy_probability(l.Output, index, sclass);
                                if (p > maxp)
                                {
                                    maxp = p;
                                    maxi = n;
                                }
                            }
                            int index2 = size * maxi + b * l.Outputs + 5;
                            delta_region_class(l.Output, l.Delta, index2, sclass, l.Classes, l.SoftmaxTree, l.ClassScale, ref avg_cat);
                            ++class_count;
                            onlyclass = true;
                            break;
                        }
                    }
                    if (onlyclass) continue;
                }
                for (j = 0; j < l.H; ++j)
                {
                    for (i = 0; i < l.W; ++i)
                    {
                        for (n = 0; n < l.N; ++n)
                        {
                            int index = size * (j * l.W * l.N + i * l.N + n) + b * l.Outputs;
                            Box pred = get_region_box(l.Output, l.Biases, n, index, i, j, l.W, l.H);
                            float best_iou = 0;
                            int best_class = -1;
                            for (t = 0; t < 30; ++t)
                            {
                                Box truth = new Box(state.Truth, t * 5 + b * l.Truths);
                                if (truth.X == 0) break;
                                float iou = Box.box_iou(pred, truth);
                                if (iou > best_iou)
                                {
                                    best_class = (int)state.Truth[t * 5 + b * l.Truths + 4];
                                    best_iou = iou;
                                }
                            }
                            avg_anyobj += l.Output[index + 4];
                            l.Delta[index + 4] = l.NoobjectScale * ((0 - l.Output[index + 4]) * ActivationsHelper.Logistic_gradient(l.Output[index + 4]));
                            if (l.Classfix == -1) l.Delta[index + 4] = l.NoobjectScale * ((best_iou - l.Output[index + 4]) * ActivationsHelper.Logistic_gradient(l.Output[index + 4]));
                            else
                            {
                                if (best_iou > l.Thresh)
                                {
                                    l.Delta[index + 4] = 0;
                                    if (l.Classfix > 0)
                                    {
                                        delta_region_class(l.Output, l.Delta, index + 5, best_class, l.Classes, l.SoftmaxTree, l.ClassScale * (l.Classfix == 2 ? l.Output[index + 4] : 1), ref avg_cat);
                                        ++class_count;
                                    }
                                }
                            }

                            if ((state.Net.Seen) < 12800)
                            {
                                Box truth = new Box();
                                truth.X = (i + .5f) / l.W;
                                truth.Y = (j + .5f) / l.H;
                                truth.W = l.Biases[2 * n];
                                truth.H = l.Biases[2 * n + 1];

                                truth.W = l.Biases[2 * n] / l.W;
                                truth.H = l.Biases[2 * n + 1] / l.H;

                                delta_region_box(truth, l.Output, l.Biases, n, index, i, j, l.W, l.H, l.Delta, .01f);
                            }
                        }
                    }
                }
                for (t = 0; t < 30; ++t)
                {
                    Box truth = new Box(state.Truth, t * 5 + b * l.Truths);

                    if (truth.X == 0) break;
                    float best_iou = 0;
                    int best_index = 0;
                    int best_n = 0;
                    i = (int)(truth.X * l.W);
                    j = (int)(truth.Y * l.H);
                    Box truth_shift = truth;
                    truth_shift.X = 0;
                    truth_shift.Y = 0;
                    for (n = 0; n < l.N; ++n)
                    {
                        int index = size * (j * l.W * l.N + i * l.N + n) + b * l.Outputs;
                        Box pred = get_region_box(l.Output, l.Biases, n, index, i, j, l.W, l.H);
                        if (l.BiasMatch)
                        {
                            pred.W = l.Biases[2 * n];
                            pred.H = l.Biases[2 * n + 1];

                            pred.W = l.Biases[2 * n] / l.W;
                            pred.H = l.Biases[2 * n + 1] / l.H;

                        }
                        pred.X = 0;
                        pred.Y = 0;
                        float iouIn = Box.box_iou(pred, truth_shift);
                        if (iouIn > best_iou)
                        {
                            best_index = index;
                            best_iou = iouIn;
                            best_n = n;
                        }
                    }
                    float iou = delta_region_box(truth, l.Output, l.Biases, best_n, best_index, i, j, l.W, l.H, l.Delta, l.CoordScale);
                    if (iou > .5) recall += 1;
                    avg_iou += iou;

                    avg_obj += l.Output[best_index + 4];
                    l.Delta[best_index + 4] = l.ObjectScale * (1 - l.Output[best_index + 4]) * ActivationsHelper.Logistic_gradient(l.Output[best_index + 4]);
                    if (l.Rescore)
                    {
                        l.Delta[best_index + 4] = l.ObjectScale * (iou - l.Output[best_index + 4]) * ActivationsHelper.Logistic_gradient(l.Output[best_index + 4]);
                    }


                    int sclass = (int)state.Truth[t * 5 + b * l.Truths + 4];
                    if (l.Map.Length != 0) sclass = l.Map[sclass];
                    delta_region_class(l.Output, l.Delta, best_index + 5, sclass, l.Classes, l.SoftmaxTree, l.ClassScale, ref avg_cat);
                    ++count;
                    ++class_count;
                }
            }
            (l.Cost) = (float)Math.Pow(Utils.mag_array(l.Delta, l.Outputs * l.Batch), 2);
            Console.Write($"Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.W * l.H * l.N * l.Batch), recall / count, count);
        }

        public static void backward_region_layer(Layer l, NetworkState state)
        {
            Blas.Axpy_cpu(l.Batch * l.Inputs, 1, l.Delta, state.Delta);
        }

        public static void get_region_boxes(Layer l, int w, int h, float thresh, float[][] probs, Box[] boxes, bool only_objectness, int[] map)
        {
            int i, j, n;
            float[] predictions = l.Output;
            for (i = 0; i < l.W * l.H; ++i)
            {
                int row = i / l.W;
                int col = i % l.W;
                for (n = 0; n < l.N; ++n)
                {
                    int index = i * l.N + n;
                    int p_index = index * (l.Classes + 5) + 4;
                    float scale = predictions[p_index];
                    if (l.Classfix == -1 && scale < .5) scale = 0;
                    int box_index = index * (l.Classes + 5);
                    boxes[index] = get_region_box(predictions, l.Biases, n, box_index, col, row, l.W, l.H);
                    boxes[index].X *= w;
                    boxes[index].Y *= h;
                    boxes[index].W *= w;
                    boxes[index].H *= h;

                    int class_index = index * (l.Classes + 5) + 5;
                    if (l.SoftmaxTree != null)
                    {
                        l.SoftmaxTree.Hierarchy_predictions(predictions, class_index, l.Classes, 0);
                        bool found = false;
                        if (map.Length != 0)
                        {
                            for (j = 0; j < 200; ++j)
                            {
                                float prob = scale * predictions[class_index + map[j]];
                                probs[index][j] = (prob > thresh) ? prob : 0;
                            }
                        }
                        else
                        {
                            for (j = l.Classes - 1; j >= 0; --j)
                            {
                                if (!found && predictions[class_index + j] > .5)
                                {
                                    found = true;
                                }
                                else
                                {
                                    predictions[class_index + j] = 0;
                                }
                                float prob = predictions[class_index + j];
                                probs[index][j] = (scale > thresh) ? prob : 0;
                            }
                        }
                    }
                    else
                    {
                        for (j = 0; j < l.Classes; ++j)
                        {
                            float prob = scale * predictions[class_index + j];
                            probs[index][j] = (prob > thresh) ? prob : 0;
                        }
                    }
                    if (only_objectness)
                    {
                        probs[index][0] = scale;
                    }
                }
            }
        }

        public static void forward_region_layer_gpu(Layer l, NetworkState state)
        {
            Blas.flatten_ongpu(state.Input, l.H * l.W, l.N * (l.Coords + l.Classes + 1), l.Batch, 1, l.OutputGpu);
            if (l.SoftmaxTree != null)
            {
                int i;
                int count = 5;
                for (i = 0; i < l.SoftmaxTree.Groups; ++i)
                {
                    int group_size = l.SoftmaxTree.GroupSize[i];
                    Blas.softmax_gpu(l.OutputGpu, group_size, l.Classes + 5, l.W * l.H * l.N * l.Batch, 1, l.OutputGpu, count, count);
                    count += group_size;
                }
            }
            else if (l.Softmax)
            {
                Blas.softmax_gpu(l.OutputGpu, l.Classes, l.Classes + 5, l.W * l.H * l.N * l.Batch, 1, l.OutputGpu, 5, 5);
            }

            float[] in_cpu = new float[l.Batch * l.Inputs];
            float[] truth_cpu = new float[0];
            if (state.Truth.Length != 0)
            {
                int num_truth = l.Batch * l.Truths;
                truth_cpu = new float[num_truth];
                Array.Copy(state.Truth, truth_cpu, num_truth);
            }
            Array.Copy(l.OutputGpu, in_cpu, l.Batch * l.Inputs);
            NetworkState cpu_state = state;
            cpu_state.Train = state.Train;
            cpu_state.Truth = truth_cpu;
            cpu_state.Input = in_cpu;
            forward_region_layer(l, cpu_state);
            if (!state.Train) return;
            Array.Copy(l.Delta, l.DeltaGpu, l.Batch * l.Outputs);
        }

        public static void backward_region_layer_gpu(Layer l, NetworkState state)
        {
            Blas.flatten_ongpu(l.DeltaGpu, l.H * l.W, l.N * (l.Coords + l.Classes + 1), l.Batch, 0, state.Delta);
        }

        public static Layer make_route_layer(int batch, int n, int[] input_layers, int[] input_sizes)
        {
            Console.Error.Write($"route ");
            Layer l = new Layer();
            l.LayerType = LayerType.Route;
            l.Batch = batch;
            l.N = n;
            l.InputLayers = input_layers;
            l.InputSizes = input_sizes;
            int i;
            int outputs = 0;
            for (i = 0; i < n; ++i)
            {
                Console.Error.Write($" %d", input_layers[i]);
                outputs += input_sizes[i];
            }
            Console.Error.Write($"\n");
            l.Outputs = outputs;
            l.Inputs = outputs;
            l.Delta = new float[outputs * batch];
            l.Output = new float[outputs * batch];

            l.Forward = forward_route_layer;
            l.Backward = backward_route_layer;
            l.ForwardGpu = forward_route_layer_gpu;
            l.BackwardGpu = backward_route_layer_gpu;

            l.DeltaGpu = (float[])l.Delta.Clone();
            l.OutputGpu = (float[])l.Output.Clone();
            return l;
        }

        public static void resize_route_layer(Layer l, Network net)
        {
            int i;
            Layer first = net.Layers[l.InputLayers[0]];
            l.OutW = first.OutW;
            l.OutH = first.OutH;
            l.OutC = first.OutC;
            l.Outputs = first.Outputs;
            l.InputSizes[0] = first.Outputs;
            for (i = 1; i < l.N; ++i)
            {
                int index = l.InputLayers[i];
                Layer next = net.Layers[index];
                l.Outputs += next.Outputs;
                l.InputSizes[i] = next.Outputs;
                if (next.OutW == first.OutW && next.OutH == first.OutH)
                {
                    l.OutC += next.OutC;
                }
                else
                {
                    Console.Write($"%d %d, %d %d\n", next.OutW, next.OutH, first.OutW, first.OutH);
                    l.OutH = l.OutW = l.OutC = 0;
                }
            }
            l.Inputs = l.Outputs;
            Array.Resize(ref l.Delta, l.Outputs * l.Batch);
            Array.Resize(ref l.Output, l.Outputs * l.Batch);

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
        }

        public static void forward_route_layer(Layer l, NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < l.N; ++i)
            {
                int index = l.InputLayers[i];
                float[] input = state.Net.Layers[index].Output;
                int input_size = l.InputSizes[i];
                for (j = 0; j < l.Batch; ++j)
                {
                    Blas.Copy_cpu(input_size, input, l.Output, j * input_size, offset + j * l.Outputs);
                }
                offset += input_size;
            }
        }

        public static void backward_route_layer(Layer l, NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < l.N; ++i)
            {
                int index = l.InputLayers[i];
                float[] delta = state.Net.Layers[index].Delta;
                int input_size = l.InputSizes[i];
                for (j = 0; j < l.Batch; ++j)
                {
                    Blas.Axpy_cpu(input_size, 1, l.Delta, delta, offset + j * l.Outputs, j * input_size);
                }
                offset += input_size;
            }
        }

        public static void forward_route_layer_gpu(Layer l, NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < l.N; ++i)
            {
                int index = l.InputLayers[i];
                float[] input = state.Net.Layers[index].OutputGpu;
                int input_size = l.InputSizes[i];
                for (j = 0; j < l.Batch; ++j)
                {
                    Blas.copy_ongpu(input_size, input, l.OutputGpu, j * input_size, offset + j * l.Outputs);
                }
                offset += input_size;
            }
        }

        public static void backward_route_layer_gpu(Layer l, NetworkState state)
        {
            int i, j;
            int offset = 0;
            for (i = 0; i < l.N; ++i)
            {
                int index = l.InputLayers[i];
                float[] delta = state.Net.Layers[index].DeltaGpu;
                int input_size = l.InputSizes[i];
                for (j = 0; j < l.Batch; ++j)
                {
                    Blas.axpy_ongpu(input_size, 1, l.DeltaGpu, delta, offset + j * l.Outputs, j * input_size);
                }
                offset += input_size;
            }
        }

        public static Layer make_reorg_layer(int batch, int w, int h, int c, int stride, bool reverse)
        {
            Layer l = new Layer();
            l.LayerType = LayerType.Reorg;
            l.Batch = batch;
            l.Stride = stride;
            l.H = h;
            l.W = w;
            l.C = c;
            if (reverse)
            {
                l.OutW = w * stride;
                l.OutH = h * stride;
                l.OutC = c / (stride * stride);
            }
            else
            {
                l.OutW = w / stride;
                l.OutH = h / stride;
                l.OutC = c * (stride * stride);
            }
            l.Reverse = reverse;
            Console.Error.Write($"reorg              /%2d  %4d x%4d x%4d   .  %4d x%4d x%4d\n", stride, w, h, c, l.OutW, l.OutH, l.OutC);
            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = h * w * c;
            int output_size = l.OutH * l.OutW * l.OutC * batch;
            l.Output = new float[output_size];
            l.Delta = new float[output_size];

            l.Forward = forward_reorg_layer;
            l.Backward = backward_reorg_layer;
            l.ForwardGpu = forward_reorg_layer_gpu;
            l.BackwardGpu = backward_reorg_layer_gpu;

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            return l;
        }

        public static void resize_reorg_layer(Layer l, int w, int h)
        {
            int stride = l.Stride;
            int c = l.C;

            l.H = h;
            l.W = w;

            if (l.Reverse)
            {
                l.OutW = w * stride;
                l.OutH = h * stride;
                l.OutC = c / (stride * stride);
            }
            else
            {
                l.OutW = w / stride;
                l.OutH = h / stride;
                l.OutC = c * (stride * stride);
            }

            l.Outputs = l.OutH * l.OutW * l.OutC;
            l.Inputs = l.Outputs;
            int output_size = l.Outputs * l.Batch;

            Array.Resize(ref l.Output, output_size);
            Array.Resize(ref l.Delta, output_size);

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
        }

        public static void forward_reorg_layer(Layer l, NetworkState state)
        {
            if (l.Reverse)
            {
                Blas.Reorg_cpu(state.Input, l.W, l.H, l.C, l.Batch, l.Stride, 1, l.Output);
            }
            else
            {
                Blas.Reorg_cpu(state.Input, l.W, l.H, l.C, l.Batch, l.Stride, 0, l.Output);
            }
        }

        public static void backward_reorg_layer(Layer l, NetworkState state)
        {
            if (l.Reverse)
            {
                Blas.Reorg_cpu(l.Delta, l.W, l.H, l.C, l.Batch, l.Stride, 0, state.Delta);
            }
            else
            {
                Blas.Reorg_cpu(l.Delta, l.W, l.H, l.C, l.Batch, l.Stride, 1, state.Delta);
            }
        }

        public static void forward_reorg_layer_gpu(Layer l, NetworkState state)
        {
            if (l.Reverse)
            {
                Blas.reorg_ongpu(state.Input, l.W, l.H, l.C, l.Batch, l.Stride, 1, l.OutputGpu);
            }
            else
            {
                Blas.reorg_ongpu(state.Input, l.W, l.H, l.C, l.Batch, l.Stride, 0, l.OutputGpu);
            }
        }

        public static void backward_reorg_layer_gpu(Layer l, NetworkState state)
        {
            if (l.Reverse)
            {
                Blas.reorg_ongpu(l.DeltaGpu, l.W, l.H, l.C, l.Batch, l.Stride, 0, state.Delta);
            }
            else
            {
                Blas.reorg_ongpu(l.DeltaGpu, l.W, l.H, l.C, l.Batch, l.Stride, 1, state.Delta);
            }
        }

        public static Layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
        {
            Console.Error.Write($"Shortcut Layer: %d\n", index);
            Layer l = new Layer();
            l.LayerType = LayerType.Shortcut;
            l.Batch = batch;
            l.W = w2;
            l.H = h2;
            l.C = c2;
            l.OutW = w;
            l.OutH = h;
            l.OutC = c;
            l.Outputs = w * h * c;
            l.Inputs = l.Outputs;

            l.Index = index;

            l.Delta = new float[l.Outputs * batch];
            l.Output = new float[l.Outputs * batch];

            l.Forward = forward_shortcut_layer;
            l.Backward = backward_shortcut_layer;
            l.ForwardGpu = forward_shortcut_layer_gpu;
            l.BackwardGpu = backward_shortcut_layer_gpu;

            l.DeltaGpu = (float[])l.Delta.Clone();
            l.OutputGpu = (float[])l.Output.Clone();
            return l;
        }
        
        public static void forward_shortcut_layer(Layer l, NetworkState state)
        {
            Blas.Copy_cpu(l.Outputs * l.Batch, state.Input, l.Output);
            Blas.Shortcut_cpu(l.Batch, l.W, l.H, l.C, state.Net.Layers[l.Index].Output, l.OutW, l.OutH, l.OutC, l.Output);
            ActivationsHelper.Activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }
        
        public static void backward_shortcut_layer(Layer l, NetworkState state)
        {
            ActivationsHelper.Gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);
            Blas.Axpy_cpu(l.Outputs * l.Batch, 1, l.Delta, state.Delta);
            Blas.Shortcut_cpu(l.Batch, l.OutW, l.OutH, l.OutC, l.Delta, l.W, l.H, l.C, state.Net.Layers[l.Index].Delta);
        }
        
        public static void forward_shortcut_layer_gpu(Layer l, NetworkState state)
        {
            Blas.copy_ongpu(l.Outputs * l.Batch, state.Input, l.OutputGpu);
            Blas.shortcut_gpu(l.Batch, l.W, l.H, l.C, state.Net.Layers[l.Index].OutputGpu, l.OutW, l.OutH, l.OutC, l.OutputGpu);
            ActivationsHelper.activate_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation);
        }
        
        public static void backward_shortcut_layer_gpu(Layer l, NetworkState state)
        {
            ActivationsHelper.gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            Blas.axpy_ongpu(l.Outputs * l.Batch, 1, l.DeltaGpu, state.Delta);
            Blas.shortcut_gpu(l.Batch, l.OutW, l.OutH, l.OutC, l.DeltaGpu, l.W, l.H, l.C, state.Net.Layers[l.Index].DeltaGpu);
        }
        
        public static Layer make_softmax_layer(int batch, int inputs, int groups)
        {
            Console.Error.Write($"softmax                                        %4d\n", inputs);
            Layer l = new Layer();
            l.LayerType = LayerType.Softmax;
            l.Batch = batch;
            l.Groups = groups;
            l.Inputs = inputs;
            l.Outputs = inputs;
            l.Output = new float[inputs * batch];
            l.Delta = new float[inputs * batch];

            l.Forward = forward_softmax_layer;
            l.Backward = backward_softmax_layer;
            l.ForwardGpu = forward_softmax_layer_gpu;
            l.BackwardGpu = backward_softmax_layer_gpu;

            l.OutputGpu = (float[])l.Output.Clone();
            l.DeltaGpu = (float[])l.Delta.Clone();
            return l;
        }
        
        public static void softmax_tree(float[] input, int batch, int inputs, float temp, Tree hierarchy, float[] output)
        {
            int b;
            for (b = 0; b < batch; ++b)
            {
                int i;
                int count = 0;
                for (i = 0; i < hierarchy.Groups; ++i)
                {
                    int group_size = hierarchy.GroupSize[i];
                    Blas.Softmax(input , group_size, temp, output, b * inputs + count,  b * inputs + count);
                    count += group_size;
                }
            }
        }
        
        public static void forward_softmax_layer(Layer l, NetworkState state)
        {
            int b;
            int inputs = l.Inputs / l.Groups;
            int batch = l.Batch * l.Groups;
            if (l.SoftmaxTree != null)
            {
                softmax_tree(state.Input, batch, inputs, l.Temperature, l.SoftmaxTree, l.Output);
            }
            else
            {
                for (b = 0; b < batch; ++b)
                {
                    Blas.Softmax(state.Input, inputs, l.Temperature, l.Output, b * inputs, b * inputs);
                }
            }
        }
        
        public static void backward_softmax_layer(Layer l, NetworkState state)
        {
            int i;
            for (i = 0; i < l.Inputs * l.Batch; ++i)
            {
                state.Delta[i] += l.Delta[i];
            }
        }
        
        public static void pull_softmax_layer_output(Layer Layer)
        {
            Array.Copy(Layer.OutputGpu, Layer.Output, Layer.Inputs * Layer.Batch);
        }
        
        public static void forward_softmax_layer_gpu(Layer l, NetworkState state)
        {
            int inputs = l.Inputs / l.Groups;
            int batch = l.Batch * l.Groups;
            if (l.SoftmaxTree != null)
            {
                int i;
                int count = 0;
                for (i = 0; i < l.SoftmaxTree.Groups; ++i)
                {
                    int group_size = l.SoftmaxTree.GroupSize[i];
                    Blas.softmax_gpu(state.Input, group_size, inputs, batch, l.Temperature, l.OutputGpu, count, count);
                    count += group_size;
                }
            }
            else
            {
                Blas.softmax_gpu(state.Input, inputs, inputs, batch, l.Temperature, l.OutputGpu);
            }
        }
        
        public static void backward_softmax_layer_gpu(Layer Layer, NetworkState state)
        {
            Blas.axpy_ongpu(Layer.Batch * Layer.Inputs, 1, Layer.DeltaGpu, state.Delta);
        }
        
        public static Layer make_rnn_layer(int batch, int inputs, int hidden, int outputs, int steps, Activation activation, bool batch_normalize, int log)
        {
            Console.Error.Write($"RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
            batch = batch / steps;
            Layer l = new Layer();
            l.Batch = batch;
            l.LayerType = LayerType.Rnn;
            l.Steps = steps;
            l.Hidden = hidden;
            l.Inputs = inputs;

            l.State = new float[batch * hidden * (steps + 1)];

            l.InputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.InputLayer) = make_connected_layer(batch * steps, inputs, hidden, activation, batch_normalize);
            l.InputLayer.Batch = batch;

            l.SelfLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.SelfLayer) = make_connected_layer(batch * steps, hidden, hidden, (log == 2) ? Activation.Loggy : (log == 1 ? Activation.Logistic : activation), batch_normalize);
            l.SelfLayer.Batch = batch;

            l.OutputLayer = new Layer();
            Console.Error.Write($"\t\t");
            (l.OutputLayer) = make_connected_layer(batch * steps, hidden, outputs, activation, batch_normalize);
            l.OutputLayer.Batch = batch;

            l.Outputs = outputs;
            l.Output = l.OutputLayer.Output;
            l.Delta = l.OutputLayer.Delta;

            l.Forward = forward_rnn_layer;
            l.Backward = backward_rnn_layer;
            l.Update = update_rnn_layer;
            l.ForwardGpu = forward_rnn_layer_gpu;
            l.BackwardGpu = backward_rnn_layer_gpu;
            l.UpdateGpu = update_rnn_layer_gpu;
            l.StateGpu = (float[])l.State.Clone();
            l.OutputGpu = l.OutputLayer.OutputGpu;
            l.DeltaGpu = l.OutputLayer.DeltaGpu;

            return l;
        }
        
        public static void update_rnn_layer(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_connected_layer((l.InputLayer), batch, learning_rate, momentum, decay);
            update_connected_layer((l.SelfLayer), batch, learning_rate, momentum, decay);
            update_connected_layer((l.OutputLayer), batch, learning_rate, momentum, decay);
        }
        
        public static void forward_rnn_layer(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.SelfLayer);
            Layer output_layer = (l.OutputLayer);

            Blas.Fill_cpu(l.Outputs * l.Batch * l.Steps, 0, output_layer.Delta, 1);
            Blas.Fill_cpu(l.Hidden * l.Batch * l.Steps, 0, self_layer.Delta, 1);
            Blas.Fill_cpu(l.Hidden * l.Batch * l.Steps, 0, input_layer.Delta, 1);
            if (state.Train) Blas.Fill_cpu(l.Hidden * l.Batch, 0, l.State, 1);

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = state.Input;
                forward_connected_layer(input_layer, s);

                s.Input = l.State;
                forward_connected_layer(self_layer, s);

                float[] old_state = l.State;
                if (state.Train)
                {
                    Utils.IncArray(ref l.State, ref l.StateBackup, l.StateIndex, l.StateIndex += l.Hidden * l.Batch);
                }
                if (l.Shortcut)
                {
                    Blas.Copy_cpu(l.Hidden * l.Batch, old_state, l.State);
                }
                else
                {
                    Blas.Fill_cpu(l.Hidden * l.Batch, 0, l.State, 1);
                }
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, input_layer.Output, l.State);
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output, l.State);

                s.Input = l.State;
                forward_connected_layer(output_layer, s);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch);
                increment_layer(input_layer, 1);
                increment_layer(self_layer, 1);
                increment_layer(output_layer, 1);
            }
        }
        
        public static void backward_rnn_layer(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.SelfLayer);
            Layer output_layer = (l.OutputLayer);

            increment_layer(input_layer, l.Steps - 1);
            increment_layer(self_layer, l.Steps - 1);
            increment_layer(output_layer, l.Steps - 1);

            Utils.IncArray(ref l.State, ref l.StateBackup, l.StateIndex, l.StateIndex += l.Hidden * l.Batch * l.Steps);
            for (i = l.Steps - 1; i >= 0; --i)
            {
                Blas.Copy_cpu(l.Hidden * l.Batch, input_layer.Output, l.State);
                Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Output, l.State);

                s.Input = l.State;
                s.Delta = self_layer.Delta;
                backward_connected_layer(output_layer, s);

                Utils.DecArray(ref l.State, ref l.StateBackup, l.StateIndex, l.StateIndex -= l.Hidden * l.Batch);

                s.Input = l.State;
                s.Delta = new float[self_layer.Delta.Length + l.Hidden * l.Batch];
                Array.Copy(self_layer.DeltaBackup, self_layer.DeltaIndex - l.Hidden * l.Batch, s.Delta, 0, s.Delta.Length);
                if (i == 0) s.Delta = new float[0];
                backward_connected_layer(self_layer, s);

                Blas.Copy_cpu(l.Hidden * l.Batch, self_layer.Delta, input_layer.Delta);
                if (i > 0 && l.Shortcut) Blas.Axpy_cpu(l.Hidden * l.Batch, 1, self_layer.Delta, self_layer.DeltaBackup, 0, self_layer.DeltaIndex - l.Hidden * l.Batch);

                s.Input = new float[state.Input.Length - i * l.Inputs * l.Batch];
                Array.Copy(state.Input, i * l.Inputs * l.Batch, s.Input, 0, s.Input.Length);
                if (state.Delta.Length != 0)
                {
                    s.Delta = new float[state.Delta.Length - i * l.Inputs * l.Batch];
                    Array.Copy(state.Delta, i * l.Inputs * l.Batch, s.Delta, 0, s.Delta.Length);
                }
                else s.Delta = new float[0];
                backward_connected_layer(input_layer, s);

                increment_layer(input_layer, -1);
                increment_layer(self_layer, -1);
                increment_layer(output_layer, -1);
            }
        }
        
        public static void pull_rnn_layer(Layer l)
        {
            l.InputLayer.pull_connected_layer();
            l.SelfLayer.pull_connected_layer();
            l.OutputLayer.pull_connected_layer();
        }
        
        public static void push_rnn_layer(Layer l)
        {
            l.InputLayer.push_connected_layer();
            l.SelfLayer.push_connected_layer();
            l.OutputLayer.push_connected_layer();
        }
        
        public static void update_rnn_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            update_connected_layer_gpu((l.InputLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu((l.SelfLayer), batch, learning_rate, momentum, decay);
            update_connected_layer_gpu((l.OutputLayer), batch, learning_rate, momentum, decay);
        }
        
        public static void forward_rnn_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.SelfLayer);
            Layer output_layer = (l.OutputLayer);

            Blas.fill_ongpu(l.Outputs * l.Batch * l.Steps, 0, output_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Hidden * l.Batch * l.Steps, 0, self_layer.DeltaGpu, 1);
            Blas.fill_ongpu(l.Hidden * l.Batch * l.Steps, 0, input_layer.DeltaGpu, 1);
            if (state.Train) Blas.fill_ongpu(l.Hidden * l.Batch, 0, l.StateGpu, 1);

            for (i = 0; i < l.Steps; ++i)
            {
                s.Input = state.Input;
                forward_connected_layer_gpu(input_layer, s);

                s.Input = l.StateGpu;
                forward_connected_layer_gpu(self_layer, s);

                float[] old_state = l.StateGpu;
                if (state.Train)
                {
                    Utils.IncArray(ref l.StateGpu, ref l.StateGpuBackup, l.StateGpuIndex, l.StateGpuIndex += l.Hidden * l.Batch);
                }
                if (l.Shortcut)
                {
                    Blas.copy_ongpu(l.Hidden * l.Batch, old_state, l.StateGpu);
                }
                else
                {
                    Blas.fill_ongpu(l.Hidden * l.Batch, 0, l.StateGpu, 1);
                }
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, input_layer.OutputGpu, l.StateGpu);
                Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.OutputGpu, l.StateGpu);

                s.Input = l.StateGpu;
                forward_connected_layer_gpu(output_layer, s);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += l.Inputs * l.Batch);
                increment_layer(input_layer, 1);
                increment_layer(self_layer, 1);
                increment_layer(output_layer, 1);
            }
        }
        
        public static void backward_rnn_layer_gpu(Layer l, NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer input_layer = (l.InputLayer);
            Layer self_layer = (l.SelfLayer);
            Layer output_layer = (l.OutputLayer);
            increment_layer(input_layer, l.Steps - 1);
            increment_layer(self_layer, l.Steps - 1);
            increment_layer(output_layer, l.Steps - 1);

            Utils.IncArray(ref l.StateGpu, ref l.StateGpuBackup, l.StateGpuIndex, l.StateGpuIndex += l.Hidden * l.Batch * l.Steps);
            for (i = l.Steps - 1; i >= 0; --i)
            {

                s.Input = l.StateGpu;
                s.Delta = self_layer.DeltaGpu;
                backward_connected_layer_gpu(output_layer, s);

                Utils.DecArray(ref l.StateGpu, ref l.StateGpuBackup, l.StateGpuIndex, l.StateGpuIndex -= l.Hidden * l.Batch);

                Blas.copy_ongpu(l.Hidden * l.Batch, self_layer.DeltaGpu, input_layer.DeltaGpu);

                s.Input = l.StateGpu;
                s.Delta = new float[self_layer.DeltaGpu.Length + l.Hidden * l.Batch];
                Array.Copy(self_layer.DeltaGpuBackup, self_layer.DeltaGpuIndex - l.Hidden * l.Batch, s.Delta, 0, s.Delta.Length);
                if (i == 0) s.Delta = new float[0];
                backward_connected_layer_gpu(self_layer, s);

                if (i > 0 && l.Shortcut) Blas.axpy_ongpu(l.Hidden * l.Batch, 1, self_layer.DeltaGpu, self_layer.DeltaGpuBackup, 0, self_layer.DeltaGpuIndex - l.Hidden * l.Batch);
                s.Input = new float[state.Input.Length - i * l.Inputs * l.Batch];
                Array.Copy(state.Input, i * l.Inputs * l.Batch, s.Input, 0, s.Input.Length);
                if (state.Delta.Length != 0)
                {
                    s.Delta = new float[state.Delta.Length - i * l.Inputs * l.Batch];
                    Array.Copy(state.Delta, i * l.Inputs * l.Batch, s.Delta, 0, s.Delta.Length);
                }
                else s.Delta = new float[0];
                backward_connected_layer_gpu(input_layer, s);

                increment_layer(input_layer, -1);
                increment_layer(self_layer, -1);
                increment_layer(output_layer, -1);
            }
        }
    }
}