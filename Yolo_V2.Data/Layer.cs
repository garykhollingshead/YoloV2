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

            Console.Error.WriteLine($"Local Layer: {h} x {w} x {c} image, {n} filters -> {out_h} x {out_w} x {n} image");
        }

        public static Layer make_batchnorm_layer(int batch, int w, int h, int c)
        {
            Console.Error.Write($"Batch Normalization Layer: {w} x {h} x {c} image\n");
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
                    Im2Col.col2im_cpu(ColImage, C, H, W, Size, Stride, Pad, output);
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
            //    copy_ongpu(Outputs, BiasesGpu, 1, Output.Gpu + i * Outputs, 1);
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
    }
}