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
        public LayerType Type;
        public Activation Activation;
        public CostType CostType;
        public Action<Layer, NetworkState> Forward;
        public Action<Layer, NetworkState> Backward;
        public Action<Layer, int, float, float, float> Update;
        public Action<Layer, NetworkState> ForwardGpu;
        public Action<Layer, NetworkState> BackwardGpu;
        public Action<Layer, int, float, float, float> UpdateGpu;
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

        public Tree[] SoftmaxTree;
        public List<int> Map;

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

        public int Dontload;
        public int Dontloadscales;

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

        int local_out_height(Layer l)
        {
            int h = l.H;
            if (l.Pad == 0) h -= l.Size;
            else h -= 1;
            return h / l.Stride + 1;
        }

        int local_out_width(Layer l)
        {
            int w = l.W;
            if (l.Pad == 0) w -= l.Size;
            else w -= 1;
            return w / l.Stride + 1;
        }

        Layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, Activation activation)
        {
            int i;
            Layer l = new Layer();
            l.Type = LayerType.Local;

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

            float scale = (float)Math.Sqrt(2.0f/ (size * size * c));
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

            l.WeightsGpu = l.Weights.ToArray();
            l.WeightUpdatesGpu = l.WeightUpdates.ToArray();

            l.BiasesGpu = l.Biases.ToArray();
            l.BiasUpdatesGpu = l.BiasUpdates.ToArray();

            l.ColImageGpu = l.ColImage.ToArray();
            l.DeltaGpu = l.Delta.ToArray();
            l.OutputGpu = l.Output.ToArray();

            l.Activation = activation;

            Console.Error.WriteLine($"Local Layer: {h} x {w} x {c} image, {n} filters -> {out_h} x {out_w} x {n} image");

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
                var index = i * l.Outputs;
                var output = l.Output.Skip(index).ToArray();
                Blas.Copy_cpu(l.Outputs, l.Biases, 1, output, 1);
                CombineLists(l.Output, index, output);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                var index = i * l.W * l.H * l.C;
                float[] input = state.Input.Skip(index).ToArray();
                Im2Col.im2col_cpu(input, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, l.ColImage);

                index = i * l.Outputs;
                float[] output = l.Output.Skip(index).ToArray();

                for (j = 0; j < locations; ++j)
                {
                    index = j * l.Size * l.Size * l.C * l.N;
                    float[] a = l.Weights.Skip(index).ToArray();
                    float[] b = l.ColImage.Skip(j).ToArray();
                    float[] c = output.Skip(j).ToArray();

                    int m = l.N;
                    int n = 1;
                    int k = l.Size * l.Size * l.C;

                    Gemm.gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                    CombineLists(l.Weights, index, a);
                    CombineLists(l.ColImage, j, b);
                    CombineLists(output, j, c);
                }
            }
            ActivationsHelper.Activate_array(l.Output, l.Outputs * l.Batch, l.Activation);
        }

        void backward_local_layer(Layer l, NetworkState state)
        {
            int i, j;
            int locations = l.OutW * l.OutH;

            ActivationsHelper.Gradient_array(l.Output, l.Outputs * l.Batch, l.Activation, l.Delta);

            for (i = 0; i < l.Batch; ++i)
            {
                var index = i * l.Outputs;
                Blas.Axpy_cpu(l.Outputs, 1, l.Delta.Skip(index).ToArray(), 1, l.BiasUpdates, 1);
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

            //for (i = 0; i < l.Batch; ++i)
            //{
            //    copy_ongpu(l.Outputs, l.BiasesGpu, 1, l.Output.Gpu + i * l.Outputs, 1);
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

        void backward_local_layer_gpu(Layer l, NetworkState state)
        {
            int i, j;
            int locations = l.OutW * l.OutH;

            ActivationsHelper.gradient_array_ongpu(l.OutputGpu, l.Outputs * l.Batch, l.Activation, l.DeltaGpu);
            for (i = 0; i < l.Batch; ++i)
            {
                var tmp = l.DeltaGpu.Skip(i * l.Outputs).ToArray();
                Blas.Axpy_gpu(l.Outputs, 1, tmp, 1, l.BiasUpdatesGpu, 1);
            }

            for (i = 0; i < l.Batch; ++i)
            {
                float[] input = state.Input + i * l.W * l.H * l.C;
                im2col_ongpu(input, l.C, l.H, l.W,
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

                        gemm_ongpu(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
                    }

                    col2im_ongpu(l.ColImageGpu, l.C, l.H, l.W, l.Size, l.Stride, l.Pad, state.Delta + i * l.C * l.H * l.W);
                }
            }
        }

        void update_local_layer_gpu(Layer l, int batch, float learning_rate, float momentum, float decay)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            axpy_ongpu(l.Outputs, learning_rate / batch, l.BiasUpdatesGpu, 1, l.BiasesGpu, 1);
            scal_ongpu(l.Outputs, momentum, l.BiasUpdatesGpu, 1);

            axpy_ongpu(size, -decay * batch, l.WeightsGpu, 1, l.WeightUpdatesGpu, 1);
            axpy_ongpu(size, learning_rate / batch, l.WeightUpdatesGpu, 1, l.WeightsGpu, 1);
            scal_ongpu(size, momentum, l.WeightUpdatesGpu, 1);
        }

        void pull_local_layer(Layer l)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            cuda_pull_array(l.WeightsGpu, l.Weights, size);
            cuda_pull_array(l.BiasesGpu, l.Biases, l.Outputs);
        }

        void push_local_layer(Layer l)
        {
            int locations = l.OutW * l.OutH;
            int size = l.Size * l.Size * l.C * l.N * locations;
            cuda_push_array(l.WeightsGpu, l.Weights, size);
            cuda_push_array(l.BiasesGpu, l.Biases, l.Outputs);
        }
    }
}