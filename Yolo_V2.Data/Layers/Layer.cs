using System;
using System.Linq;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    abstract public class Layer
    {
        public Layers LayerType;
        public Activation Activation;
        public CostType CostType;
        abstract public void Forward(ref NetworkState net);
        abstract public void Backward(ref NetworkState net);
        abstract public void Update(ref int i, ref float a, ref float b, ref float c);
        abstract public void UpdateGpu(ref int i, ref float a, ref float b, ref float c);
        abstract public void ForwardGpu(ref NetworkState net);
        abstract public void BackwardGpu(ref NetworkState net);
        public bool BatchNormalize;
        public bool Shortcut;
        public int Batch;
        public int Forced;
        public int Flipped;
        public int Inputs;
        public int Outputs;
        public int Truths;
        public int Height, Width, NumberOfChannels;
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
        public bool Random;
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
        public int StateIndex;
        public float[] State;
        public float[] StateBackup;
        public float[] PrevState;
        public float[] ForgotState;
        public float[] ForgotDelta;

        public float[] BinaryWeights;

        public int BiasesIndex;
        public float[] BiasesComplete;
        public float[] BiasUpdates;

        public float[] Scales;
        public float[] ScaleUpdates;

        public int WeightsIndex;
        public float[] WeightsComplete;
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
        //public cudnnTensorStruct SrcTensorDesc, DstTensorDesc;
        //public cudnnTensorStruct DsrcTensorDesc, DdstTensorDesc;
        //public cudnnFilterStruct WeightDesc;
        //public cudnnFilterStruct DweightDesc;
        //public cudnnConvolutionStruct ConvDesc;
        //public unsafe ConvolutionFwdAlgo* FwAlgo;
        //public unsafe ConvolutionBwdDataAlgo* BdAlgo;
        //public unsafe ConvolutionBwdFilterAlgo* BfAlgo;

        public int local_out_height()
        {
            int h = Height;
            if (Pad == 0) h -= Size;
            else h -= 1;
            return h / Stride + 1;
        }

        public int local_out_width()
        {
            int w = Width;
            if (Pad == 0) w -= Size;
            else w -= 1;
            return w / Stride + 1;
        }

        public static void backward_scale_cpu(float[] xNorm, float[] delta, int batch, int n, int size, float[] scaleUpdates)
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
                        sum += delta[index] * xNorm[index];
                    }
                }
                scaleUpdates[f] += sum;
            }
        }

        public static void mean_delta_cpu(float[] delta, float[] variance, int batch, int filters, int spatial, float[] meanDelta)
        {

            int i, j, k;
            for (i = 0; i < filters; ++i)
            {
                meanDelta[i] = 0;
                for (j = 0; j < batch; ++j)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + i * spatial + k;
                        meanDelta[i] += delta[index];
                    }
                }
                meanDelta[i] *= (-1.0f / (float)Math.Sqrt(variance[i] + .00001f));
            }
        }

        public static void variance_delta_cpu(float[] x, float[] delta, float[] mean, float[] variance, int batch, int filters, int spatial, float[] varianceDelta)
        {

            int i, j, k;
            for (i = 0; i < filters; ++i)
            {
                varianceDelta[i] = 0;
                for (j = 0; j < batch; ++j)
                {
                    for (k = 0; k < spatial; ++k)
                    {
                        int index = j * filters * spatial + i * spatial + k;
                        varianceDelta[i] += delta[index] * (x[index] - mean[i]);
                    }
                }
                varianceDelta[i] *= -.5f * (float)Math.Pow(variance[i] + .00001f, (-3.0f / 2.0f));
            }
        }

        public static void normalize_delta_cpu(float[] x, float[] mean, float[] variance, float[] meanDelta, float[] varianceDelta, int batch, int filters, int spatial, float[] delta)
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
                                       + varianceDelta[f] * 2.0f * (x[index] - mean[f]) / (spatial * batch)
                                       + meanDelta[f] / (spatial * batch);
                    }
                }
            }
        }

        public void swap_binary()
        {
            float[] swap = WeightsComplete;
            Array.Resize(ref WeightsComplete, BinaryWeights.Length + WeightsIndex);
            Array.Copy(BinaryWeights, 0, WeightsComplete, WeightsIndex, BinaryWeights.Length);
            BinaryWeights = new float[swap.Length - WeightsIndex];
            Array.Copy(swap, WeightsIndex, BinaryWeights, 0, BinaryWeights.Length);

            swap = WeightsGpu;
            WeightsGpu = BinaryWeightsGpu;
            BinaryWeightsGpu = swap;
        }

        public static void binarize_weights(float[] weights, int n, int size, float[] binary, int weightsStart = 0)
        {
            int i, f;
            for (f = 0; f < n; ++f)
            {
                float mean = 0;
                for (i = 0; i < size; ++i)
                {
                    mean += Math.Abs(weights[weightsStart + f * size + i]);
                }
                mean = mean / size;
                for (i = 0; i < size; ++i)
                {
                    binary[f * size + i] = (weights[weightsStart + f * size + i] > 0) ? mean : -mean;
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

        public ulong get_workspace_size()
        {
            return (ulong)(OutH * OutW * Size * Size * NumberOfChannels * sizeof(float));
        }

        public static void add_bias(float[] output, float[] biases, int batch, int n, int size, int biasesStart = 0)
        {
            int i, j, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    for (j = 0; j < size; ++j)
                    {
                        output[(b * n + i) * size + j] += biases[biasesStart + i];
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

        public static void backward_bias(float[] biasUpdates, float[] delta, int batch, int n, int size)
        {
            int i, b;
            for (b = 0; b < batch; ++b)
            {
                for (i = 0; i < n; ++i)
                {
                    var part = new float[size];
                    Array.Copy(delta, size * (i + b * n), part, 0, size);
                    biasUpdates[i] += part.Sum();
                }
            }
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

        public static void binarize_kernel(float[] x, int n, float[] binary)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            binary[i] = (x[i] >= 0) ? 1 : -1;
        }

        [GpuManaged]
        public void binarize_gpu(float[] x, int n,ref float[] binary)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var tempOutput = Gpu.Default.Allocate(binary);
            Gpu.Default.Launch(binarize_kernel, lp, x, n, tempOutput);
            binary = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

        public static void binarize_weights_kernel(float[] weights, int n, int size, float[] binary)
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

        public void binarize_weights_gpu(float[] weights, int n, int size,ref  float[] binary)
        {
            var lp = CudaUtils.cuda_gridsize(n);
            var tempOutput = Gpu.Default.Allocate(binary);
            Gpu.Default.Launch(binarize_weights_kernel, lp, weights, n, size, tempOutput);
            binary = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

        public static float get_pixel_kernel(float[] image, int w, int h, int x, int y, int c, int imageStart = 0)
        {
            if (x < 0 || x >= w || y < 0 || y >= h) return 0;
            return image[imageStart + x + w * (y + c * h)];
        }

        public static float3 rgb_to_hsv_kernel(float3 rgb)
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
            return new float3(h, s, v);
        }

        public static float3 hsv_to_rgb_kernel(float3 hsv)
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
            return new float3(r, g, b);
        }

        public static float bilinear_interpolate_kernel(float[] image, int w, int h, float x, float y, int c, int imageStart = 0)
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

        public static void levels_image_kernel(float[] image, float[] rand, int batch, int w, int h, bool train, float saturation, float exposure, float translate, float scale, float shift)
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
            float3 rgb = new float3(r, g, b);
            if (train)
            {
                float3 hsv = rgb_to_hsv_kernel(rgb);
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

        public void get_region_boxes(int w, int h, float thresh, ref float[][] probs, ref Box[] boxes, bool onlyObjectness, int[] map)
        {
            int i, j, n;
            float[] predictions = Output;
            for (i = 0; i < Width * Height; ++i)
            {
                int row = i / Width;
                int col = i % Width;
                for (n = 0; n < N; ++n)
                {
                    int index = i * N + n;
                    int pIndex = index * (Classes + 5) + 4;
                    float scale = predictions[pIndex];
                    if (Classfix == -1 && scale < .5) scale = 0;
                    int boxIndex = index * (Classes + 5);
                    boxes[index] = get_region_box(predictions, BiasesComplete, n, boxIndex, col, row, Width, Height, BiasesIndex);
                    boxes[index].X *= w;
                    boxes[index].Y *= h;
                    boxes[index].W *= w;
                    boxes[index].H *= h;

                    int classIndex = index * (Classes + 5) + 5;
                    if (SoftmaxTree != null)
                    {
                        SoftmaxTree.Hierarchy_predictions(predictions, classIndex, Classes, false);
                        bool found = false;
                        if (map.Length != 0)
                        {
                            for (j = 0; j < 200; ++j)
                            {
                                float prob = scale * predictions[classIndex + map[j]];
                                probs[index][j] = (prob > thresh) ? prob : 0;
                            }
                        }
                        else
                        {
                            for (j = Classes - 1; j >= 0; --j)
                            {
                                if (!found && predictions[classIndex + j] > .5)
                                {
                                    found = true;
                                }
                                else
                                {
                                    predictions[classIndex + j] = 0;
                                }
                                float prob = predictions[classIndex + j];
                                probs[index][j] = (scale > thresh) ? prob : 0;
                            }
                        }
                    }
                    else
                    {
                        for (j = 0; j < Classes; ++j)
                        {
                            float prob = scale * predictions[classIndex + j];
                            probs[index][j] = (prob > thresh) ? prob : 0;
                        }
                    }
                    if (onlyObjectness)
                    {
                        probs[index][0] = scale;
                    }
                }
            }
        }

        public Box get_region_box(float[] x, float[] biases, int n, int index, int i, int j, int w, int h, int biasesStart = 0)
        {
            Box b = new Box();
            b.X = (i + ActivationsHelper.Logistic_activate(x[index + 0])) / w;
            b.Y = (j + ActivationsHelper.Logistic_activate(x[index + 1])) / h;
            b.W = (float)Math.Exp(x[index + 2]) * biases[biasesStart + 2 * n];
            b.H = (float)Math.Exp(x[index + 3]) * biases[biasesStart + 2 * n + 1];

            b.W = (float)Math.Exp(x[index + 2]) * biases[biasesStart + 2 * n] / w;
            b.H = (float)Math.Exp(x[index + 3]) * biases[biasesStart + 2 * n + 1] / h;

            return b;
        }

        public void get_detection_boxes(int w, int h, float thresh, ref float[][] probs, ref Box[] boxes, bool onlyObjectness)
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
                    int pIndex = Side * Side * Classes + i * N + n;
                    float scale = predictions[pIndex];
                    int boxIndex = Side * Side * (Classes + N) + (i * N + n) * 4;
                    boxes[index].X = (predictions[boxIndex + 0] + col) / Side * w;
                    boxes[index].Y = (predictions[boxIndex + 1] + row) / Side * h;
                    boxes[index].W = (float)Math.Pow(predictions[boxIndex + 2], (Sqrt ? 2 : 1)) * w;
                    boxes[index].H = (float)Math.Pow(predictions[boxIndex + 3], (Sqrt ? 2 : 1)) * h;
                    for (j = 0; j < Classes; ++j)
                    {
                        int classIndex = i * Classes;
                        float prob = scale * predictions[classIndex + j];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                    if (onlyObjectness)
                    {
                        probs[index][0] = scale;
                    }
                }
            }
        }

        public void rgbgr_weights()
        {
            int i;
            for (i = 0; i < N; ++i)
            {
                Image im = get_convolutional_weight(i);
                if (im.NumberOfChannels == 3)
                {
                    //LoadArgs.rgbgr_image(im);
                }
            }
        }

        public void rescale_weights(float scale, float trans)
        {
            int i;
            for (i = 0; i < N; ++i)
            {
                Image im = get_convolutional_weight(i);
                if (im.NumberOfChannels == 3)
                {
                    LoadArgs.scale_image(im, scale);
                    float sum = im.Data.Sum();
                    BiasesComplete[BiasesIndex + i] += sum * trans;
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

        public Image get_convolutional_weight(int i)
        {
            int h = Size;
            int w = Size;
            int c = NumberOfChannels;
            var temp = new float[WeightsComplete.Length - WeightsIndex - i * h * w * c];
            Array.Copy(WeightsComplete, WeightsIndex + i * h * w * c, temp, 0, temp.Length);
            return new Image(w, h, c, temp);
        }

        public void pull_convolutional_layer()
        {
            Array.Copy(WeightsGpu, 0, WeightsComplete, WeightsIndex, NumberOfChannels * N * Size * Size);
            Array.Copy(BiasesGpu, 0, BiasesComplete, BiasesIndex, N);
            Array.Copy(WeightUpdatesGpu, 0, WeightUpdates, WeightsIndex, NumberOfChannels * N * Size * Size);
            Array.Copy(BiasUpdatesGpu, BiasUpdates, N);
            if (BatchNormalize)
            {
                Array.Copy(ScalesGpu, Scales, N);
                Array.Copy(RollingMeanGpu, RollingMean, N);
                Array.Copy(RollingVarianceGpu, RollingVariance, N);
            }
            if (Adam)
            {
                Array.Copy(MGpu, M, NumberOfChannels * N * Size * Size);
                Array.Copy(VGpu, V, NumberOfChannels * N * Size * Size);
            }
        }

        public void push_convolutional_layer()
        {
            Array.Copy(WeightsComplete, WeightsIndex, WeightsGpu, 0, NumberOfChannels * N * Size * Size);
            Array.Copy(BiasesComplete, BiasesIndex, BiasesGpu, 0, N);
            Array.Copy(WeightUpdates, WeightUpdatesGpu, NumberOfChannels * N * Size * Size);
            Array.Copy(BiasUpdates, BiasUpdatesGpu, N);
            if (BatchNormalize)
            {
                Array.Copy(Scales, ScalesGpu, N);
                Array.Copy(RollingMean, RollingMeanGpu, N);
                Array.Copy(RollingVariance, RollingVarianceGpu, N);
            }
            if (Adam)
            {
                Array.Copy(M, MGpu, NumberOfChannels * N * Size * Size);
                Array.Copy(V, VGpu, NumberOfChannels * N * Size * Size);
            }
        }

        public void denormalize_convolutional_layer()
        {
            int i, j;
            for (i = 0; i < N; ++i)
            {
                float scale = Scales[i] / (float)Math.Sqrt(RollingVariance[i] + .00001);
                for (j = 0; j < NumberOfChannels * Size * Size; ++j)
                {
                    WeightsComplete[WeightsIndex + i * NumberOfChannels * Size * Size + j] *= scale;
                }
                BiasesComplete[BiasesIndex + i] -= RollingMean[i] * scale;
                Scales[i] = 1;
                RollingMean[i] = 0;
                RollingVariance[i] = 1;
            }
        }

        public void denormalize_connected_layer()
        {
            int i, j;
            for (i = 0; i < Outputs; ++i)
            {
                float scale = Scales[i] / (float)Math.Sqrt(RollingVariance[i] + .000001);
                for (j = 0; j < Inputs; ++j)
                {
                    WeightsComplete[WeightsIndex + i * Inputs + j] *= scale;
                }
                BiasesComplete[BiasesIndex + i] -= RollingMean[i] * scale;
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
            Utils.print_statistics(BiasesComplete, Outputs, BiasesIndex);
            Console.Write($"Weights ");
            Utils.print_statistics(WeightsComplete, Outputs, WeightsIndex);
        }

        public void pull_connected_layer()
        {
            Array.Copy(WeightsGpu, 0, WeightsComplete, WeightsIndex, Inputs * Outputs);
            Array.Copy(BiasesGpu, BiasesComplete, Outputs);
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
            Array.Copy(WeightsComplete, WeightsIndex, WeightsGpu, 0, Inputs * Outputs);
            Array.Copy(BiasesComplete, BiasesIndex, BiasesGpu, 0, Outputs);
            Array.Copy(WeightUpdates, WeightUpdatesGpu, Inputs * Outputs);
            Array.Copy(BiasUpdates, BiasUpdatesGpu, Outputs);
            if (BatchNormalize)
            {
                Array.Copy(Scales, ScalesGpu, Outputs);
                Array.Copy(RollingMean, RollingMeanGpu, Outputs);
                Array.Copy(RollingVariance, RollingVarianceGpu, Outputs);
            }
        }

        public void pull_batchnorm_layer()
        {
            Array.Copy(ScalesGpu, Scales, NumberOfChannels);
            Array.Copy(RollingMeanGpu, RollingMean, NumberOfChannels);
            Array.Copy(RollingVarianceGpu, RollingVariance, NumberOfChannels);
        }

        public void push_batchnorm_layer()
        {
            Array.Copy(Scales, ScalesGpu, NumberOfChannels);
            Array.Copy(RollingMean, RollingMeanGpu, NumberOfChannels);
            Array.Copy(RollingVariance, RollingVarianceGpu, NumberOfChannels);
        }

        public void pull_local_layer()
        {
            int locations = OutW * OutH;
            int size = Size * Size * NumberOfChannels * N * locations;
            Array.Copy(WeightsGpu, 0, WeightsComplete, WeightsIndex, size);
            Array.Copy(BiasesGpu, 0, BiasesComplete, BiasesIndex, Outputs);
        }

        public void push_local_layer()
        {
            int locations = OutW * OutH;
            int size = Size * Size * NumberOfChannels * N * locations;
            Array.Copy(WeightsComplete, WeightsIndex, WeightsGpu, 0, size);
            Array.Copy(BiasesComplete, BiasesIndex, BiasesGpu, 0, Outputs);
        }

    }
}