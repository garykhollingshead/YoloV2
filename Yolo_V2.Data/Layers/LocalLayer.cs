using System;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class LocalLayer:Layer
    {

        public LocalLayer(int batch, int height, int width, int numberOfChannels, int n, int size, int stride, int pad, Activation activation)
        {
            int i;

            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            N = n;
            Batch = batch;
            Stride = stride;
            Size = size;
            Pad = pad;

            int outH = local_out_height();
            int outW = local_out_width();
            int locations = outH * outW;
            OutH = outH;
            OutW = outW;
            OutC = n;
            Outputs = OutH * OutW * OutC;
            Inputs = Width * Height * NumberOfChannels;

            WeightsComplete = new float[numberOfChannels * n * size * size * locations];
            WeightsIndex = 0;
            WeightUpdates = new float[numberOfChannels * n * size * size * locations];

            BiasesComplete = new float[Outputs];
            BiasesIndex = 0;
            BiasUpdates = new float[Outputs];

            float scale = (float)Math.Sqrt(2.0f / (size * size * numberOfChannels));
            for (i = 0; i < numberOfChannels * n * size * size; ++i) WeightsComplete[i] = scale * Utils.rand_uniform(-1, 1);

            ColImage = new float[outH * outW * size * size * numberOfChannels];
            Output = new float[Batch * outH * outW * n];
            Delta = new float[Batch * outH * outW * n];

            WeightsGpu = (float[])WeightsComplete.Clone();
            WeightUpdatesGpu = (float[])WeightUpdates.Clone();

            BiasesGpu = (float[])BiasesComplete.Clone();
            BiasUpdatesGpu = (float[])BiasUpdates.Clone();

            ColImageGpu = (float[])ColImage.Clone();
            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();

            Activation = activation;

            Console.Error.WriteLine($"Local Layer: {height} x {width} x {numberOfChannels} Image, {n} filters . {outH} x {outW} x {n} Image");
        }

        public override void Forward(ref NetworkState state)
        {
            int outH = local_out_height();
            int outW = local_out_width();
            int i, j;
            int locations = outH * outW;

            for (i = 0; i < Batch; ++i)
            {
                var index = i * Outputs;
                var output = new float[Outputs];
                Blas.Copy_cpu(Outputs, BiasesComplete, output, BiasesIndex, index);
            }

            for (i = 0; i < Batch; ++i)
            {
                var index = i * Width * Height * NumberOfChannels;
                Im2Col.im2col_cpu(state.Input, NumberOfChannels, Height, Width, Size, Stride, Pad, ColImage, index);

                index = i * Outputs;
                float[] output = new float[Output.Length - index];
                Array.Copy(Output, index, output, 0, output.Length);

                for (j = 0; j < locations; ++j)
                {
                    index = j * Size * Size * NumberOfChannels * N;
                    float[] a = new float[WeightsComplete.Length - WeightsIndex - index];
                    float[] b = new float[ColImage.Length - j];
                    float[] c = new float[output.Length - j];
                    Array.Copy(WeightsComplete, WeightsIndex + index, a, 0, a.Length);
                    Array.Copy(ColImage, j, b, 0, b.Length);
                    Array.Copy(output, j, c, 0, c.Length);

                    int m = N;
                    int n = 1;
                    int k = Size * Size * NumberOfChannels;

                    GemmUtils.Gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
                    Array.Copy(c, 0, output, j, c.Length);
                }
                Array.Copy(output, 0, Output, index, output.Length);
            }
            ActivationsHelper.Activate_array(Output, Outputs * Batch, Activation);
        }

        public override void Backward(ref NetworkState state)
        {
            int i, j;
            int locations = OutW * OutH;

            ActivationsHelper.Gradient_array(Output, Outputs * Batch, Activation, Delta);

            for (i = 0; i < Batch; ++i)
            {
                var index = i * Outputs;
                Blas.Axpy_cpu(Outputs, 1, Delta, BiasUpdates, index);
            }

            for (i = 0; i < Batch; ++i)
            {
                var index = i * Width * Height * NumberOfChannels;
                float[] input = new float[state.Input.Length - index];
                Array.Copy(state.Input, index, input, 0, input.Length);
                Im2Col.im2col_cpu(input, NumberOfChannels, Height, Width,
                    Size, Stride, Pad, ColImage);

                for (j = 0; j < locations; ++j)
                {
                    var indexA = i * Outputs + j;
                    var indexC = j * Size * Size * NumberOfChannels * N;

                    float[] a = new float[Delta.Length - indexA];
                    Array.Copy(Delta, indexA, a, 0, a.Length);

                    float[] b = new float[ColImage.Length - j];
                    Array.Copy(ColImage, j, b, 0, b.Length);

                    float[] c = new float[WeightUpdates.Length - indexC];
                    Array.Copy(WeightUpdates, indexC, c, 0, c.Length);

                    int m = N;
                    int n = Size * Size * NumberOfChannels;
                    int k = 1;

                    GemmUtils.Gemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);

                    Array.Copy(c, 0, WeightUpdates, indexC, c.Length);
                }

                if (state.Delta.Any())
                {
                    for (j = 0; j < locations; ++j)
                    {
                        var indexA = j * Size * Size * NumberOfChannels * N;
                        var indexB = i * Outputs + j;

                        float[] a = new float[WeightsComplete.Length - WeightsIndex - indexA];
                        Array.Copy(WeightsComplete, WeightsIndex + indexA, a, 0, a.Length);

                        float[] b = new float[Delta.Length - indexB];
                        Array.Copy(Delta, indexB, b, 0, b.Length);

                        float[] c = new float[ColImage.Length - j];
                        Array.Copy(ColImage, j, c, 0, c.Length);

                        int m = Size * Size * NumberOfChannels;
                        int n = 1;
                        int k = N;

                        GemmUtils.Gemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);

                        Array.Copy(c, 0, ColImage, j, c.Length);
                    }

                    index = i * NumberOfChannels * Height * Width;
                    Im2Col.col2im_cpu(ColImage, NumberOfChannels, Height, Width, Size, Stride, Pad, state.Delta, index);
                }
            }
        }

        public override void Update(ref int batch,ref float learningRate,ref float momentum,ref float decay)
        {
            int locations = OutW * OutH;
            int size = Size * Size * NumberOfChannels * N * locations;
            Blas.Axpy_cpu(Outputs, learningRate / batch, BiasUpdates, BiasesComplete, 0, BiasesIndex);
            Blas.Scal_cpu(Outputs, momentum, BiasUpdates, 1);

            Blas.Axpy_cpu(size, -decay * batch, WeightsComplete, WeightUpdates, WeightsIndex);
            Blas.Axpy_cpu(size, learningRate / batch, WeightUpdates, WeightsComplete, 0, WeightsIndex);
            Blas.Scal_cpu(size, momentum, WeightUpdates, 1);
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            int outH = local_out_height();
            int outW = local_out_width();
            int i, j;
            int locations = outH * outW;


            for (i = 0; i < Batch; ++i)
            {
                var inIndex = i * Width * Height * NumberOfChannels;
                float[] input = new float[state.Input.Length - inIndex];
                Array.Copy(state.Input, inIndex, input, 0, input.Length);
                Im2Col.im2col_ongpu(input, NumberOfChannels, Height, Width,
                    Size, Stride, Pad, ref ColImageGpu);
                var outIndex = i * Outputs;
                float[] output = new float[OutputGpu.Length - outIndex];
                Array.Copy(OutputGpu, outIndex, output, 0, output.Length);
                for (j = 0; j < locations; ++j)
                {
                    var aIndex = j * Size * Size * NumberOfChannels * N;

                    float[] a = new float[WeightsGpu.Length - aIndex];
                    Array.Copy(WeightsGpu, aIndex, a, 0, a.Length);

                    float[] b = new float[Delta.Length - j];
                    Array.Copy(Delta, j, b, 0, b.Length);

                    float[] c = new float[output.Length - j];
                    Array.Copy(output, j, c, 0, c.Length);

                    int m = N;
                    int n = 1;
                    int k = Size * Size * NumberOfChannels;

                    GemmUtils.gemm_ongpu(0, 0, m, n, k, 1, a, k, b, locations, 1, ref c, locations);

                    Array.Copy(c, 0, output, j, c.Length);
                }
                Array.Copy(output, 0, OutputGpu, outIndex, output.Length);
            }
            ActivationsHelper.activate_array_ongpu(ref OutputGpu, Outputs * Batch, Activation);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            int i, j;
            int locations = OutW * OutH;

            ActivationsHelper.gradient_array_ongpu(OutputGpu, Outputs * Batch, Activation, ref DeltaGpu);
            for (i = 0; i < Batch; ++i)
            {
                Blas.axpy_ongpu(Outputs, 1, DeltaGpu, BiasUpdatesGpu, i * Outputs);
            }

            for (i = 0; i < Batch; ++i)
            {
                int index = i * Width * Height * NumberOfChannels;
                float[] input = new float[state.Input.Length - index];
                Array.Copy(state.Input, index, input, 0, input.Length);
                Im2Col.im2col_ongpu(input, NumberOfChannels, Height, Width,
                    Size, Stride, Pad, ref ColImageGpu);
                for (j = 0; j < locations; ++j)
                {
                    int aIndex = i * Outputs + j;
                    int cIndex = j * Size * Size * NumberOfChannels * N;

                    float[] a = new float[DeltaGpu.Length - aIndex];
                    Array.Copy(DeltaGpu, aIndex, a, 0, a.Length);

                    float[] b = new float[ColImageGpu.Length - j];
                    Array.Copy(ColImageGpu, j, b, 0, b.Length);

                    float[] c = new float[WeightUpdatesGpu.Length - cIndex];
                    Array.Copy(WeightUpdatesGpu, cIndex, c, 0, c.Length);

                    int m = N;
                    int n = Size * Size * NumberOfChannels;
                    int k = 1;

                    GemmUtils.gemm_ongpu(0, 1, m, n, k, 1, a, locations, b, locations, 1, ref c, n);
                    Array.Copy(c, 0, WeightUpdatesGpu, cIndex, c.Length);
                }

                if (state.Delta.Any())
                {
                    for (j = 0; j < locations; ++j)
                    {
                        int aIndex = j * Size * Size * NumberOfChannels * N;
                        int bIndex = i * Outputs + j;

                        float[] a = new float[WeightsGpu.Length - aIndex];
                        Array.Copy(WeightsGpu, aIndex, a, 0, a.Length);

                        float[] b = new float[Delta.Length - bIndex];
                        Array.Copy(Delta, bIndex, b, 0, b.Length);

                        float[] c = new float[DeltaGpu.Length - j];
                        Array.Copy(DeltaGpu, j, c, 0, c.Length);

                        int m = Size * Size * NumberOfChannels;
                        int n = 1;
                        int k = N;

                        GemmUtils.gemm_ongpu(1, 0, m, n, k, 1, a, m, b, locations, 0, ref c, locations);
                        Array.Copy(c, 0, DeltaGpu, j, c.Length);
                    }

                    var dIndex = i * NumberOfChannels * Height * Width;
                    Im2Col.col2im_ongpu(ColImageGpu, NumberOfChannels, Height, Width, Size, Stride, Pad, state.Delta, dIndex);
                }
            }
        }

        public override void UpdateGpu(ref int batch,ref float learningRate,ref float momentum,ref float decay)
        {
            int locations = OutW * OutH;
            int size = Size * Size * NumberOfChannels * N * locations;
            Blas.axpy_ongpu(Outputs, learningRate / batch, BiasUpdatesGpu, BiasesGpu);
            Blas.scal_ongpu(Outputs, momentum, ref BiasUpdatesGpu, 1);

            Blas.axpy_ongpu(size, -decay * batch, WeightsGpu, WeightUpdatesGpu);
            Blas.axpy_ongpu(size, learningRate / batch, WeightUpdatesGpu, WeightsGpu);
            Blas.scal_ongpu(size, momentum, ref WeightUpdatesGpu, 1);
        }

    }
}