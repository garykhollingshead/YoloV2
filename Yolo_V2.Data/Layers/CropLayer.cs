using System;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class CropLayer : Layer
    {
        public override void Backward(ref NetworkState state) { }
        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void BackwardGpu(ref NetworkState state) { }

        public CropLayer(int batch, int height, int width, int numberOfChannels, int cropHeight, int cropWidth, bool flip, float angle, float saturation, float exposure)
        {
            Console.Error.Write($"Crop Layer: {height} x {width} . {cropHeight} x {cropWidth} x {numberOfChannels} Image\n");
            LayerType = Layers.Crop;
            Batch = batch;
            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            Scale = (float)cropHeight / height;
            Flip = flip;
            Angle = angle;
            Saturation = saturation;
            Exposure = exposure;
            OutW = cropWidth;
            OutH = cropHeight;
            OutC = numberOfChannels;
            Inputs = Width * Height * NumberOfChannels;
            Outputs = OutW * OutH * OutC;
            Output = new float[Outputs * batch];
            OutputGpu = (float[])Output.Clone();
            RandGpu = new float[Batch * 8];
        }

        public void resize_crop_layer(int w, int h)
        {
            Width = w;
            Height = h;

            OutW = (int)(Scale * w);
            OutH = (int)(Scale * h);

            Inputs = Width * Height * NumberOfChannels;
            Outputs = OutH * OutW * OutC;

            Array.Resize(ref Output, Batch * Outputs);

            OutputGpu = (float[])Output.Clone();

        }

        public override void Forward(ref NetworkState state)
        {
            int i, j, c, b, row, col;
            int index;
            int count = 0;
            bool flip = (Flip && Utils.Rand.Next() % 2 != 0);
            int dh = Utils.Rand.Next() % (Height - OutH + 1);
            int dw = Utils.Rand.Next() % (Width - OutW + 1);
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
                dh = (Height - OutH) / 2;
                dw = (Width - OutW) / 2;
            }
            for (b = 0; b < Batch; ++b)
            {
                for (c = 0; c < NumberOfChannels; ++c)
                {
                    for (i = 0; i < OutH; ++i)
                    {
                        for (j = 0; j < OutW; ++j)
                        {
                            if (flip)
                            {
                                col = Width - dw - j - 1;
                            }
                            else
                            {
                                col = j + dw;
                            }
                            row = i + dh;
                            index = col + Width * (row + Height * (c + NumberOfChannels * b));
                            Output[count++] = state.Input[index] * scale + trans;
                        }
                    }
                }
            }
        }

        public void forward_crop_layer_kernel(float[] input, float[] rand, int size, int c, int h, int w, int cropHeight, int cropWidth, bool train, bool flip, float angle, float[] output)
        {
            int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (id >= size) return;

            float cx = w / 2.0f;
            float cy = h / 2.0f;

            int count = id;
            int j = id % cropWidth;
            id /= cropWidth;
            int i = id % cropHeight;
            id /= cropHeight;
            int k = id % c;
            id /= c;
            int b = id;

            float r4 = rand[8 * b + 4];
            float r5 = rand[8 * b + 5];
            float r6 = rand[8 * b + 6];
            float r7 = rand[8 * b + 7];

            float dw = (w - cropWidth) * r4;
            float dh = (h - cropHeight) * r5;
            flip = (flip && (r6 > .5));
            angle = 2 * angle * r7 - angle;
            if (!train)
            {
                dw = (w - cropWidth) / 2.0f;
                dh = (h - cropHeight) / 2.0f;
                flip = false;
                angle = 0;
            }

            float x = (flip) ? w - dw - j - 1 : j + dw;
            float y = i + dh;

            float rx = (float)(Math.Cos(angle) * (x - cx) - Math.Sin(angle) * (y - cy) + cx);
            float ry = (float)(Math.Sin(angle) * (x - cx) + Math.Cos(angle) * (y - cy) + cy);

            output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k, w * h * c * b);
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            CudaUtils.cuda_random(RandGpu, (ulong)Batch * 8);

            float radians = Angle * 3.14159265f / 180.0f;

            float scale = 2;
            float translate = -1;
            if (Noadjust)
            {
                scale = 1;
                translate = 0;
            }

            int size = Batch * Width * Height;

            var lp = CudaUtils.cuda_gridsize(size);
            var tempOutput = Gpu.Default.Allocate(state.Input);
            Gpu.Default.Launch(levels_image_kernel, lp, tempOutput, RandGpu, Batch, Width, Height, state.Train, Saturation, Exposure, translate, scale, Shift);
            state.Input = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);

            size = Batch * NumberOfChannels * OutW * OutH;

            lp = CudaUtils.cuda_gridsize(size);
            tempOutput = Gpu.Default.Allocate(OutputGpu);
            Gpu.Default.Launch(forward_crop_layer_kernel, lp, state.Input, RandGpu, size, NumberOfChannels, Height, Width, OutH, OutW, state.Train, Flip, radians, tempOutput);
            OutputGpu = Gpu.CopyToHost(tempOutput);
            Gpu.Free(tempOutput);
        }

    }
}