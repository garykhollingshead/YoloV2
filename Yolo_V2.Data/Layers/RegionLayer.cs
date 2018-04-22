using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class RegionLayer: Layer
    {

        public RegionLayer(int batch, int width, int height, int n, int classes, int coords)
        {
            LayerType = Layers.Region;

            N = n;
            Batch = batch;
            Height = height;
            Width = width;
            Classes = classes;
            Coords = coords;
            Cost = 0;
            BiasesComplete = new float[n * 2];
            BiasesIndex = 0;
            BiasUpdates = new float[n * 2];
            Outputs = height * width * n * (classes + coords + 1);
            Inputs = Outputs;
            Truths = 30 * (5);
            Delta = new float[batch * Outputs];
            Output = new float[batch * Outputs];
            int i;
            for (i = 0; i < n * 2; ++i)
            {
                BiasesComplete[i] = .5f;
            }

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();

            Console.Error.Write($"detection\n");
        }

        public void resize_region_layer(int w, int h)
        {
            Width = w;
            Height = h;

            Outputs = h * w * N * (Classes + Coords + 1);
            Inputs = Outputs;

            Array.Resize(ref Output, Batch * Outputs * sizeof(float));
            Array.Resize(ref Delta, Batch * Outputs * sizeof(float));

            DeltaGpu = (float[])Delta.Clone();
            OutputGpu = (float[])Output.Clone();
        }

        public float delta_region_box(Box truth, float[] x, float[] biases, int n, int index, int i, int j, int w, int h, float[] delta, float scale, int biasesStart = 0)
        {
            Box pred = get_region_box(x, biases, n, index, i, j, w, h, biasesStart);
            float iou = Box.box_iou(pred, truth);

            float tx = (truth.X * w - i);
            float ty = (truth.Y * h - j);
            float tw = (float)Math.Log(truth.W / biases[biasesStart + 2 * n]);
            float th = (float)Math.Log(truth.H / biases[biasesStart + 2 * n + 1]);

            tw = (float)Math.Log(truth.W * w / biases[biasesStart + 2 * n]);
            th = (float)Math.Log(truth.H * h / biases[biasesStart + 2 * n + 1]);

            delta[index + 0] = scale * (tx - ActivationsHelper.Logistic_activate(x[index + 0])) * ActivationsHelper.Logistic_gradient(ActivationsHelper.Logistic_activate(x[index + 0]));
            delta[index + 1] = scale * (ty - ActivationsHelper.Logistic_activate(x[index + 1])) * ActivationsHelper.Logistic_gradient(ActivationsHelper.Logistic_activate(x[index + 1]));
            delta[index + 2] = scale * (tw - x[index + 2]);
            delta[index + 3] = scale * (th - x[index + 3]);
            return iou;
        }

        public void delta_region_class(float[] output, float[] delta, int index, int sclass, int classes, Tree hier, float scale, ref float avgCat)
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
                avgCat += pred;
            }
            else
            {
                for (n = 0; n < classes; ++n)
                {
                    delta[index + n] = scale * (((n == sclass) ? 1 : 0) - output[index + n]);
                    if (n == sclass) avgCat += output[index + n];
                }
            }
        }

        public override void Forward(ref NetworkState state)
        {
            int i, j, b, t, n;
            int size = Coords + Classes + 1;
            Array.Copy(Output, state.Input, Outputs * Batch);
            for (b = 0; b < Batch; ++b)
            {
                for (i = 0; i < Height * Width * N; ++i)
                {
                    int index = size * i + b * Outputs;
                    Output[index + 4] = ActivationsHelper.Logistic_activate(Output[index + 4]);
                }
            }

            if (!state.Train) return;
            Delta = new float[Outputs * Batch];
            float avgIou = 0;
            float recall = 0;
            float avgCat = 0;
            float avgObj = 0;
            float avgAnyobj = 0;
            int count = 0;
            int classCount = 0;
            (Cost) = 0;
            for (b = 0; b < Batch; ++b)
            {
                if (SoftmaxTree != null)
                {
                    bool onlyclass = false;
                    for (t = 0; t < 30; ++t)
                    {
                        Box truth = new Box(state.Truth, t * 5 + b * Truths);
                        if (truth.X == 0) break;
                        int sclass = (int)state.Truth[t * 5 + b * Truths + 4];
                        float maxp = 0;
                        int maxi = 0;
                        if (truth.X > 100000 && truth.Y > 100000)
                        {
                            for (n = 0; n < N * Width * Height; ++n)
                            {
                                int index = size * n + b * Outputs + 5;
                                float scale = Output[index - 1];
                                float p = scale * SoftmaxTree.Get_hierarchy_probability(Output, index, sclass);
                                if (p > maxp)
                                {
                                    maxp = p;
                                    maxi = n;
                                }
                            }
                            int index2 = size * maxi + b * Outputs + 5;
                            delta_region_class(Output, Delta, index2, sclass, Classes, SoftmaxTree, ClassScale, ref avgCat);
                            ++classCount;
                            onlyclass = true;
                            break;
                        }
                    }
                    if (onlyclass) continue;
                }
                for (j = 0; j < Height; ++j)
                {
                    for (i = 0; i < Width; ++i)
                    {
                        for (n = 0; n < N; ++n)
                        {
                            int index = size * (j * Width * N + i * N + n) + b * Outputs;
                            Box pred = get_region_box(Output, BiasesComplete, n, index, i, j, Width, Height, BiasesIndex);
                            float bestIou = 0;
                            int bestClass = -1;
                            for (t = 0; t < 30; ++t)
                            {
                                Box truth = new Box(state.Truth, t * 5 + b * Truths);
                                if (truth.X == 0) break;
                                float iou = Box.box_iou(pred, truth);
                                if (iou > bestIou)
                                {
                                    bestClass = (int)state.Truth[t * 5 + b * Truths + 4];
                                    bestIou = iou;
                                }
                            }
                            avgAnyobj += Output[index + 4];
                            Delta[index + 4] = NoobjectScale * ((0 - Output[index + 4]) * ActivationsHelper.Logistic_gradient(Output[index + 4]));
                            if (Classfix == -1) Delta[index + 4] = NoobjectScale * ((bestIou - Output[index + 4]) * ActivationsHelper.Logistic_gradient(Output[index + 4]));
                            else
                            {
                                if (bestIou > Thresh)
                                {
                                    Delta[index + 4] = 0;
                                    if (Classfix > 0)
                                    {
                                        delta_region_class(Output, Delta, index + 5, bestClass, Classes, SoftmaxTree, ClassScale * (Classfix == 2 ? Output[index + 4] : 1), ref avgCat);
                                        ++classCount;
                                    }
                                }
                            }

                            if ((state.Net.Seen) < 12800)
                            {
                                Box truth = new Box();
                                truth.X = (i + .5f) / Width;
                                truth.Y = (j + .5f) / Height;
                                truth.W = BiasesComplete[BiasesIndex + 2 * n];
                                truth.H = BiasesComplete[BiasesIndex + 2 * n + 1];

                                truth.W = BiasesComplete[BiasesIndex + 2 * n] / Width;
                                truth.H = BiasesComplete[BiasesIndex + 2 * n + 1] / Height;

                                delta_region_box(truth, Output, BiasesComplete, n, index, i, j, Width, Height, Delta, .01f, BiasesIndex);
                            }
                        }
                    }
                }
                for (t = 0; t < 30; ++t)
                {
                    Box truth = new Box(state.Truth, t * 5 + b * Truths);

                    if (truth.X == 0) break;
                    float bestIou = 0;
                    int bestIndex = 0;
                    int bestN = 0;
                    i = (int)(truth.X * Width);
                    j = (int)(truth.Y * Height);
                    Box truthShift = truth;
                    truthShift.X = 0;
                    truthShift.Y = 0;
                    for (n = 0; n < N; ++n)
                    {
                        int index = size * (j * Width * N + i * N + n) + b * Outputs;
                        Box pred = get_region_box(Output, BiasesComplete, n, index, i, j, Width, Height, BiasesIndex);
                        if (BiasMatch)
                        {
                            pred.W = BiasesComplete[BiasesIndex + 2 * n];
                            pred.H = BiasesComplete[BiasesIndex + 2 * n + 1];

                            pred.W = BiasesComplete[BiasesIndex + 2 * n] / Width;
                            pred.H = BiasesComplete[BiasesIndex + 2 * n + 1] / Height;

                        }
                        pred.X = 0;
                        pred.Y = 0;
                        float iouIn = Box.box_iou(pred, truthShift);
                        if (iouIn > bestIou)
                        {
                            bestIndex = index;
                            bestIou = iouIn;
                            bestN = n;
                        }
                    }
                    float iou = delta_region_box(truth, Output, BiasesComplete, bestN, bestIndex, i, j, Width, Height, Delta, CoordScale, BiasesIndex);
                    if (iou > .5) recall += 1;
                    avgIou += iou;

                    avgObj += Output[bestIndex + 4];
                    Delta[bestIndex + 4] = ObjectScale * (1 - Output[bestIndex + 4]) * ActivationsHelper.Logistic_gradient(Output[bestIndex + 4]);
                    if (Rescore)
                    {
                        Delta[bestIndex + 4] = ObjectScale * (iou - Output[bestIndex + 4]) * ActivationsHelper.Logistic_gradient(Output[bestIndex + 4]);
                    }


                    int sclass = (int)state.Truth[t * 5 + b * Truths + 4];
                    if (Map != null) sclass = Map[sclass];
                    delta_region_class(Output, Delta, bestIndex + 5, sclass, Classes, SoftmaxTree, ClassScale, ref avgCat);
                    ++count;
                    ++classCount;
                }
            }
            (Cost) = (float)Math.Pow(Utils.mag_array(Delta, Outputs * Batch), 2);
            Console.Write(
                $"Region Avg IOU: {avgIou / count}, Class: {avgCat / classCount}, Obj: {avgObj / count}, No Obj: {avgAnyobj / (Width * Height * N * Batch)}, Avg Recall: {recall / count},  count: {count}\n");
        }

        public override void Backward(ref NetworkState state)
        {
            Blas.Axpy_cpu(Batch * Inputs, 1, Delta, state.Delta);
        }

        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            return;
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            return;
            throw new NotImplementedException();
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            Blas.flatten_ongpu(state.Input, Height * Width, N * (Coords + Classes + 1), Batch, 1, ref OutputGpu);
            if (SoftmaxTree != null)
            {
                int i;
                int count = 5;
                for (i = 0; i < SoftmaxTree.Groups; ++i)
                {
                    int groupSize = SoftmaxTree.GroupSize[i];
                    Blas.softmax_gpu(OutputGpu, groupSize, Classes + 5, Width * Height * N * Batch, 1, ref OutputGpu, count, count);
                    count += groupSize;
                }
            }
            else if (Softmax)
            {
                Blas.softmax_gpu(OutputGpu, Classes, Classes + 5, Width * Height * N * Batch, 1, ref OutputGpu, 5, 5);
            }

            float[] inCpu = new float[Batch * Inputs];
            float[] truthCpu = new float[0];
            if (state.Truth.Length != 0)
            {
                int numTruth = Batch * Truths;
                truthCpu = new float[numTruth];
                Array.Copy(state.Truth, truthCpu, numTruth);
            }
            Array.Copy(OutputGpu, inCpu, Batch * Inputs);
            NetworkState cpuState = state;
            cpuState.Train = state.Train;
            cpuState.Truth = truthCpu;
            cpuState.Input = inCpu;
            Forward(ref cpuState);
            if (!state.Train) return;
            Array.Copy(Delta, DeltaGpu, Batch * Outputs);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            Blas.flatten_ongpu(DeltaGpu, Height * Width, N * (Coords + Classes + 1), Batch, 0, ref state.Delta);
        }

    }
}