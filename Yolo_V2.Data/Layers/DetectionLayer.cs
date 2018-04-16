using System;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class DetectionLayer:Layer
    {
        public DetectionLayer(int batch, int inputs, int n, int side, int classes, int coords, bool rescore)
        {
            LayerType = Layers.Detection;

            N = n;
            Batch = batch;
            Inputs = inputs;
            Classes = classes;
            Coords = coords;
            Rescore = rescore;
            Side = side;
            Width = side;
            Height = side;
            Cost = 0;
            Outputs = Inputs;
            Truths = Side * Side * (1 + Coords + Classes);
            Output = new float[batch * Outputs];
            Delta = new float[batch * Outputs];

            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();

            Console.Error.Write($"Detection Layer\n");
        }

        public override void Forward(ref NetworkState state)
        {
            int locations = Side * Side;
            int i, j;
            Array.Copy(state.Input, 0, Output, 0, Outputs * Batch);
            int b;
            if (Softmax)
            {
                for (b = 0; b < Batch; ++b)
                {
                    int index = b * Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int offset = i * Classes;
                        Blas.Softmax(Output, Classes, 1,
                            Output, index + offset, index + offset);
                    }
                }
            }
            if (state.Train)
            {
                float avgIou = 0;
                float avgCat = 0;
                float avgAllcat = 0;
                float avgObj = 0;
                float avgAnyobj = 0;
                int count = 0;
                (Cost) = null;
                int size = Inputs * Batch;
                Delta = new float[size];
                for (b = 0; b < Batch; ++b)
                {
                    int index = b * Inputs;
                    for (i = 0; i < locations; ++i)
                    {
                        int truthIndex = (b * locations + i) * (1 + Coords + Classes);
                        bool isObj = state.Truth[truthIndex] != 0;
                        for (j = 0; j < N; ++j)
                        {
                            int pnIndex = index + locations * Classes + i * N + j;
                            Delta[pnIndex] = NoobjectScale * (0 - Output[pnIndex]);
                            (Cost) += NoobjectScale * (float)Math.Pow(Output[pnIndex], 2);
                            avgAnyobj += Output[pnIndex];
                        }

                        int bestIndex = -1;
                        float bestIou = 0;
                        float bestRmse = 20;

                        if (!isObj)
                        {
                            continue;
                        }

                        int classIndex = index + i * Classes;
                        for (j = 0; j < Classes; ++j)
                        {
                            Delta[classIndex + j] = ClassScale * (state.Truth[truthIndex + 1 + j] - Output[classIndex + j]);
                            (Cost) += ClassScale * (float)Math.Pow(state.Truth[truthIndex + 1 + j] - Output[classIndex + j], 2);
                            if (state.Truth[truthIndex + 1 + j] != 0) avgCat += Output[classIndex + j];
                            avgAllcat += Output[classIndex + j];
                        }

                        Box truth = new Box(state.Truth, truthIndex + 1 + Classes);
                        truth.X /= Side;
                        truth.Y /= Side;

                        for (j = 0; j < N; ++j)
                        {
                            int boxIndex = index + locations * (Classes + N) + (i * N + j) * Coords;
                            Box outputout = new Box(Output, boxIndex);
                            outputout.X /= Side;
                            outputout.Y /= Side;

                            if (Sqrt)
                            {
                                outputout.W = outputout.W * outputout.W;
                                outputout.H = outputout.H * outputout.H;
                            }

                            float iou = Box.box_iou(outputout, truth);
                            float rmse = Box.box_rmse(outputout, truth);
                            if (bestIou > 0 || iou > 0)
                            {
                                if (iou > bestIou)
                                {
                                    bestIou = iou;
                                    bestIndex = j;
                                }
                            }
                            else
                            {
                                if (rmse < bestRmse)
                                {
                                    bestRmse = rmse;
                                    bestIndex = j;
                                }
                            }
                        }

                        if (Forced != 0)
                        {
                            bestIndex = truth.W * truth.H < .1 ? 1 : 0;
                        }
                        if (Random && state.Net.Seen < 64000)
                        {
                            bestIndex = Utils.Rand.Next() % N;
                        }

                        int boxIndex2 = index + locations * (Classes + N) + (i * N + bestIndex) * Coords;
                        int tboxIndex = truthIndex + 1 + Classes;

                        Box outputout2 = new Box(Output, boxIndex2);
                        outputout2.X /= Side;
                        outputout2.Y /= Side;
                        if (Sqrt)
                        {
                            outputout2.W = outputout2.W * outputout2.W;
                            outputout2.H = outputout2.H * outputout2.H;
                        }
                        float iou2 = Box.box_iou(outputout2, truth);

                        int pIndex = index + locations * Classes + i * N + bestIndex;
                        (Cost) -= NoobjectScale * (float)Math.Pow(Output[pIndex], 2);
                        (Cost) += ObjectScale * (float)Math.Pow(1 - Output[pIndex], 2);
                        avgObj += Output[pIndex];
                        Delta[pIndex] = ObjectScale * (1.0f - Output[pIndex]);

                        if (Rescore)
                        {
                            Delta[pIndex] = ObjectScale * (iou2 - Output[pIndex]);
                        }

                        Delta[boxIndex2 + 0] = CoordScale * (state.Truth[tboxIndex + 0] - Output[boxIndex2 + 0]);
                        Delta[boxIndex2 + 1] = CoordScale * (state.Truth[tboxIndex + 1] - Output[boxIndex2 + 1]);
                        Delta[boxIndex2 + 2] = CoordScale * (state.Truth[tboxIndex + 2] - Output[boxIndex2 + 2]);
                        Delta[boxIndex2 + 3] = CoordScale * (state.Truth[tboxIndex + 3] - Output[boxIndex2 + 3]);
                        if (Sqrt)
                        {
                            Delta[boxIndex2 + 2] = CoordScale * ((float)Math.Sqrt(state.Truth[tboxIndex + 2]) - Output[boxIndex2 + 2]);
                            Delta[boxIndex2 + 3] = CoordScale * ((float)Math.Sqrt(state.Truth[tboxIndex + 3]) - Output[boxIndex2 + 3]);
                        }

                        (Cost) += (float)Math.Pow(1 - iou2, 2);
                        avgIou += iou2;
                        ++count;
                    }
                }
                (Cost) = (float)Math.Pow(Utils.mag_array(Delta, Outputs * Batch), 2);


                Console.Write(
                    $"Detection Avg IOU: {avgIou / count}, Pos Cat: {avgCat / count}, All Cat: {avgAllcat / (count * Classes)}, Pos Obj: {avgObj / count}, Any Obj: {avgAnyobj / (Batch * locations * N)}, count: {count}\n");
            }
        }

        public override void Backward(ref NetworkState state)
        {
            Blas.Axpy_cpu(Batch * Inputs, 1, Delta, state.Delta);
        }

        public override void Update(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void UpdateGpu(ref int i, ref float a, ref float b, ref float c)
        {
            throw new NotImplementedException();
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            if (!state.Train)
            {
                Blas.copy_ongpu(Batch * Inputs, state.Input, ref OutputGpu);
                return;
            }

            float[] inCpu = new float[Batch * Inputs];
            float[] truthCpu = null;
            if (state.Truth.Any())
            {
                int numTruth = Batch * Side * Side * (1 + Coords + Classes);
                truthCpu = new float[numTruth];
                Array.Copy(state.Truth, truthCpu, numTruth);
            }
            Array.Copy(state.Input, inCpu, Batch * Inputs);
            NetworkState cpuState = state;
            cpuState.Train = state.Train;
            cpuState.Truth = truthCpu;
            cpuState.Input = inCpu;
            Forward(ref cpuState);
            Array.Copy(Output, OutputGpu, Batch * Outputs);
            Array.Copy(Delta, DeltaGpu, Batch * Inputs);
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            Blas.axpy_ongpu(Batch * Inputs, 1, DeltaGpu, state.Delta);
        }

    }
}