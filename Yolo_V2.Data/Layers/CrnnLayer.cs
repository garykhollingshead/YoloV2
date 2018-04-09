using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class CrnnLayer:Layer
    {
        public new ConvolutionalLayer InputLayer;
        public new ConvolutionalLayer SelfLayer;
        public new ConvolutionalLayer OutputLayer;

        public CrnnLayer(int batch, int height, int width, int numberOfChannels, int hiddenFilters, int outputFilters, int steps, Activation activation, bool batchNormalize)
        {
            Console.Error.Write($"Layers.Crnn Layer: {height} x {width} x {numberOfChannels} Image, {outputFilters} filters\n");
            batch = batch / steps;
            Batch = batch;
            LayerType = Layers.Crnn;
            Steps = steps;
            Height = height;
            Width = width;
            NumberOfChannels = numberOfChannels;
            OutH = height;
            OutW = width;
            OutC = outputFilters;
            Inputs = height * width * numberOfChannels;
            Hidden = height * width * hiddenFilters;
            Outputs = OutH * OutW * OutC;

            State = new float[Hidden * batch * (steps + 1)];

            Console.Error.Write($"\t\t");
            (InputLayer) = new ConvolutionalLayer(batch * steps, height, width, numberOfChannels, hiddenFilters, 3, 1, 1, activation, batchNormalize, false, false, false);
            InputLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (SelfLayer) = new ConvolutionalLayer(batch * steps, height, width, hiddenFilters, hiddenFilters, 3, 1, 1, activation, batchNormalize, false, false, false);
            SelfLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (OutputLayer) = new ConvolutionalLayer(batch * steps, height, width, hiddenFilters, outputFilters, 3, 1, 1, activation, batchNormalize, false, false, false);
            OutputLayer.Batch = batch;

            Output = OutputLayer.Output;
            Delta = OutputLayer.Delta;


            StateGpu = (float[])State.Clone();
            OutputGpu = OutputLayer.OutputGpu;
            DeltaGpu = OutputLayer.DeltaGpu;
        }

        public override void Update(ref int batch, ref float learningRate,ref  float momentum, ref float decay)
        {
            InputLayer.Update(ref batch, ref learningRate,ref momentum,ref decay);
            SelfLayer.Update(ref batch,ref learningRate,ref momentum,ref decay);
            OutputLayer.Update(ref batch,ref learningRate,ref momentum,ref decay);
        }

        public override void Forward(ref NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;

            Blas.Fill_cpu(Outputs * Batch * Steps, 0, OutputLayer.Delta, 1);
            Blas.Fill_cpu(Hidden * Batch * Steps, 0, SelfLayer.Delta, 1);
            Blas.Fill_cpu(Hidden * Batch * Steps, 0, InputLayer.Delta, 1);
            if (state.Train) Blas.Fill_cpu(Hidden * Batch, 0, State, 1);

            for (i = 0; i < Steps; ++i)
            {
                s.Input = state.Input;
                InputLayer.Forward(ref s);

                s.Input = State;
                SelfLayer.Forward(ref s);

                float[] oldState = State;
                if (state.Train)
                {
                    Utils.IncArray(ref State, ref StateBackup, StateIndex, StateIndex += Hidden * Batch);
                }
                if (Shortcut)
                {
                    Blas.Copy_cpu(Hidden * Batch, oldState, State);
                }
                else
                {
                    Blas.Fill_cpu(Hidden * Batch, 0, State, 1);
                }
                Blas.Axpy_cpu(Hidden * Batch, 1, InputLayer.Output, State);
                Blas.Axpy_cpu(Hidden * Batch, 1, SelfLayer.Output, State);

                s.Input = State;
                OutputLayer.Forward(ref s);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += Inputs * Batch);
                increment_layer(InputLayer, 1);
                increment_layer(SelfLayer, 1);
                increment_layer(OutputLayer, 1);
            }
        }

        public override void Backward(ref NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;
            Layer inputLayer = (InputLayer);
            Layer selfLayer = (InputLayer);
            Layer outputLayer = OutputLayer;

            increment_layer(inputLayer, Steps - 1);
            increment_layer(selfLayer, Steps - 1);
            increment_layer(outputLayer, Steps - 1);

            Utils.IncArray(ref State, ref StateBackup, StateIndex, StateIndex += Hidden * Batch * Steps);
            for (i = Steps - 1; i >= 0; --i)
            {
                Blas.Copy_cpu(Hidden * Batch, inputLayer.Output, State);
                Blas.Axpy_cpu(Hidden * Batch, 1, selfLayer.Output, State);

                s.Input = State;
                s.Delta = selfLayer.Delta;
                OutputLayer.Backward(ref s);

                Utils.DecArray(ref State, ref StateBackup, StateIndex, StateIndex -= Hidden * Batch);

                s.Input = State;
                if (i == 0)
                {
                    s.Delta = new float[0];
                }

                SelfLayer.Backward(ref s);

                Blas.Copy_cpu(Hidden * Batch, selfLayer.Delta, inputLayer.Delta);
                var tempFloats = new float[selfLayer.Delta.Length - Hidden * Batch];
                Array.Copy(selfLayer.DeltaBackup, selfLayer.DeltaIndex - Hidden * Batch, tempFloats, 0, tempFloats.Length);
                if (i > 0 && Shortcut) Blas.Axpy_cpu(Hidden * Batch, 1, selfLayer.Delta, tempFloats);
                Array.Copy(tempFloats, 0, selfLayer.DeltaBackup, selfLayer.DeltaIndex - Hidden * Batch, tempFloats.Length);
                Array.Copy(state.Input, i * Inputs * Batch, s.Input, 0, state.Input.Length);
                if (state.Delta.Length != 0)
                {
                    Array.Copy(state.Delta, i * Inputs * Batch, s.Delta, 0, state.Delta.Length);
                }
                else s.Delta = new float[0];
                InputLayer.Backward(ref s);

                increment_layer(inputLayer, -1);
                increment_layer(selfLayer, -1);
                increment_layer(outputLayer, -1);
            }
        }

        public override void UpdateGpu(ref int batch,ref  float learningRate,ref  float momentum,ref  float decay)
        {
            InputLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            SelfLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            OutputLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;

            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref OutputLayer.DeltaGpu, 1);
            Blas.fill_ongpu(Hidden * Batch * Steps, 0, ref SelfLayer.DeltaGpu, 1);
            Blas.fill_ongpu(Hidden * Batch * Steps, 0, ref InputLayer.DeltaGpu, 1);
            if (state.Train) Blas.fill_ongpu(Hidden * Batch, 0, ref StateGpu, 1);

            for (i = 0; i < Steps; ++i)
            {
                s.Input = state.Input;
                InputLayer.ForwardGpu(ref s);

                s.Input = StateGpu;
                SelfLayer.ForwardGpu(ref s);

                float[] oldState = StateGpu;
                if (state.Train)
                {
                    Utils.IncArray(ref StateGpu, ref StateGpuBackup, StateGpuIndex, StateGpuIndex += Hidden * Batch);
                }
                if (Shortcut)
                {
                    Blas.copy_ongpu(Hidden * Batch, oldState, StateGpu);
                }
                else
                {
                    Blas.fill_ongpu(Hidden * Batch, 0, ref StateGpu, 1);
                }
                Blas.axpy_ongpu(Hidden * Batch, 1, InputLayer.OutputGpu, StateGpu);
                Blas.axpy_ongpu(Hidden * Batch, 1, SelfLayer.OutputGpu, StateGpu);

                s.Input = StateGpu;
                OutputLayer.ForwardGpu(ref s);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += Inputs * Batch);
                increment_layer(InputLayer, 1);
                increment_layer(SelfLayer, 1);
                increment_layer(OutputLayer, 1);
            }
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;

            increment_layer(InputLayer, Steps - 1);
            increment_layer(SelfLayer, Steps - 1);
            increment_layer(OutputLayer, Steps - 1);
            Utils.IncArray(ref StateGpu, ref StateGpuBackup, StateGpuIndex, StateGpuIndex += Hidden * Batch * Steps);
            for (i = Steps - 1; i >= 0; --i)
            {
                Blas.copy_ongpu(Hidden * Batch, InputLayer.OutputGpu, StateGpu);
                Blas.axpy_ongpu(Hidden * Batch, 1, SelfLayer.OutputGpu, StateGpu);

                s.Input = StateGpu;
                s.Delta = SelfLayer.DeltaGpu;
                OutputLayer.BackwardGpu(ref s);

                Utils.DecArray(ref StateGpu, ref StateGpuBackup, StateGpuIndex, StateGpuIndex -= Hidden * Batch);

                s.Input = StateGpu;
                Array.Copy(SelfLayer.DeltaGpuBackup, SelfLayer.DeltaGpuIndex - Hidden * Batch, s.Delta, 0, SelfLayer.DeltaGpu.Length + Hidden * Batch);
                if (i == 0) s.Delta = new float[0];
                SelfLayer.BackwardGpu(ref s);

                Blas.copy_ongpu(Hidden * Batch, SelfLayer.DeltaGpu, InputLayer.DeltaGpu);

                if (i > 0 && Shortcut)
                {
                    var tempFloat = new float[SelfLayer.DeltaGpu.Length + Hidden * Batch];
                    Array.Copy(SelfLayer.DeltaGpuBackup, SelfLayer.DeltaGpuIndex + Hidden * Batch, tempFloat, 0, tempFloat.Length);
                    Blas.axpy_ongpu(Hidden * Batch, 1, SelfLayer.DeltaGpu, tempFloat);
                    Array.Copy(tempFloat, tempFloat.Length, SelfLayer.DeltaGpuBackup, SelfLayer.DeltaGpu.Length + Hidden * Batch, tempFloat.Length);
                }
                Array.Copy(state.Input, state.Input.Length + i * Inputs * Batch, s.Input, 0, i * Inputs * Batch);
                if (state.Delta.Length != 0)
                {
                    Array.Copy(state.Delta, state.Delta.Length + i * Inputs * Batch, s.Delta, 0, i * Inputs * Batch);
                }
                else s.Delta = new float[0];
                InputLayer.BackwardGpu(ref s);

                increment_layer(InputLayer, -1);
                increment_layer(SelfLayer, -1);
                increment_layer(OutputLayer, -1);
            }
        }

    }
}