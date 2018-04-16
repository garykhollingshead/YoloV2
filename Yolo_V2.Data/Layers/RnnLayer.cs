using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class RnnLayer:Layer
    {

        public RnnLayer(int batch, int inputs, int hidden, int outputs, int steps, Activation activation, bool batchNormalize, int log)
        {
            Console.Error.Write($"RNN Layer: {inputs} inputs, {outputs} outputs\n");
            batch = batch / steps;
            Batch = batch;
            LayerType = Layers.Rnn;
            Steps = steps;
            Hidden = hidden;
            Inputs = inputs;

            State = new float[batch * hidden * (steps + 1)];

            Console.Error.Write($"\t\t");
            (InputLayer) = new ConnectedLayer(batch * steps, inputs, hidden, activation, batchNormalize);
            InputLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (SelfLayer) = new ConnectedLayer(batch * steps, hidden, hidden, (log == 2) ? Activation.Loggy : (log == 1 ? Activation.Logistic : activation), batchNormalize);
            SelfLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (OutputLayer) = new ConnectedLayer(batch * steps, hidden, outputs, activation, batchNormalize);
            OutputLayer.Batch = batch;

            Outputs = outputs;
            Output = OutputLayer.Output;
            Delta = OutputLayer.Delta;

            StateGpu = (float[])State.Clone();
            OutputGpu = OutputLayer.OutputGpu;
            DeltaGpu = OutputLayer.DeltaGpu;
        }

        public override void Update(ref int batch,ref float learningRate,ref float momentum,ref float decay)
        {
            InputLayer.Update(ref batch,ref learningRate,ref momentum,ref decay);
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

            increment_layer(InputLayer, Steps - 1);
            increment_layer(SelfLayer, Steps - 1);
            increment_layer(OutputLayer, Steps - 1);

            Utils.IncArray(ref State, ref StateBackup, StateIndex, StateIndex += Hidden * Batch * Steps);
            for (i = Steps - 1; i >= 0; --i)
            {
                Blas.Copy_cpu(Hidden * Batch, InputLayer.Output, State);
                Blas.Axpy_cpu(Hidden * Batch, 1, SelfLayer.Output, State);

                s.Input = State;
                s.Delta = SelfLayer.Delta;
                OutputLayer.Backward(ref s);

                Utils.DecArray(ref State, ref StateBackup, StateIndex, StateIndex -= Hidden * Batch);

                s.Input = State;
                s.Delta = new float[SelfLayer.Delta.Length + Hidden * Batch];
                Array.Copy(SelfLayer.DeltaBackup, SelfLayer.DeltaIndex - Hidden * Batch, s.Delta, 0, s.Delta.Length);
                if (i == 0) s.Delta = new float[0];
                SelfLayer.Backward(ref s);

                Blas.Copy_cpu(Hidden * Batch, SelfLayer.Delta, InputLayer.Delta);
                if (i > 0 && Shortcut) Blas.Axpy_cpu(Hidden * Batch, 1, SelfLayer.Delta, SelfLayer.DeltaBackup, 0, SelfLayer.DeltaIndex - Hidden * Batch);

                s.Input = new float[state.Input.Length - i * Inputs * Batch];
                Array.Copy(state.Input, i * Inputs * Batch, s.Input, 0, s.Input.Length);
                if (state.Delta.Length != 0)
                {
                    s.Delta = new float[state.Delta.Length - i * Inputs * Batch];
                    Array.Copy(state.Delta, i * Inputs * Batch, s.Delta, 0, s.Delta.Length);
                }
                else s.Delta = new float[0];
                InputLayer.Backward(ref s);

                increment_layer(InputLayer, -1);
                increment_layer(SelfLayer, -1);
                increment_layer(OutputLayer, -1);
            }
        }

        public override void UpdateGpu(ref int batch,ref float learningRate,ref float momentum, ref float decay)
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

            OutputLayer.DeltaGpu = new float[Outputs * Batch * Steps];
            SelfLayer.DeltaGpu = new float[Hidden* Batch* Steps];
            InputLayer.DeltaGpu = new float[Hidden * Batch * Steps];
            if (state.Train)
                StateGpu = new float[Hidden * Batch];

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
                    Blas.copy_ongpu(Hidden * Batch, oldState, ref StateGpu);
                }
                else
                {
                    StateGpu = new float[Hidden * Batch];
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

                s.Input = StateGpu;
                s.Delta = SelfLayer.DeltaGpu;
                OutputLayer.BackwardGpu(ref s);

                Utils.DecArray(ref StateGpu, ref StateGpuBackup, StateGpuIndex, StateGpuIndex -= Hidden * Batch);

                Blas.copy_ongpu(Hidden * Batch, SelfLayer.DeltaGpu, ref InputLayer.DeltaGpu);

                s.Input = StateGpu;
                s.Delta = new float[SelfLayer.DeltaGpu.Length + Hidden * Batch];
                Array.Copy(SelfLayer.DeltaGpuBackup, SelfLayer.DeltaGpuIndex - Hidden * Batch, s.Delta, 0, s.Delta.Length);
                if (i == 0) s.Delta = new float[0];
                SelfLayer.BackwardGpu(ref s);

                if (i > 0 && Shortcut) Blas.axpy_ongpu(Hidden * Batch, 1, SelfLayer.DeltaGpu, SelfLayer.DeltaGpuBackup, 0, SelfLayer.DeltaGpuIndex - Hidden * Batch);
                s.Input = new float[state.Input.Length - i * Inputs * Batch];
                Array.Copy(state.Input, i * Inputs * Batch, s.Input, 0, s.Input.Length);
                if (state.Delta.Length != 0)
                {
                    s.Delta = new float[state.Delta.Length - i * Inputs * Batch];
                    Array.Copy(state.Delta, i * Inputs * Batch, s.Delta, 0, s.Delta.Length);
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