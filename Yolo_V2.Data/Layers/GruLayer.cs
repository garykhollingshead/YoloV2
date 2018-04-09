using System;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class GruLayer: Layer
    {
        public GruLayer(int batch, int inputs, int outputs, int steps, bool batchNormalize)
        {
            Console.Error.Write($"GRU Layer: {inputs} inputs, {outputs} outputs\n");
            batch = batch / steps;
            Batch = batch;
            LayerType = Layers.Gru;
            Steps = steps;
            Inputs = inputs;

            Console.Error.Write($"\t\t");
            (InputZLayer) = new ConnectedLayer(batch * steps, inputs, outputs, Activation.Linear, batchNormalize);
            InputZLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (StateZLayer) = new ConnectedLayer(batch * steps, outputs, outputs, Activation.Linear, batchNormalize);
            StateZLayer.Batch = batch;
            
            Console.Error.Write($"\t\t");
            (InputRLayer) = new ConnectedLayer(batch * steps, inputs, outputs, Activation.Linear, batchNormalize);
            InputRLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (StateRLayer) = new ConnectedLayer(batch * steps, outputs, outputs, Activation.Linear, batchNormalize);
            StateRLayer.Batch = batch;
            
            Console.Error.Write($"\t\t");
            (InputHLayer) = new ConnectedLayer(batch * steps, inputs, outputs, Activation.Linear, batchNormalize);
            InputHLayer.Batch = batch;

            Console.Error.Write($"\t\t");
            (StateHLayer) = new ConnectedLayer(batch * steps, outputs, outputs, Activation.Linear, batchNormalize);
            StateHLayer.Batch = batch;

            BatchNormalize = batchNormalize;


            Outputs = outputs;
            Output = new float[outputs * batch * steps];
            Delta = new float[outputs * batch * steps];
            State = new float[outputs * batch];
            PrevState = new float[outputs * batch];
            ForgotState = new float[outputs * batch];
            ForgotDelta = new float[outputs * batch];

            RCpu = new float[outputs * batch];
            ZCpu = new float[outputs * batch];
            HCpu = new float[outputs * batch];

            ForgotStateGpu = (float[])Output.Clone();
            ForgotDeltaGpu = (float[])Output.Clone();
            PrevStateGpu = (float[])Output.Clone();
            StateGpu = (float[])Output.Clone();
            OutputGpu = (float[])Output.Clone();
            DeltaGpu = (float[])Delta.Clone();
            RGpu = (float[])OutputGpu.Clone();
            ZGpu = (float[])OutputGpu.Clone();
            HGpu = (float[])OutputGpu.Clone();
        }

        public override void Update(ref int batch,ref  float learningRate, ref float momentum, ref float decay)
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

            Blas.Fill_cpu(Outputs * Batch * Steps, 0, InputZLayer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, InputRLayer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, InputHLayer.Delta, 1);

            Blas.Fill_cpu(Outputs * Batch * Steps, 0, StateZLayer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, StateRLayer.Delta, 1);
            Blas.Fill_cpu(Outputs * Batch * Steps, 0, StateHLayer.Delta, 1);
            if (state.Train)
            {
                Blas.Fill_cpu(Outputs * Batch * Steps, 0, Delta, 1);
                Blas.Copy_cpu(Outputs * Batch, State, PrevState);
            }

            for (i = 0; i < Steps; ++i)
            {
                s.Input = State;
                StateZLayer.Forward(ref s);
                StateRLayer.Forward(ref s);

                s.Input = state.Input;
                InputZLayer.Forward(ref s);
                InputRLayer.Forward(ref s);
                InputHLayer.Forward(ref s);


                Blas.Copy_cpu(Outputs * Batch, InputZLayer.Output, ZCpu);
                Blas.Axpy_cpu(Outputs * Batch, 1, StateZLayer.Output, ZCpu);

                Blas.Copy_cpu(Outputs * Batch, InputRLayer.Output, RCpu);
                Blas.Axpy_cpu(Outputs * Batch, 1, StateRLayer.Output, RCpu);

                ActivationsHelper.Activate_array(ZCpu, Outputs * Batch, Activation.Logistic);
                ActivationsHelper.Activate_array(RCpu, Outputs * Batch, Activation.Logistic);

                Blas.Copy_cpu(Outputs * Batch, State, ForgotState);
                Blas.Mul_cpu(Outputs * Batch, RCpu, 1, ForgotState, 1);

                s.Input = ForgotState;
                StateHLayer.Forward(ref s);

                Blas.Copy_cpu(Outputs * Batch, InputHLayer.Output, HCpu);
                Blas.Axpy_cpu(Outputs * Batch, 1, StateHLayer.Output, HCpu);

                // USET ActivationsHelper.Activate_array(HCpu, Outputs * Batch, TANH);
                ActivationsHelper.Activate_array(HCpu, Outputs * Batch, Activation.Logistic);


                Blas.Weighted_sum_cpu(State, HCpu, ZCpu, Outputs * Batch, Output);

                Blas.Copy_cpu(Outputs * Batch, Output, State);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += Inputs * Batch);
                Utils.IncArray(ref Output, ref OutputBackup, OutputIndex, OutputIndex += Outputs * Batch);
                increment_layer(InputZLayer, 1);
                increment_layer(InputRLayer, 1);
                increment_layer(InputHLayer, 1);

                increment_layer(StateZLayer, 1);
                increment_layer(StateRLayer, 1);
                increment_layer(StateHLayer, 1);
            }
        }

        public override void Backward(ref NetworkState state)
        {
        }

        public override void UpdateGpu(ref int batch,ref float learningRate,ref float momentum,ref float decay)
        {
            InputRLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            InputZLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            InputHLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            StateRLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            StateZLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
            StateHLayer.UpdateGpu(ref batch,ref learningRate,ref momentum,ref decay);
        }

        public override void ForwardGpu(ref NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;

            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref InputZLayer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref InputRLayer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref InputHLayer.DeltaGpu, 1);

            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref StateZLayer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref StateRLayer.DeltaGpu, 1);
            Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref StateHLayer.DeltaGpu, 1);
            if (state.Train)
            {
                Blas.fill_ongpu(Outputs * Batch * Steps, 0, ref DeltaGpu, 1);
                Blas.copy_ongpu(Outputs * Batch, StateGpu, PrevStateGpu);
            }

            for (i = 0; i < Steps; ++i)
            {
                s.Input = StateGpu;
                StateZLayer.ForwardGpu(ref s);
                StateRLayer.ForwardGpu(ref s);

                s.Input = state.Input;
                InputZLayer.ForwardGpu(ref s);
                InputRLayer.ForwardGpu(ref s);
                InputHLayer.ForwardGpu(ref s);


                Blas.copy_ongpu(Outputs * Batch, InputZLayer.OutputGpu, ZGpu);
                Blas.axpy_ongpu(Outputs * Batch, 1, StateZLayer.OutputGpu, ZGpu);

                Blas.copy_ongpu(Outputs * Batch, InputRLayer.OutputGpu, RGpu);
                Blas.axpy_ongpu(Outputs * Batch, 1, StateRLayer.OutputGpu, RGpu);

                ActivationsHelper.activate_array_ongpu(ref ZGpu, Outputs * Batch, Activation.Logistic);
                ActivationsHelper.activate_array_ongpu(ref RGpu, Outputs * Batch, Activation.Logistic);

                Blas.copy_ongpu(Outputs * Batch, StateGpu, ForgotStateGpu);
                Blas.mul_ongpu(Outputs * Batch, RGpu, 1, ref ForgotStateGpu, 1);

                s.Input = ForgotStateGpu;
                StateHLayer.ForwardGpu(ref s);

                Blas.copy_ongpu(Outputs * Batch, InputHLayer.OutputGpu, HGpu);
                Blas.axpy_ongpu(Outputs * Batch, 1, StateHLayer.OutputGpu, HGpu);

                // USET ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, TANH);
                ActivationsHelper.activate_array_ongpu(ref HGpu, Outputs * Batch, Activation.Logistic);


                Blas.weighted_sum_gpu(StateGpu, HGpu, ZGpu, Outputs * Batch, ref OutputGpu);

                Blas.copy_ongpu(Outputs * Batch, OutputGpu, StateGpu);

                Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += Inputs * Batch);
                Utils.IncArray(ref OutputGpu, ref OutputGpuBackup, OutputGpuIndex, OutputGpuIndex += Outputs * Batch);
                increment_layer(InputZLayer, 1);
                increment_layer(InputRLayer, 1);
                increment_layer(InputHLayer, 1);

                increment_layer(StateZLayer, 1);
                increment_layer(StateRLayer, 1);
                increment_layer(StateHLayer, 1);
            }
        }

        public override void BackwardGpu(ref NetworkState state)
        {
            NetworkState s = new NetworkState();
            s.Train = state.Train;
            int i;

            increment_layer(InputZLayer, Steps - 1);
            increment_layer(InputRLayer, Steps - 1);
            increment_layer(InputHLayer, Steps - 1);

            increment_layer(StateZLayer, Steps - 1);
            increment_layer(StateRLayer, Steps - 1);
            increment_layer(StateHLayer, Steps - 1);

            Utils.IncArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex += Inputs * Batch * (Steps - 1));
            if (state.Delta.Length != 0)
            {
                Utils.IncArray(ref state.Delta, ref state.DeltaBackup, state.DeltaIndex, state.DeltaIndex += Inputs * Batch * (Steps - 1));
            }
            Utils.IncArray(ref OutputGpu, ref OutputGpuBackup, OutputGpuIndex, OutputGpuIndex += Outputs * Batch * (Steps - 1));
            Utils.IncArray(ref DeltaGpu, ref DeltaGpuBackup, DeltaGpuIndex, DeltaGpuIndex += Outputs * Batch * (Steps - 1));
            for (i = Steps - 1; i >= 0; --i)
            {
                if (i != 0)
                {
                    Blas.copy_ongpu(Outputs * Batch, OutputGpuBackup, PrevStateGpu, OutputGpuIndex - Outputs * Batch);
                }
                float[] prevDeltaGpu;
                if (i == 0)
                {
                    prevDeltaGpu = new float[0];
                }
                else
                {
                    prevDeltaGpu = new float[DeltaGpu.Length + Outputs * Batch];
                    Array.Copy(DeltaGpuBackup, DeltaGpuIndex - Outputs * Batch, prevDeltaGpu, 0, Outputs * Batch);
                    Array.Copy(DeltaGpu, 0, prevDeltaGpu, Outputs * Batch, DeltaGpu.Length);
                }

                Blas.copy_ongpu(Outputs * Batch, InputZLayer.OutputGpu, ZGpu);
                Blas.axpy_ongpu(Outputs * Batch, 1, StateZLayer.OutputGpu, ZGpu);

                Blas.copy_ongpu(Outputs * Batch, InputRLayer.OutputGpu, RGpu);
                Blas.axpy_ongpu(Outputs * Batch, 1, StateRLayer.OutputGpu, RGpu);

                ActivationsHelper.activate_array_ongpu(ref ZGpu, Outputs * Batch, Activation.Logistic);
                ActivationsHelper.activate_array_ongpu(ref RGpu, Outputs * Batch, Activation.Logistic);

                Blas.copy_ongpu(Outputs * Batch, InputHLayer.OutputGpu, HGpu);
                Blas.axpy_ongpu(Outputs * Batch, 1, StateHLayer.OutputGpu, HGpu);

                // USET ActivationsHelper.activate_array_ongpu(HGpu, Outputs * Batch, TANH);
                ActivationsHelper.activate_array_ongpu(ref HGpu, Outputs * Batch, Activation.Logistic);


                Blas.weighted_delta_gpu(PrevStateGpu, HGpu, ZGpu, ref prevDeltaGpu, ref InputHLayer.DeltaGpu, ref InputZLayer.DeltaGpu, Outputs * Batch, DeltaGpu);

                // USET ActivationsHelper.gradient_array_ongpu(HGpu, Outputs * Batch, TANH, input_h_layer.DeltaGpu);
                ActivationsHelper.gradient_array_ongpu(HGpu, Outputs * Batch, Activation.Logistic, ref InputHLayer.DeltaGpu);


                Blas.copy_ongpu(Outputs * Batch, InputHLayer.DeltaGpu, StateHLayer.DeltaGpu);

                Blas.copy_ongpu(Outputs * Batch, PrevStateGpu, ForgotStateGpu);
                Blas.mul_ongpu(Outputs * Batch, RGpu, 1, ref ForgotStateGpu, 1);
                Blas.fill_ongpu(Outputs * Batch, 0, ref ForgotDeltaGpu, 1);

                s.Input = ForgotStateGpu;
                s.Delta = ForgotDeltaGpu;

                StateHLayer.BackwardGpu(ref s);
                if (prevDeltaGpu.Length != 0)
                {
                    Blas.mult_add_into_gpu(Outputs * Batch, ForgotDeltaGpu, RGpu, ref prevDeltaGpu);
                }
                Blas.mult_add_into_gpu(Outputs * Batch, ForgotDeltaGpu, PrevStateGpu, ref InputRLayer.DeltaGpu);

                ActivationsHelper.gradient_array_ongpu(RGpu, Outputs * Batch, Activation.Logistic, ref InputRLayer.DeltaGpu);
                Blas.copy_ongpu(Outputs * Batch, InputRLayer.DeltaGpu, StateRLayer.DeltaGpu);

                ActivationsHelper.gradient_array_ongpu(ZGpu, Outputs * Batch, Activation.Logistic, ref InputZLayer.DeltaGpu);
                Blas.copy_ongpu(Outputs * Batch, InputZLayer.DeltaGpu, StateZLayer.DeltaGpu);

                s.Input = PrevStateGpu;
                s.Delta = prevDeltaGpu;

                StateRLayer.BackwardGpu(ref s);
                StateZLayer.BackwardGpu(ref s);

                s.Input = state.Input;
                s.Delta = state.Delta;

                InputHLayer.BackwardGpu(ref s);
                InputRLayer.BackwardGpu(ref s);
                InputZLayer.BackwardGpu(ref s);

                Utils.DecArray(ref state.Input, ref state.InputBackup, state.InputIndex, state.InputIndex -= Inputs * Batch);
                if (state.Delta.Length != 0)
                {
                    Utils.DecArray(ref state.Delta, ref state.DeltaBackup, state.DeltaIndex, state.DeltaIndex -= Inputs * Batch);
                }
                Utils.DecArray(ref OutputGpu, ref OutputGpuBackup, OutputGpuIndex, OutputGpuIndex -= Outputs * Batch);
                Utils.DecArray(ref DeltaGpu, ref DeltaGpuBackup, DeltaGpuIndex, DeltaGpuIndex -= Outputs * Batch);
                increment_layer(InputZLayer, -1);
                increment_layer(InputRLayer, -1);
                increment_layer(InputHLayer, -1);

                increment_layer(StateZLayer, -1);
                increment_layer(StateRLayer, -1);
                increment_layer(StateHLayer, -1);
            }
        }

    }
}