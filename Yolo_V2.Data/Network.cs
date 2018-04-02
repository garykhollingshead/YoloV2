using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class Network
    {
        public float[] Workspace;
        public int N;
        public int Batch;
        public int Seen;
        public float Epoch;
        public int Subdivisions;
        public float Momentum;
        public float Decay;
        public Layer[] Layers;
        public int Outputs;
        public float[] Output;
        public LearningRatePolicy Policy;

        public float LearningRate;
        public float Gamma;
        public float Scale;
        public float Power;
        public int TimeSteps;
        public int Step;
        public int MaxBatches;
        public float[] Scales;
        public int[] Steps;
        public int NumSteps;
        public int BurnIn;

        public bool Adam;
        public float B1;
        public float B2;
        public float Eps;

        public int Inputs;
        public int H, W, C;
        public int MaxCrop;
        public int MinCrop;
        public float Angle;
        public float Aspect;
        public float Exposure;
        public float Saturation;
        public float Hue;

        public Tree Hierarchy;

        public float[] InputGpu;
        public float[] TruthGpu;

        public Network()
        {
            InputGpu = new float[0];
            TruthGpu = new float[0];
            Scales = new float[0];
            Steps = new int[0];
            Workspace = new float[0];
            Seen = 0;
            Layers = new Layer[0];
            Output = new float[0];
        }

        public Network(int n)
        {
            N = n;
            Layers = new Layer[n];
            Seen = 0;
            InputGpu = new float[0];
            TruthGpu = new float[0];
        }

        public static int get_current_batch(Network net)
        {
            int batch_num = (net.Seen) / (net.Batch * net.Subdivisions);
            return batch_num;
        }

        public static void reset_momentum(Network net)
        {
            if (net.Momentum == 0) return;
            net.LearningRate = 0;
            net.Momentum = 0;
            net.Decay = 0;
        }

        public static float get_current_rate(Network net)
        {
            int batch_num = get_current_batch(net);
            int i;
            float rate;
            switch (net.Policy)
            {
                case LearningRatePolicy.Constant:
                    return net.LearningRate;
                case LearningRatePolicy.Step:
                    return net.LearningRate * (float)Math.Pow(net.Scale, batch_num / net.Step);
                case LearningRatePolicy.Steps:
                    rate = net.LearningRate;
                    for (i = 0; i < net.NumSteps; ++i)
                    {
                        if (net.Steps[i] > batch_num) return rate;
                        rate *= net.Scales[i];
                        //if(net.steps[i] > batch_num - 1 && net.Scales[i] > 1) reset_momentum(net);
                    }
                    return rate;
                case LearningRatePolicy.Exp:
                    return net.LearningRate * (float)Math.Pow(net.Gamma, batch_num);
                case LearningRatePolicy.Poly:
                    if (batch_num < net.BurnIn) return net.LearningRate * (float)Math.Pow((float)batch_num / net.BurnIn, net.Power);
                    return net.LearningRate * (float)Math.Pow(1 - (float)batch_num / net.MaxBatches, net.Power);
                case LearningRatePolicy.Random:
                    return net.LearningRate * (float)Math.Pow(Utils.rand_uniform(0, 1), net.Power);
                case LearningRatePolicy.Sig:
                    return net.LearningRate * (1.0f / (1.0f + (float)Math.Exp(net.Gamma * (batch_num - net.Step))));
                default:
                    Console.Error.WriteLine("Policy is weird!");
                    return net.LearningRate;
            }
        }

        public static string get_layer_string(LayerType a)
        {
            return a.ToString();
        }

        public static void forward_network(Network net, NetworkState state)
        {
            state.Workspace = net.Workspace;
            int i;
            for (i = 0; i < net.N; ++i)
            {
                state.Index = i;
                Layer l = net.Layers[i];
                if (l.Delta.Any())
                {
                    Blas.Scal_cpu(l.Outputs * l.Batch, 0, l.Delta, 1);
                }
                l.Forward(l, state);
                state.Input = l.Output;
            }
        }

        public static void update_network(Network net)
        {
            int i;
            int update_batch = net.Batch * net.Subdivisions;
            float rate = get_current_rate(net);
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                l.Update?.Invoke(l, update_batch, rate, net.Momentum, net.Decay);
            }
        }

        public static float[] get_network_output(Network net)
        {
            if (CudaUtils.UseGpu) return get_network_output_gpu(net);
            int i;
            for (i = net.N - 1; i > 0; --i) if (net.Layers[i].LayerType != LayerType.Cost) break;
            return net.Layers[i].Output;
        }

        public static float get_network_cost(Network net)
        {
            int i;
            float sum = 0;
            int count = 0;
            for (i = 0; i < net.N; ++i)
            {
                if (net.Layers.Length > i)
                {
                    sum += net.Layers[i].Cost ?? 0;
                    ++count;
                }
            }
            return sum / count;
        }

        public static int get_predicted_class_network(Network net)
        {
            float[] outputF = get_network_output(net);
            int k = get_network_output_size(net);
            return Utils.max_index(outputF, k);
        }

        public static void backward_network(Network net, NetworkState state)
        {
            int i;
            float[] original_input = state.Input;
            float[] original_delta = state.Delta;
            state.Workspace = net.Workspace;
            for (i = net.N - 1; i >= 0; --i)
            {
                state.Index = i;
                if (i == 0)
                {
                    state.Input = original_input;
                    state.Delta = original_delta;
                }
                else
                {
                    Layer prev = net.Layers[i - 1];
                    state.Input = prev.Output;
                    state.Delta = prev.Delta;
                }
                Layer l = net.Layers[i];
                l.Backward(l, state);
            }
        }

        public static float train_network_datum(Network net, float[] x, float[] y)
        {
            if (CudaUtils.UseGpu) return train_network_datum_gpu(net, x, y);
            NetworkState state = new NetworkState();
            net.Seen += net.Batch;
            state.Index = 0;
            state.Net = net;
            state.Input = x;
            state.Delta = new float[0];
            state.Truth = y;
            state.Train = true;
            forward_network(net, state);
            backward_network(net, state);
            float error = get_network_cost(net);
            if ((net.Seen / net.Batch) % net.Subdivisions == 0) update_network(net);
            return error;
        }

        public static float train_network_sgd(Network net, Data d, int n)
        {
            int batch = net.Batch;
            float[] X = new float[batch * d.X.Cols];
            float[] y = new float[batch * d.Y.Cols];

            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                d.get_random_batch(batch, X, y);
                float err = train_network_datum(net, X, y);
                sum += err;
            }
            return sum / (n * batch);
        }

        public static float train_network(Network net, Data d)
        {
            int batch = net.Batch;
            int n = d.X.Rows / batch;
            float[] X = new float[batch * d.X.Cols];
            float[] y = new float[batch * d.Y.Cols];

            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                d.get_next_batch(batch, i * batch, X, y);
                float err = train_network_datum(net, X, y);
                sum += err;
            }
            return (float)sum / (n * batch);
        }

        public static float train_network_batch(Network net, Data d, int n)
        {
            int i, j;
            NetworkState state = new NetworkState();
            state.Index = 0;
            state.Net = net;
            state.Train = true;
            state.Delta = new float[0];
            float sum = 0;
            int batch = 2;
            for (i = 0; i < n; ++i)
            {
                for (j = 0; j < batch; ++j)
                {
                    int index = Utils.Rand.Next() % d.X.Rows;
                    state.Input = d.X.Vals[index];
                    state.Truth = d.Y.Vals[index];
                    forward_network(net, state);
                    backward_network(net, state);
                    sum += get_network_cost(net);
                }
                update_network(net);
            }
            return (float)sum / (n * batch);
        }

        public static void set_batch_network(Network net, int b)
        {
            net.Batch = b;
            int i;
            for (i = 0; i < net.N; ++i)
            {
                net.Layers[i].Batch = b;
                if (net.Layers[i].LayerType == LayerType.Convolutional)
                {
                    net.Layers[i].cudnn_convolutional_setup();
                }
            }
        }

        public static int resize_network(Network net, int w, int h)
        {
            if (CudaUtils.UseGpu)
            {
                net.Workspace = null;
            }
            int i;
            net.W = w;
            net.H = h;
            int inputs = 0;
            ulong workspace_size = 0;

            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    l.resize_convolutional_layer( w, h);
                }
                else if (l.LayerType == LayerType.Crop)
                {
                    l.resize_crop_layer(w, h);
                }
                else if (l.LayerType == LayerType.Maxpool)
                {
                    l.resize_maxpool_layer( w, h);
                }
                else if (l.LayerType == LayerType.Region)
                {
                    l.resize_region_layer( w, h);
                }
                else if (l.LayerType == LayerType.Route)
                {
                    Layer.resize_route_layer(l, net);
                }
                else if (l.LayerType == LayerType.Reorg)
                {
                    Layer.resize_reorg_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Avgpool)
                {
                    l.resize_avgpool_layer( w, h);
                }
                else if (l.LayerType == LayerType.Normalization)
                {
                    l.resize_normalization_layer( w, h);
                }
                else if (l.LayerType == LayerType.Cost)
                {
                    l.resize_cost_layer( inputs);
                }
                else
                {
                    Utils.Error("Cannot resize this type of Layer");
                }
                if (l.WorkspaceSize > workspace_size) workspace_size = l.WorkspaceSize;
                inputs = l.Outputs;
                net.Layers[i] = l;
                w = l.OutW;
                h = l.OutH;
                if (l.LayerType == LayerType.Avgpool) break;
            }
            if (CudaUtils.UseGpu)
            {
                if (net.InputGpu.Any())
                {
                    net.InputGpu = null;
                    net.TruthGpu = null;
                }
                net.Workspace = new float[workspace_size];
            }
            else
            {
                net.Workspace = new float[1];
            }
            return 0;
        }

        public static int get_network_output_size(Network net)
        {
            int i;
            for (i = net.N - 1; i > 0; --i) if (net.Layers[i].LayerType != LayerType.Cost) break;
            return net.Layers[i].Outputs;
        }

        public static int get_network_input_size(Network net)
        {
            return net.Layers[0].Inputs;
        }

        public static Layer get_network_detection_layer(Network net)
        {
            int i;
            for (i = 0; i < net.N; ++i)
            {
                if (net.Layers[i].LayerType == LayerType.Detection)
                {
                    return net.Layers[i];
                }
            }
            Console.Error.WriteLine("Detection Layer not found!!");
            return new Layer();
        }

        public static Image get_network_image_layer(Network net, int i)
        {
            Layer l = net.Layers[i];
            if (l.OutW != 0 && l.OutH != 0 && l.OutC != 0)
            {
                return new Image(l.OutW, l.OutH, l.OutC, l.Output);
            }
            return new Image();
        }

        public static Image get_network_image(Network net)
        {
            int i;
            for (i = net.N - 1; i >= 0; --i)
            {
                Image m = get_network_image_layer(net, i);
                if (m.H != 0) return m;
            }
            return new Image();
        }

        public static void visualize_network(Network net)
        {
            int i;

            for (i = 0; i < net.N; ++i)
            {
                string buff = $"Layer {i}";
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    l.visualize_convolutional_layer( buff);
                }
            }
        }

        public static void top_predictions(Network net, int k, int[] index)
        {
            int size = get_network_output_size(net);
            float[] outputF = get_network_output(net);
            Utils.top_k(outputF, size, k, index);
        }

        public static float[] network_predict(Network net, float[] input)
        {
            if (CudaUtils.UseGpu) return network_predict_gpu(net, input);

            NetworkState state = new NetworkState();
            state.Net = net;
            state.Index = 0;
            state.Input = input;
            state.Truth = null;
            state.Train = false;
            state.Delta = null;
            forward_network(net, state);
            float[] outputF = get_network_output(net);
            return outputF;
        }

        public static Matrix network_predict_data_multi(Network net, Data test, int n)
        {
            int i, j, b, m;
            int k = get_network_output_size(net);
            Matrix pred = new Matrix(test.X.Rows, k);
            float[] X = new float[net.Batch * test.X.Rows];
            for (i = 0; i < test.X.Rows; i += net.Batch)
            {
                for (b = 0; b < net.Batch; ++b)
                {
                    if (i + b == test.X.Rows) break;
                    Array.Copy(test.X.Vals[i + b], 0, X, b * test.X.Cols, test.X.Cols);
                }
                for (m = 0; m < n; ++m)
                {
                    float[] outputF = network_predict(net, X);
                    for (b = 0; b < net.Batch; ++b)
                    {
                        if (i + b == test.X.Rows) break;
                        for (j = 0; j < k; ++j)
                        {
                            pred.Vals[i + b][j] += outputF[j + b * k] / n;
                        }
                    }
                }
            }
            return pred;
        }

        public static Matrix network_predict_data(Network net, Data test)
        {
            int i, j, b;
            int k = get_network_output_size(net);
            Matrix pred = new Matrix(test.X.Rows, k);
            float[] X = new float[net.Batch * test.X.Cols];
            for (i = 0; i < test.X.Rows; i += net.Batch)
            {
                for (b = 0; b < net.Batch; ++b)
                {
                    if (i + b == test.X.Rows) break;
                    Array.Copy(test.X.Vals[i + b], 0, X, b * test.X.Cols, test.X.Cols);
                }
                float[] outputF = network_predict(net, X);
                for (b = 0; b < net.Batch; ++b)
                {
                    if (i + b == test.X.Rows) break;
                    for (j = 0; j < k; ++j)
                    {
                        pred.Vals[i + b][j] = outputF[j + b * k];
                    }
                }
            }
            return pred;
        }

        public static void print_network(Network net)
        {
            int i, j;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                float[] output = l.Output;
                int n = l.Outputs;
                float mean = Utils.mean_array(output, n);
                float vari = Utils.variance_array(output, n);
                Console.Error.WriteLine($"Layer {i} - Mean: {mean}, Variance: {vari}");
                if (n > 100) n = 100;
                for (j = 0; j < n; ++j) Console.Error.Write($"{output[j]}, ");
                if (n == 100) Console.Error.Write(".....\n");
                Console.Error.Write("\n");
            }
        }

        public static void compare_networks(Network n1, Network n2, Data test)
        {
            Matrix g1 = network_predict_data(n1, test);
            Matrix g2 = network_predict_data(n2, test);
            int i;
            int a, b, c, d;
            a = b = c = d = 0;
            for (i = 0; i < g1.Rows; ++i)
            {
                int truth = Utils.max_index(test.Y.Vals[i], test.Y.Cols);
                int p1 = Utils.max_index(g1.Vals[i], g1.Cols);
                int p2 = Utils.max_index(g2.Vals[i], g2.Cols);
                if (p1 == truth)
                {
                    if (p2 == truth) ++d;
                    else ++c;
                }
                else
                {
                    if (p2 == truth) ++b;
                    else ++a;
                }
            }
            Console.Write($"{a:5} {b:5}\n{c:5} {d:5}\n");
            float num = (float)Math.Pow((Math.Abs(b - c) - 1), 2);
            float den = b + c;
            Console.Write($"{num / den}\n");
        }

        public static float network_accuracy(Network net, Data d)
        {
            Matrix guess = network_predict_data(net, d);
            float acc = Matrix.matrix_topk_accuracy(d.Y, guess, 1);
            return acc;
        }

        public static float[] network_accuracies(Network net, Data d, int n)
        {
            float[] acc = new float[2];
            Matrix guess = network_predict_data(net, d);
            acc[0] = Matrix.matrix_topk_accuracy(d.Y, guess, 1);
            acc[1] = Matrix.matrix_topk_accuracy(d.Y, guess, n);
            return acc;
        }

        public static float network_accuracy_multi(Network net, Data d, int n)
        {
            Matrix guess = network_predict_data_multi(net, d, n);
            float acc = Matrix.matrix_topk_accuracy(d.Y, guess, 1);
            return acc;
        }

        public static void forward_network_gpu(Network net, NetworkState state)
        {
            state.Workspace = net.Workspace;
            int i;
            for (i = 0; i < net.N; ++i)
            {
                state.Index = i;
                Layer l = net.Layers[i];
                if (l.DeltaGpu.Any())
                {
                    Blas.fill_ongpu(l.Outputs * l.Batch, 0, l.DeltaGpu, 1);
                }
                l.ForwardGpu(l, state);
                state.Input = l.OutputGpu;
            }
        }
        
        public static void backward_network_gpu(Network net, NetworkState state)
        {
            state.Workspace = net.Workspace;
            int i;
            float[] original_input = state.Input;
            float[] original_delta = state.Delta;
            for (i = net.N - 1; i >= 0; --i)
            {
                state.Index = i;
                Layer l = net.Layers[i];
                if (i == 0)
                {
                    state.Input = original_input;
                    state.Delta = original_delta;
                }
                else
                {
                    Layer prev = net.Layers[i - 1];
                    state.Input = prev.OutputGpu;
                    state.Delta = prev.DeltaGpu;
                }
                l.BackwardGpu(l, state);
            }
        }
        
        public static void update_network_gpu(Network net)
        {
            int i;
            int update_batch = net.Batch * net.Subdivisions;
            float rate = get_current_rate(net);
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                l.T = get_current_batch(net);
                l.UpdateGpu?.Invoke(l, update_batch, rate, net.Momentum, net.Decay);
            }
        }
        
        public static void forward_backward_network_gpu(Network net, float[] x, float[] y)
        {
            NetworkState state = new NetworkState();
            state.Index = 0;
            state.Net = net;
            int x_size = get_network_input_size(net) * net.Batch;
            int y_size = get_network_output_size(net) * net.Batch;
            if (net.Layers[net.N - 1].Truths != 0) y_size = net.Layers[net.N - 1].Truths * net.Batch;
            if (!net.InputGpu.Any())
            {
                net.InputGpu = (float[])x.Clone(); 
                net.TruthGpu = (float[])y.Clone();
            }
            else
            {
                Array.Copy(x, net.InputGpu, x_size);
                Array.Copy(y, net.TruthGpu, y_size);
            }
            state.Input = net.InputGpu;
            state.Delta = new float[0];
            state.Truth = net.TruthGpu;
            state.Train = true;
            forward_network_gpu(net, state);
            backward_network_gpu(net, state);
        }
        
        public static float train_network_datum_gpu(Network net, float[] x, float[] y)
        {
            net.Seen += net.Batch;
            forward_backward_network_gpu(net, x, y);
            float error = get_network_cost(net);
            if ((net.Seen / net.Batch) % net.Subdivisions == 0) update_network_gpu(net);

            return error;
        }
        
        public static void train_thread(TrainArgs ptr)
        {
            ptr.Err[0] = train_network(ptr.Net, ptr.D);
        }
        
        public static Thread train_network_in_thread(Network net, Data d, float[] err)
        {
            var ptr = new TrainArgs
            {
                Net = net,
                D = d,
                Err = err
            };
            var thread = new Thread(() => {train_thread(ptr);});
            thread.Start();
            return thread;
        }
        
        public static void pull_updates(Layer l)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Array.Copy(l.BiasUpdatesGpu, l.BiasUpdates, l.N);
                Array.Copy(l.WeightUpdatesGpu, l.WeightUpdates, l.N * l.Size * l.Size * l.C);
                if (l.ScaleUpdates.Any()) Array.Copy(l.ScaleUpdatesGpu, l.ScaleUpdates, l.N);
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Array.Copy(l.BiasUpdatesGpu, l.BiasUpdates, l.Outputs);
                Array.Copy(l.WeightUpdatesGpu, l.WeightUpdates, l.Outputs * l.Inputs);
            }
        }
        
        public static void push_updates(Layer l)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Array.Copy(l.BiasUpdatesGpu, l.BiasUpdates, l.N);
                Array.Copy(l.WeightUpdatesGpu, l.WeightUpdates, l.N * l.Size * l.Size * l.C);
                if (l.ScaleUpdates.Any()) Array.Copy(l.ScaleUpdatesGpu, l.ScaleUpdates, l.N);
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Array.Copy(l.BiasUpdatesGpu, l.BiasUpdates, l.Outputs);
                Array.Copy(l.WeightUpdatesGpu, l.WeightUpdates, l.Outputs * l.Inputs);
            }
        }
        
        public static void update_layer(Layer l, Network net)
        {
            int update_batch = net.Batch * net.Subdivisions;
            float rate = get_current_rate(net);
            l.T = get_current_batch(net);
            l.UpdateGpu?.Invoke(l, update_batch, rate, net.Momentum, net.Decay);
        }
        
        public static void merge_weights(Layer l, Layer baseLayer)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Blas.Axpy_cpu(l.N, 1, l.Biases, baseLayer.Biases);
                Blas.Axpy_cpu(l.N * l.Size * l.Size * l.C, 1, l.Weights, baseLayer.Weights);
                if (l.Scales.Any())
                {
                    Blas.Axpy_cpu(l.N, 1, l.Scales, baseLayer.Scales);
                }
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.Biases, baseLayer.Biases);
                Blas.Axpy_cpu(l.Outputs * l.Inputs, 1, l.Weights, baseLayer.Weights);
            }
        }
        
        public static void scale_weights(Layer l, float s)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Blas.Scal_cpu(l.N, s, l.Biases, 1);
                Blas.Scal_cpu(l.N * l.Size * l.Size * l.C, s, l.Weights, 1);
                if (l.Scales.Any())
                {
                    Blas.Scal_cpu(l.N, s, l.Scales, 1);
                }
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Blas.Scal_cpu(l.Outputs, s, l.Biases, 1);
                Blas.Scal_cpu(l.Outputs * l.Inputs, s, l.Weights, 1);
            }
        }
        
        public static void pull_weights(Layer l)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Array.Copy(l.BiasesGpu, l.Biases, l.N);
                Array.Copy(l.WeightsGpu, l.Weights, l.N * l.Size * l.Size * l.C);
                if (l.Scales.Any()) Array.Copy(l.ScalesGpu, l.Scales, l.N);
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Array.Copy(l.BiasesGpu, l.Biases, l.Outputs);
                Array.Copy(l.WeightsGpu, l.Weights, l.Outputs * l.Inputs);
            }
        }
        
        public static void push_weights(Layer l)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Array.Copy(l.BiasesGpu, l.Biases, l.N);
                Array.Copy(l.WeightsGpu, l.Weights, l.N * l.Size * l.Size * l.C);
                if (l.Scales.Any()) Array.Copy(l.ScalesGpu, l.Scales, l.N);
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Array.Copy(l.BiasesGpu, l.Biases, l.Outputs);
                Array.Copy(l.WeightsGpu, l.Weights, l.Outputs * l.Inputs);
            }
        }
        
        public static void distribute_weights(Layer l, Layer baseLayer)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Array.Copy(baseLayer.Biases, l.BiasesGpu, l.N);
                Array.Copy(baseLayer.Weights, l.WeightsGpu, l.N * l.Size * l.Size * l.C);
                if (baseLayer.Scales.Any()) Array.Copy(l.ScalesGpu, baseLayer.Scales, l.N);
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Array.Copy(l.BiasesGpu, baseLayer.Biases, l.Outputs);
                Array.Copy(l.WeightsGpu, baseLayer.Weights, l.Outputs * l.Inputs);
            }
        }
        
        public static void merge_updates(Layer l, Layer baseLayer)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Blas.Axpy_cpu(l.N, 1, l.BiasUpdates, baseLayer.BiasUpdates);
                Blas.Axpy_cpu(l.N * l.Size * l.Size * l.C, 1, l.WeightUpdates, baseLayer.WeightUpdates);
                if (l.ScaleUpdates.Any())
                {
                    Blas.Axpy_cpu(l.N, 1, l.ScaleUpdates, baseLayer.ScaleUpdates);
                }
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Blas.Axpy_cpu(l.Outputs, 1, l.BiasUpdates, baseLayer.BiasUpdates);
                Blas.Axpy_cpu(l.Outputs * l.Inputs, 1, l.WeightUpdates, baseLayer.WeightUpdates);
            }
        }
        
        public static void distribute_updates(Layer l, Layer baseLayer)
        {
            if (l.LayerType == LayerType.Convolutional)
            {
                Array.Copy(l.BiasUpdatesGpu, baseLayer.BiasUpdates, l.N);
                Array.Copy(l.WeightUpdatesGpu, baseLayer.WeightUpdates, l.N * l.Size * l.Size * l.C);
                if (baseLayer.ScaleUpdates.Any()) Array.Copy(l.ScaleUpdatesGpu, baseLayer.ScaleUpdates, l.N);
            }
            else if (l.LayerType == LayerType.Connected)
            {
                Array.Copy(l.BiasUpdatesGpu, baseLayer.BiasUpdates, l.Outputs);
                Array.Copy(l.WeightUpdatesGpu, baseLayer.WeightUpdates, l.Outputs * l.Inputs);
            }
        }
        
        public static void sync_layer(Network[] nets, int n, int j)
        {
            int i;
            Network net = nets[0];
            Layer baseLayer = net.Layers[j];
            pull_weights(baseLayer);
            for (i = 1; i < n; ++i)
            {
                Layer l = nets[i].Layers[j];
                pull_weights(l);
                merge_weights(l, baseLayer);
            }
            scale_weights(baseLayer, 1.0f/ n);
            for (i = 0; i < n; ++i)
            {
                Layer l = nets[i].Layers[j];
                distribute_weights(l, baseLayer);
            }
        }
        
        public static void sync_layer_thread(SyncArgs ptr)
        {
            sync_layer(ptr.Nets, ptr.N, ptr.J);
            return;
        }
        
        public static Thread sync_layer_in_thread(Network[] nets, int n, int j)
        {
            SyncArgs ptr = new SyncArgs();
            ptr.Nets = nets;
            ptr.N = n;
            ptr.J = j;

            Thread thread = new Thread(() => sync_layer_thread(ptr));
            thread.Start();
            return thread;
        }
        
        public static void sync_nets(Network[] nets, int n, int interval)
        {
            int j;
            int layers = nets[0].N;
            Thread[] threads = new Thread[layers];

            nets[0].Seen += interval * (n - 1) * nets[0].Batch * nets[0].Subdivisions;
            for (j = 0; j < n; ++j)
            {
                nets[j].Seen = nets[0].Seen;
            }
            for (j = 0; j < layers; ++j)
            {
                threads[j] = sync_layer_in_thread(nets, n, j);
            }
            for (j = 0; j < layers; ++j)
            {
                threads[j].Join();
            }
        }
        
        public static float train_networks(Network[] nets, int n, Data d, int interval)
        {
            int i;
            int batch = nets[0].Batch;
            int subdivisions = nets[0].Subdivisions;
            Thread[] threads = new Thread[n];
            float[][] errors = new float[n][];

            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                Data p = Data.get_data_part(d, i, n);
                errors[i] = new float[1];
                threads[i] = train_network_in_thread(nets[i], p, errors[i]);
            }
            for (i = 0; i < n; ++i)
            {
                threads[i].Join();
                sum += errors[i][0];
            }
            //cudaDeviceSynchronize();
            if (get_current_batch(nets[0]) % interval == 0)
            {
                Console.Write("Syncing... ");
                sync_nets(nets, n, interval);
                Console.Write("Done!\n");
            }
            return (float)sum / (n);
        }
        
        public static float[] get_network_output_layer_gpu(Network net, int i)
        {
            Layer l = net.Layers[i];
            if (l.LayerType != LayerType.Region) Array.Copy(l.Output, l.OutputGpu, l.Outputs * l.Batch);
            return l.Output;
        }
        
        public static float[] get_network_output_gpu(Network net)
        {
            int i;
            for (i = net.N - 1; i > 0; --i) if (net.Layers[i].LayerType != LayerType.Cost) break;
            return get_network_output_layer_gpu(net, i);
        }
        
        public static float[] network_predict_gpu(Network net, float[] input)
        {
            int size = get_network_input_size(net) * net.Batch;
            NetworkState state = new NetworkState
            {
                Index = 0,
                Net = net,
                Input = new float[size],
                Truth = new float[0],
                Train = false,
                Delta = new float[0]
            };
            Array.Copy(input, 0, state.Input, 0, size);
            forward_network_gpu(net, state);
            float[] output = get_network_output_gpu(net);
            return output;
        }

    }
}