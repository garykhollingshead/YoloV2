using System;
using System.Collections.Generic;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class Network
    {
        public float[] Workspace;
        public int N;
        public int Batch;
        public int[] Seen;
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

        public int Adam;
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

        public int GpuIndex;
        public Tree[] Hierarchy;

        public float[][] InputGpu;
        public float[][] TruthGpu;

        public Network()
        {
            InputGpu = new float[0][];
            TruthGpu = new float[0][];
            Scales = new float[0];
            Steps = new int[0];
            Workspace = new float[0];
            Seen = new int[0];
            Layers = new Layer[0];
            Output = new float[0];
        }

        public Network(int n)
        {
            N = n;
            Layers = new Layer[n];
            Seen = new int[1];
            InputGpu = new float[1][];
            TruthGpu = new float[1][];
        }

        int get_current_batch(Network net)
        {
            int batch_num = (net.Seen[0]) / (net.Batch * net.Subdivisions);
            return batch_num;
        }

        void reset_momentum(Network net)
        {
            if (net.Momentum == 0) return;
            net.LearningRate = 0;
            net.Momentum = 0;
            net.Decay = 0;
        }

        float get_current_rate(Network net)
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
                        //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
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

        string get_layer_string(LayerType a)
        {
            return a.ToString();
        }

        void forward_network(Network net, NetworkState state)
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

        void update_network(Network net)
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

        float[] get_network_output(Network net)
        {
            if (CudaUtils.UseGpu) return get_network_output_gpu(net);
            int i;
            for (i = net.N - 1; i > 0; --i) if (net.Layers[i].LayerType != LayerType.Cost) break;
            return net.Layers[i].Output;
        }

        float get_network_cost(Network net)
        {
            int i;
            float sum = 0;
            int count = 0;
            for (i = 0; i < net.N; ++i)
            {
                if (net.Layers.Length > i)
                {
                    sum += net.Layers[i].Cost[0];
                    ++count;
                }
            }
            return sum / count;
        }

        int get_predicted_class_network(Network net)
        {
            float[] outputF = get_network_output(net);
            int k = get_network_output_size(net);
            return Utils.max_index(outputF, k);
        }

        void backward_network(Network net, NetworkState state)
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

        float train_network_datum(Network net, float[] x, float[] y)
        {
            if (CudaUtils.UseGpu) return train_network_datum_gpu(net, x, y);
            NetworkState state = new NetworkState();
            net.Seen[0] += net.Batch;
            state.Index = 0;
            state.Net = net;
            state.Input = x;
            state.Delta = new float[0];
            state.Truth = y;
            state.Train = 1;
            forward_network(net, state);
            backward_network(net, state);
            float error = get_network_cost(net);
            if (((net.Seen[0]) / net.Batch) % net.Subdivisions == 0) update_network(net);
            return error;
        }

        float train_network_sgd(Network net, Data d, int n)
        {
            int batch = net.Batch;
            float[] X = new float[batch * d.X.Cols];
            float[] y = new float[batch * d.Y.Cols];

            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                get_random_batch(d, batch, X, y);
                float err = train_network_datum(net, X, y);
                sum += err;
            }
            return sum / (n * batch);
        }

        float train_network(Network net, Data d)
        {
            int batch = net.Batch;
            int n = d.X.Rows / batch;
            float[] X = new float[batch * d.X.Cols];
            float[] y = new float[batch * d.Y.Cols];

            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                get_next_batch(d, batch, i * batch, X, y);
                float err = train_network_datum(net, X, y);
                sum += err;
            }
            return (float)sum / (n * batch);
        }
        
        float train_network_batch(Network net, Data d, int n)
        {
            int i, j;
            NetworkState state = new NetworkState();
            state.Index = 0;
            state.Net = net;
            state.Train = 1;
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

        void set_batch_network(Network net, int b)
        {
            net.Batch = b;
            int i;
            for (i = 0; i < net.N; ++i)
            {
                net.Layers[i].Batch = b;
                if (net.Layers[i].LayerType == LayerType.Convolutional)
                {
                    Ccudnn_convolutional_setup(net.Layers + i);
                }
            }
        }

        int resize_network(Network net, int w, int h)
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
                    resize_convolutional_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Crop)
                {
                    resize_crop_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Maxpool)
                {
                    resize_maxpool_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Region)
                {
                    resize_region_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Route)
                {
                    resize_route_layer(l, net);
                }
                else if (l.LayerType == LayerType.Reorg)
                {
                    resize_reorg_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Avgpool)
                {
                    resize_avgpool_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Normalization)
                {
                    resize_normalization_layer(l, w, h);
                }
                else if (l.LayerType == LayerType.Cost)
                {
                    resize_cost_layer(l, inputs);
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
                net.Workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
            }
            else
            {
                net.Workspace = new float[1];
            }
            //fprintf(stderr, " Done!\n");
            return 0;
        }

        int get_network_output_size(Network net)
        {
            int i;
            for (i = net.N - 1; i > 0; --i) if (net.Layers[i].LayerType != LayerType.Cost) break;
            return net.Layers[i].Outputs;
        }

        int get_network_input_size(Network net)
        {
            return net.Layers[0].Inputs;
        }

        Layer get_network_detection_layer(Network net)
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

        Image get_network_image_layer(Network net, int i)
        {
            Layer l = net.Layers[i];
            if (l.OutW != 0 && l.OutH != 0 && l.OutC != 0)
            {
                return new Image(l.OutW, l.OutH, l.OutC, l.Output);
            }
            return new Image();
        }

        Image get_network_image(Network net)
        {
            int i;
            for (i = net.N - 1; i >= 0; --i)
            {
                Image m = get_network_image_layer(net, i);
                if (m.H != 0) return m;
            }
            return new Image();
        }

        void visualize_network(Network net)
        {
            Image prev = null;
            int i;
            
            for (i = 0; i < net.N; ++i)
            {
                string buff = $"Layer {i}";
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    prev = visualize_convolutional_layer(l, buff, prev);
                }
            }
        }

        void top_predictions(Network net, int k, int[] index)
        {
            int size = get_network_output_size(net);
            float[] outputF = get_network_output(net);
            Utils.top_k(outputF, size, k, index);
        }
        
        float[] network_predict(Network net, float[] input)
        {
            if (CudaUtils.UseGpu) return network_predict_gpu(net, input);

            NetworkState state = new NetworkState();
            state.Net = net;
            state.Index = 0;
            state.Input = input;
            state.Truth = null;
            state.Train = 0;
            state.Delta = null;
            forward_network(net, state);
            float[] outputF = get_network_output(net);
            return outputF;
        }

        Matrix network_predict_data_multi(Network net, Data test, int n)
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

        Matrix network_predict_data(Network net, Data test)
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

        void print_network(Network net)
        {
            int i, j;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                float[] output = l.Output;
                int n = l.Outputs;
                float mean = mean_array(output, n);
                float vari = variance_array(output, n);
                fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n", i, mean, vari);
                if (n > 100) n = 100;
                for (j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
                if (n == 100) fprintf(stderr, ".....\n");
                fprintf(stderr, "\n");
            }
        }

        void compare_networks(Network n1, Network n2, Data test)
        {
            Matrix g1 = network_predict_data(n1, test);
            Matrix g2 = network_predict_data(n2, test);
            int i;
            int a, b, c, d;
            a = b = c = d = 0;
            for (i = 0; i < g1.Rows; ++i)
            {
                int truth = max_index(test.Y.Vals[i], test.Y.Cols);
                int p1 = max_index(g1.Vals[i], g1.Cols);
                int p2 = max_index(g2.Vals[i], g2.Cols);
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
            printf("%5d %5d\n%5d %5d\n", a, b, c, d);
            float num = Math.Pow((abs(b - c) - 1.), 2.);
            float den = b + c;
            printf("%f\n", num / den);
        }

        float network_accuracy(Network net, Data d)
        {
            Matrix guess = network_predict_data(net, d);
            float acc = matrix_topk_accuracy(d.Y, guess, 1);
            free_matrix(guess);
            return acc;
        }

        float[] network_accuracies(Network net, Data d, int n)
        {
            static float acc[2];
            Matrix guess = network_predict_data(net, d);
            acc[0] = matrix_topk_accuracy(d.Y, guess, 1);
            acc[1] = matrix_topk_accuracy(d.Y, guess, n);
            free_matrix(guess);
            return acc;
        }

        float network_accuracy_multi(Network net, Data d, int n)
        {
            Matrix guess = network_predict_data_multi(net, d, n);
            float acc = matrix_topk_accuracy(d.Y, guess, 1);
            free_matrix(guess);
            return acc;
        }

        void free_network(Network net)
        {
            int i;
            for (i = 0; i < net.N; ++i)
            {
                free_layer(net.Layers[i]);
            }
            free(net.Layers);
            if (*net.InputGpu) cuda_free(*net.InputGpu);
            if (*net.TruthGpu) cuda_free(*net.TruthGpu);
            if (net.InputGpu) free(net.InputGpu);
            if (net.TruthGpu) free(net.TruthGpu);
        }
    }
}