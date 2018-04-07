using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public class YoloV2
    {
        public static void Main(string[] argsa)
        {
            CvInvoke.UseOpenCL = true;
            CvInvoke.UseOptimized = true;
            var args = argsa.ToList();
            if (args.Count < 2)
            {
                Console.Error.WriteLine($"usage: {args[0]} <function>");
                return;
            }

            CudaUtils.UseGpu = CudaUtils.HaveGpu();
            if (Utils.find_arg(args.ToList(), "-nogpu"))
            {
                CudaUtils.UseGpu = false;
            }


            switch (args[1])
            {
                case "average":
                    Average(args);
                    break;
                case "yolo":
                    Yolo.run_yolo(args);
                    break;
                case "voxel":
                    Voxel.run_voxel(args);
                    break;
                case "super":
                    Super.run_super(args);
                    break;
                case "detector":
                    Detector.run_detector(args);
                    break;
                case "detect":
                    float thresh = Utils.find_value_arg(args, "-thresh", .24f);
                    string filename = (args.Count > 4) ? args[4] : null;
                    Detector.test_detector("cfg/coco.Data", args[2], args[3], filename, thresh);
                    break;
                case "cifar":
                    CiFar.run_cifar(args);
                    break;
                case "rnn":
                    Rnn.run_char_rnn(args);
                    break;
                case "vid":
                    RnnVid.run_vid_rnn(args);
                    break;
                case "coco":
                    Coco.run_coco(args);
                    break;
                case "classify":
                    Classifier.predict_classifier("cfg/imagenet1k.Data", args[2], args[3], args[4], 5);
                    break;
                case "classifier":
                    Classifier.run_classifier(args);
                    break;
                case "art":
                    Art.run_art(args);
                    break;
                case "tag":
                    Tag.run_tag(args);
                    break;
                case "compare":
                    Compare.run_compare(args);
                    break;
                case "writing":
                    Writing.run_writing(args);
                    break;
                case "3d":
                    LoadArgs.composite_3d(args[2], args[3], args[4], (args.Count > 5) ? int.Parse(args[5]) : 0);
                    break;
                case "test":
                    LoadArgs.test_resize(args[2]);
                    break;
                case "nightmare":
                    Nightmare.run_nightmare(args);
                    break;
                case "rgbgr":
                    rgbgr_net(args[2], args[3], args[4]);
                    break;
                case "reset":
                    reset_normalize_net(args[2], args[3], args[4]);
                    break;
                case "denormalize":
                    denormalize_net(args[2], args[3], args[4]);
                    break;
                case "statistics":
                    statistics_net(args[2], args[3]);
                    break;
                case "normalize":
                    normalize_net(args[2], args[3], args[4]);
                    break;
                case "rescale":
                    rescale_net(args[2], args[3], args[4]);
                    break;
                case "ops":
                    Operations(args[2]);
                    break;
                case "speed":
                    Speed(args[2], (args.Count > 3 && !string.IsNullOrEmpty(args[3])) ? int.Parse(args[3]) : 0);
                    break;
                case "oneoff":
                    Oneoff(args[2], args[3], args[4]);
                    break;
                case "partial":
                    Partial(args[2], args[3], args[4], int.Parse(args[5]));
                    break;
                case "visualize":
                    Visualize(args[2], (args.Count > 3) ? args[3] : null);
                    break;
                case "imtest":
                    LoadArgs.test_resize(args[2]);
                    break;
                default:
                    Console.Error.WriteLine($"Not an option: {args[1]}");
                    break;
            }
        }

        private static void Average(List<string> args)
        {
            string cfgfile = args[2];
            string outfile = args[3];

            Network net = Parser.parse_network_cfg(cfgfile);
            Network sum = Parser.parse_network_cfg(cfgfile);

            string weightfile = args[4];
            Parser.load_weights(sum, weightfile);

            int i, j;
            int n = args.Count - 5;
            for (i = 0; i < n; ++i)
            {
                weightfile = args[i + 5];
                Parser.load_weights(net, weightfile);
                for (j = 0; j < net.N; ++j)
                {
                    Layer l = net.Layers[j];
                    Layer outl = sum.Layers[j];
                    if (l.LayerType == LayerType.Convolutional)
                    {
                        int num = l.N * l.C * l.Size * l.Size;
                        Blas.Axpy_cpu(l.N, 1, l.BiasesComplete, outl.BiasesComplete, l.BiasesIndex, outl.BiasesIndex);
                        Blas.Axpy_cpu(num, 1, l.WeightsComplete, outl.WeightsComplete, l.WeightsIndex, outl.WeightsIndex);
                        if (l.BatchNormalize)
                        {
                            Blas.Axpy_cpu(l.N, 1, l.Scales, outl.Scales);
                            Blas.Axpy_cpu(l.N, 1, l.RollingMean, outl.RollingMean);
                            Blas.Axpy_cpu(l.N, 1, l.RollingVariance, outl.RollingVariance);
                        }
                    }
                    if (l.LayerType == LayerType.Connected)
                    {
                        Blas.Axpy_cpu(l.Outputs, 1, l.BiasesComplete, outl.BiasesComplete, l.BiasesIndex, outl.BiasesIndex);
                        Blas.Axpy_cpu(l.Outputs * l.Inputs, 1, l.WeightsComplete, outl.WeightsComplete, l.WeightsIndex, outl.WeightsIndex);
                    }
                }
            }
            n = n + 1;
            for (j = 0; j < net.N; ++j)
            {
                Layer l = sum.Layers[j];
                if (l.LayerType == LayerType.Convolutional)
                {
                    int num = l.N * l.C * l.Size * l.Size;
                    Blas.Scal_cpu(l.N, 1.0f / n, l.BiasesComplete, 1, l.BiasesIndex);
                    Blas.Scal_cpu(num, 1.0f / n, l.WeightsComplete, 1, l.WeightsIndex);
                    if (l.BatchNormalize)
                    {
                        Blas.Scal_cpu(l.N, 1.0f / n, l.Scales, 1);
                        Blas.Scal_cpu(l.N, 1.0f / n, l.RollingMean, 1);
                        Blas.Scal_cpu(l.N, 1.0f / n, l.RollingVariance, 1);
                    }
                }
                if (l.LayerType == LayerType.Connected)
                {
                    Blas.Scal_cpu(l.Outputs, 1.0f / n, l.BiasesComplete, 1, l.BiasesIndex);
                    Blas.Scal_cpu(l.Outputs * l.Inputs, 1.0f / n, l.WeightsComplete, 1, l.WeightsIndex);
                }
            }
            Parser.save_weights(sum, outfile);
        }

        private static void Speed(string cfgfile, int tics)
        {
            if (tics == 0) tics = 1000;
            Network net = Parser.parse_network_cfg(cfgfile);
            Network.set_batch_network(net, 1);
            int i;
            var sw = new Stopwatch();
            sw.Start();
            Mat im = new Mat(new Size(net.W, net.H), DepthType.Cv8U, net.C);
            for (i = 0; i < tics; ++i)
            {
                Network.network_predict(net, im.GetData());
            }
            sw.Stop();
            var t = sw.Elapsed.Seconds;
            Console.Write($"\n%d evals, %f Seconds\n", tics, t);
            Console.Write($"Speed: %f sec/eval\n", t / tics);
            Console.Write($"Speed: %f Hz\n", tics / t);
        }

        private static void Operations(string cfgfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            int i;
            long ops = 0;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    ops += 2L * l.N * l.Size * l.Size * l.C * l.OutH * l.OutW;
                }
                else if (l.LayerType == LayerType.Connected)
                {
                    ops += 2L * l.Inputs * l.Outputs;
                }
            }
            Console.Write($"Floating Point Operations: %ld\n", ops);
            Console.Write($"Floating Point Operations: %.2f Bn\n", ops / 1000000000.0f);
        }

        private static void Oneoff(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            int oldn = net.Layers[net.N - 2].N;
            int c = net.Layers[net.N - 2].C;
            net.Layers[net.N - 2].N = 9372;
            net.Layers[net.N - 2].BiasesIndex += 5;
            net.Layers[net.N - 2].WeightsIndex += 5 * c;
            if (!string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            net.Layers[net.N - 2].BiasesIndex -= 5;
            net.Layers[net.N - 2].WeightsIndex -= 5 * c;
            net.Layers[net.N - 2].N = oldn;
            Console.Write($"%d\n", oldn);
            Layer l = net.Layers[net.N - 2];
            Blas.Copy_cpu(l.N / 3, l.BiasesComplete, l.BiasesComplete, l.BiasesIndex, l.BiasesIndex + l.N / 3);
            Blas.Copy_cpu(l.N / 3, l.BiasesComplete, l.BiasesComplete, l.BiasesIndex, l.BiasesIndex + 2 * l.N / 3);
            Blas.Copy_cpu(l.N / 3 * l.C, l.WeightsComplete, l.WeightsComplete, l.WeightsIndex, l.WeightsIndex + l.N / 3 * l.C);
            Blas.Copy_cpu(l.N / 3 * l.C, l.WeightsComplete, l.WeightsComplete, l.WeightsIndex, l.WeightsIndex + 2 * l.N / 3 * l.C);
            net.Seen = 0;
            Parser.save_weights(net, outfile);
        }

        private static void Partial(string cfgfile, string weightfile, string outfile, int max)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights_upto(net, weightfile, max);
            }
            net.Seen = 0;
            Parser.save_weights_upto(net, outfile, max);
        }

        private static void rescale_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    l.rescale_weights(2, -.5f);
                    break;
                }
            }
            Parser.save_weights(net, outfile);
        }

        private static void rgbgr_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    l.rgbgr_weights();
                    break;
                }
            }
            Parser.save_weights(net, outfile);
        }

        private static void reset_normalize_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional && l.BatchNormalize)
                {
                    l.denormalize_convolutional_layer();
                }
                if (l.LayerType == LayerType.Connected && l.BatchNormalize)
                {
                    l.denormalize_connected_layer();
                }
                if (l.LayerType == LayerType.Gru && l.BatchNormalize)
                {
                    l.InputZLayer.denormalize_connected_layer();
                    l.InputRLayer.denormalize_connected_layer();
                    l.InputHLayer.denormalize_connected_layer();
                    l.StateZLayer.denormalize_connected_layer();
                    l.StateRLayer.denormalize_connected_layer();
                    l.StateHLayer.denormalize_connected_layer();
                }
            }
            Parser.save_weights(net, outfile);
        }

        private static Layer normalize_layer(Layer l, int n)
        {
            int j;
            l.BatchNormalize = true;
            l.Scales = new float[n];
            for (j = 0; j < n; ++j)
            {
                l.Scales[j] = 1;
            }
            l.RollingMean = new float[n];
            l.RollingVariance = new float[n];
            return l;
        }

        private static void normalize_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional && !l.BatchNormalize)
                {
                    net.Layers[i] = normalize_layer(l, l.N);
                }
                if (l.LayerType == LayerType.Connected && !l.BatchNormalize)
                {
                    net.Layers[i] = normalize_layer(l, l.Outputs);
                }
                if (l.LayerType == LayerType.Gru && l.BatchNormalize)
                {
                    l.InputZLayer = normalize_layer(l.InputZLayer, l.InputZLayer.Outputs);
                    l.InputRLayer = normalize_layer(l.InputRLayer, l.InputRLayer.Outputs);
                    l.InputHLayer = normalize_layer(l.InputHLayer, l.InputHLayer.Outputs);
                    l.StateZLayer = normalize_layer(l.StateZLayer, l.StateZLayer.Outputs);
                    l.StateRLayer = normalize_layer(l.StateRLayer, l.StateRLayer.Outputs);
                    l.StateHLayer = normalize_layer(l.StateHLayer, l.StateHLayer.Outputs);
                    net.Layers[i].BatchNormalize = true;
                }
            }
            Parser.save_weights(net, outfile);
        }

        private static void statistics_net(string cfgfile, string weightfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Connected && l.BatchNormalize)
                {
                    Console.Write($"Connected Layer %d\n", i);
                    l.statistics_connected_layer();
                }
                if (l.LayerType == LayerType.Gru && l.BatchNormalize)
                {
                    Console.Write($"LayerType.Gru Layer %d\n", i);
                    Console.Write($"Input Z\n");
                    l.InputZLayer.statistics_connected_layer();
                    Console.Write($"Input R\n");
                    l.InputRLayer.statistics_connected_layer();
                    Console.Write($"Input H\n");
                    l.InputHLayer.statistics_connected_layer();
                    Console.Write($"State Z\n");
                    l.StateZLayer.statistics_connected_layer();
                    Console.Write($"State R\n");
                    l.StateRLayer.statistics_connected_layer();
                    Console.Write($"State H\n");
                    l.StateHLayer.statistics_connected_layer();
                }
                Console.Write($"\n");
            }
        }

        private static void denormalize_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = false;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional && l.BatchNormalize)
                {
                    l.denormalize_convolutional_layer();
                    net.Layers[i].BatchNormalize = false;
                }
                if (l.LayerType == LayerType.Connected && l.BatchNormalize)
                {
                    l.denormalize_connected_layer();
                    net.Layers[i].BatchNormalize = false;
                }
                if (l.LayerType == LayerType.Gru && l.BatchNormalize)
                {
                    l.InputZLayer.denormalize_connected_layer();
                    l.InputRLayer.denormalize_connected_layer();
                    l.InputHLayer.denormalize_connected_layer();
                    l.StateZLayer.denormalize_connected_layer();
                    l.StateRLayer.denormalize_connected_layer();
                    l.StateHLayer.denormalize_connected_layer();
                    l.InputZLayer.BatchNormalize = false;
                    l.InputRLayer.BatchNormalize = false;
                    l.InputHLayer.BatchNormalize = false;
                    l.StateZLayer.BatchNormalize = false;
                    l.StateRLayer.BatchNormalize = false;
                    l.StateHLayer.BatchNormalize = false;
                    net.Layers[i].BatchNormalize = false;
                }
            }
            Parser.save_weights(net, outfile);
        }

        private static void Visualize(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.visualize_network(net);
            CvInvoke.WaitKey();
        }
    }
}
