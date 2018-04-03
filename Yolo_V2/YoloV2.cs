using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    class YoloV2
    {
        static void Main(string[] argsa)
        {
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
                    average(args);
                    break;
                case "yolo":
                    run_yolo(args);
                    break;
                case "voxel":
                    run_voxel(args);
                    break;
                case "super":
                    run_super(args);
                    break;
                case "detector":
                    run_detector(args);
                    break;
                case "detect":
                    float thresh = Utils.find_int_arg(args, "-thresh", .24f);
                    string filename = (args.Count > 4) ? args[4] : null;
                    test_detector("cfg/coco.Data", args[2], args[3], filename, thresh);
                    break;
                case "cifar":
                    run_cifar(args);
                    break;
                case "rnn":
                    run_char_rnn(args);
                    break;
                case "vid":
                    run_vid_rnn(args);
                    break;
                case "coco":
                    run_coco(args);
                    break;
                case "classify":
                    predict_classifier("cfg/imagenet1k.Data", args[2], args[3], args[4], 5);
                    break;
                case "classifier":
                    run_classifier(args);
                    break;
                case "art":
                    run_art(args);
                    break;
                case "tag":
                    run_tag(args);
                    break;
                case "compare":
                    run_compare(args);
                    break;
                case "writing":
                    run_writing(args);
                    break;
                case "3d":
                    LoadArgs.composite_3d(args[2], args[3], args[4], (args.Count > 5) ? int.Parse(args[5]) : 0);
                    break;
                case "test":
                    LoadArgs.test_resize(args[2]);
                    break;
                case "nightmare":
                    run_nightmare(args);
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
                    operations(args[2]);
                    break;
                case "speed":
                    speed(args[2], (args.Count > 3 && !string.IsNullOrEmpty(args[3])) ? int.Parse(args[3]) : 0);
                    break;
                case "oneoff":
                    oneoff(args[2], args[3], args[4]);
                    break;
                case "partial":
                    partial(args[2], args[3], args[4], int.Parse(args[5]));
                    break;
                case "visualize":
                    visualize(args[2], (args.Count > 3) ? args[3] : null);
                    break;
                case "imtest":
                    LoadArgs.test_resize(args[2]);
                    break;
                default:
                    Console.Error.WriteLine($"Not an option: {args[1]}");
                    break;
            }
        }

        private static void average(List<string> args)
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
                        Blas.Axpy_cpu(l.N, 1, l.BiasesComplete, outl.BiasesComplete);
                        Blas.Axpy_cpu(num, 1, l.WeightsComplete, outl.WeightsComplete);
                        if (l.BatchNormalize)
                        {
                            Blas.Axpy_cpu(l.N, 1, l.Scales, outl.Scales);
                            Blas.Axpy_cpu(l.N, 1, l.RollingMean, outl.RollingMean);
                            Blas.Axpy_cpu(l.N, 1, l.RollingVariance, outl.RollingVariance);
                        }
                    }
                    if (l.LayerType == LayerType.Connected)
                    {
                        Blas.Axpy_cpu(l.Outputs, 1, l.BiasesComplete, outl.BiasesComplete);
                        Blas.Axpy_cpu(l.Outputs * l.Inputs, 1, l.WeightsComplete, outl.WeightsComplete);
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
                    Blas.Scal_cpu(l.N, 1.0f / n, l.BiasesComplete, 1);
                    Blas.Scal_cpu(num, 1.0f / n, l.WeightsComplete, 1);
                    if (l.BatchNormalize)
                    {
                        Blas.Scal_cpu(l.N, 1.0f / n, l.Scales, 1);
                        Blas.Scal_cpu(l.N, 1.0f / n, l.RollingMean, 1);
                        Blas.Scal_cpu(l.N, 1.0f / n, l.RollingVariance, 1);
                    }
                }
                if (l.LayerType == LayerType.Connected)
                {
                    Blas.Scal_cpu(l.Outputs, 1.0f / n, l.BiasesComplete, 1);
                    Blas.Scal_cpu(l.Outputs * l.Inputs, 1.0f / n, l.WeightsComplete, 1);
                }
            }
            Parser.save_weights(sum, outfile);
        }

        public static void speed(string cfgfile, int tics)
        {
            if (tics == 0) tics = 1000;
            Network net = Parser.parse_network_cfg(cfgfile);
            Network.set_batch_network(net, 1);
            int i;
            var sw = new Stopwatch();
            sw.Start();
            Image im = new Image(net.W, net.H, net.C);
            for (i = 0; i < tics; ++i)
            {
                Network.network_predict(net, im.Data);
            }
            sw.Stop();
            var t = sw.Elapsed.Seconds;
            Console.Write($"\n%d evals, %f Seconds\n", tics, t);
            Console.Write($"Speed: %f sec/eval\n", t / tics);
            Console.Write($"Speed: %f Hz\n", tics / t);
        }

        public static void operations(string cfgfile)
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
                    ops += 2l * l.N * l.Size * l.Size * l.C * l.OutH * l.OutW;
                }
                else if (l.LayerType == LayerType.Connected)
                {
                    ops += 2l * l.Inputs * l.Outputs;
                }
            }
            Console.Write($"Floating Point Operations: %ld\n", ops);
            Console.Write($"Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.0f);
        }

        public static void oneoff(string cfgfile, string weightfile, string outfile)
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

        public static void partial(string cfgfile, string weightfile, string outfile, int max)
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

        public static void rescale_net(string cfgfile, string weightfile, string outfile)
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

        public static void rgbgr_net(string cfgfile, string weightfile, string outfile)
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

        public static void reset_normalize_net(string cfgfile, string weightfile, string outfile)
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

        public static Layer normalize_layer(Layer l, int n)
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

        public static void normalize_net(string cfgfile, string weightfile, string outfile)
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

        public static void statistics_net(string cfgfile, string weightfile)
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

        public static void denormalize_net(string cfgfile, string weightfile, string outfile)
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

        public static void visualize(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.visualize_network(net);
            CvInvoke.WaitKey();
        }

        #region yolo-file

        public static string[] voc_names = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

        public static void train_yolo(string cfgfile, string weightfile)
        {
            string train_images = "/Data.Data/voc/train.txt";
            string backup_directory = "/home/pjreddie/backup/";

            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data buffer = new Data.Data();

            Layer l = net.Layers[net.N - 1];

            int side = l.Side;
            int classes = l.Classes;
            float jitter = l.Jitter;

            string[] paths = Data.Data.GetPaths(train_images);

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.N = imgs;
            args.M = paths.Length;
            args.Classes = classes;
            args.Jitter = jitter;
            args.NumBoxes = side;
            args.D = buffer;
            args.Type = DataType.RegionData;

            args.Angle = net.Angle;
            args.Exposure = net.Exposure;
            args.Saturation = net.Saturation;
            args.Hue = net.Hue;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Start();
                load_thread.Join();
                var train = buffer;
                load_thread = Data.Data.load_data_in_thread(args);

                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    string buff = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }

        public static void print_yolo_detections(FileStream fps, string id, Box[] boxes, float[][] probs, int probsJ, int total, int classes, int w, int h)
        {
            int i;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].X - boxes[i].W / 2.0f;
                float xmax = boxes[i].X + boxes[i].W / 2.0f;
                float ymin = boxes[i].Y - boxes[i].H / 2.0f;
                float ymax = boxes[i].Y + boxes[i].H / 2.0f;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                if (probs[i].Length > probsJ)
                {
                    var buff = Encoding.UTF8.GetBytes($"{id} {probs[i][probsJ]} {xmin} {ymin} {xmax} {ymax}");
                    fps.Write(buff, 0, buff.Length);
                }
            }
        }

        public static void validate_yolo(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            Network.set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum,
                net.Decay);


            string basec = "results/comp4_det_test_";
            string[] paths = Data.Data.GetPaths("/home/pjreddie/Data.Data/voc/2007_test.txt");

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;

            int j;

            Box[] boxes = new Box[l.Side * l.Side * l.N];
            float[][] probs = new float[l.Side * l.Side * l.N][];
            for (j = 0; j < l.Side * l.Side * l.N; ++j) probs[j] = new float[classes];

            int m = paths.Length;
            int i = 0;
            int t;

            float thresh = .001f;
            bool nms = true;
            float iou_thresh = .5f;

            int nthreads = 8;
            Image[] val = new Image[nthreads];
            Image[] val_resized = new Image[nthreads];
            Image[] buf = new Image[nthreads];
            Image[] buf_resized = new Image[nthreads];
            Thread[] thr = new Thread[nthreads];

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Type = DataType.ImageData;

            for (t = 0; t < nthreads; ++t)
            {
                args.Path = paths[i + t];
                args.Im = buf[t];
                args.Resized = buf_resized[t];
                thr[t] = Data.Data.load_data_in_thread(args);
            }

            var sw = new Stopwatch();
            sw.Start();
            for (i = nthreads; i < m + nthreads; i += nthreads)
            {
                Console.Error.Write($"%d\n", i);
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    thr[t].Join();
                    val[t] = buf[t];
                    val_resized[t] = buf_resized[t];
                }

                for (t = 0; t < nthreads && i + t < m; ++t)
                {
                    args.Path = paths[i + t];
                    args.Im = buf[t];
                    args.Resized = buf_resized[t];
                    thr[t] = Data.Data.load_data_in_thread(args);
                }

                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    string path = paths[i + t - nthreads];
                    string id = Utils.Basecfg(path);
                    float[] X = val_resized[t].Data;
                    Network.network_predict(net, X);
                    int w = val[t].W;
                    int h = val[t].H;
                    l.get_detection_boxes(w, h, thresh, probs, boxes, false);
                    if (nms)
                    {
                        Box.do_nms_sort(boxes, probs, l.Side * l.Side * l.N, classes, iou_thresh);
                    }

                    for (j = 0; j < classes; ++j)
                    {
                        string buff = $"{basec}{voc_names[j]}.txt";
                        using (var fps = File.OpenWrite(buff))
                        {
                            print_yolo_detections(fps, id, boxes, probs, j, l.Side * l.Side * l.N, classes, w, h);
                        }
                    }
                }
            }

            sw.Stop();
            Console.Error.Write($"Total Detection Time: %f Seconds\n", sw.Elapsed.Seconds);
        }

        public static void validate_yolo_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);


            string basec = "results/comp4_det_test_";
            string[] paths = Data.Data.GetPaths("Data.Data/voc.2007.test");

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            int side = l.Side;

            int j, k;
            Box[] boxes = new Box[side * side * l.N];
            float[][] probs = new float[side * side * l.N][];
            for (j = 0; j < side * side * l.N; ++j) probs[j] = new float[classes];

            int m = paths.Length;
            int i = 0;

            float thresh = .001f;
            float iou_thresh = .5f;
            bool nms = false;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avg_iou = 0;

            for (i = 0; i < m; ++i)
            {
                string path = paths[i];
                Image orig = LoadArgs.load_image_color(path, 0, 0);
                Image sized = LoadArgs.resize_image(orig, net.W, net.H);
                string id = Utils.Basecfg(path);
                Network.network_predict(net, sized.Data);
                l.get_detection_boxes(orig.W, orig.H, thresh, probs, boxes, true);

                string labelpath;
                Utils.find_replace(path, "images", "labels", out labelpath);
                Utils.find_replace(labelpath, "JPEGImages", "labels", out labelpath);
                Utils.find_replace(labelpath, ".jpg", ".txt", out labelpath);
                Utils.find_replace(labelpath, ".JPEG", ".txt", out labelpath);

                int num_labels = 0;
                BoxLabel[] truth = Data.Data.read_boxes(labelpath, ref num_labels);
                for (k = 0; k < side * side * l.N; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < num_labels; ++j)
                {
                    ++total;
                    Box t = new Box(truth[j].X, truth[j].Y, truth[j].W, truth[j].H);
                    float best_iou = 0;
                    for (k = 0; k < side * side * l.N; ++k)
                    {
                        float iou = Box.box_iou(boxes[k], t);
                        if (probs[k][0] > thresh && iou > best_iou)
                        {
                            best_iou = iou;
                        }
                    }
                    avg_iou += best_iou;
                    if (best_iou > iou_thresh)
                    {
                        ++correct;
                    }
                }

                Console.Error.Write($"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100f * correct / total);
            }
        }

        public static void test_yolo(string cfgfile, string weightfile, string filename, float thresh)
        {
            var alphabet = LoadArgs.load_alphabet();
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Layer l = net.Layers[net.N - 1];
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);
            var sw = new Stopwatch();
            string input = "";
            int j;
            float nms = .4f;
            Box[] boxes = new Box[l.Side * l.Side * l.N];
            float[][] probs = new float[l.Side * l.Side * l.N][];
            for (j = 0; j < l.Side * l.Side * l.N; ++j) probs[j] = new float[l.Classes];
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Image sized = LoadArgs.resize_image(im, net.W, net.H);
                float[] X = sized.Data;
                sw.Start();
                Network.network_predict(net, X);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                l.get_detection_boxes(1, 1, thresh, probs, boxes, false);
                Box.do_nms_sort(boxes, probs, l.Side * l.Side * l.N, l.Classes, nms);

                LoadArgs.draw_detections(im, l.Side * l.Side * l.N, thresh, boxes, probs, voc_names, alphabet, 20);
                LoadArgs.save_image(im, "predictions");
                LoadArgs.show_image(im, "predictions");

                CvInvoke.WaitKey();
                CvInvoke.DestroyAllWindows();
                if (!string.IsNullOrEmpty(filename)) break;
            }
        }

        public static void run_yolo(List<string> args)
        {
            string prefix = Utils.find_int_arg(args, "-prefix", "");
            float thresh = Utils.find_int_arg(args, "-thresh", .2f);
            int cam_index = Utils.find_int_arg(args, "-c", 0);
            int frame_skip = Utils.find_int_arg(args, "-s", 0);
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "test") test_yolo(cfg, weights, filename, thresh);
            else if (args[2] == "train") train_yolo(cfg, weights);
            else if (args[2] == "valid") validate_yolo(cfg, weights);
            else if (args[2] == "recall") validate_yolo_recall(cfg, weights);
            else if (args[2] == "demo") Demo.demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix);
        }

        #endregion

        #region voxelFile

        public static void extract_voxel(string lfile, string rfile, string prefix)
        {
            int w = 1920;
            int h = 1080;
            int shift = 0;
            int count = 0;
            VideoCapture lcap = new VideoCapture(lfile);
            VideoCapture rcap = new VideoCapture(rfile);
            while (true)
            {
                Image l = LoadArgs.get_image_from_stream(lcap);
                Image r = LoadArgs.get_image_from_stream(rcap);
                if (l.W == 0 || r.W == 0) break;
                if (count % 100 == 0)
                {
                    shift = LoadArgs.best_3d_shift_r(l, r, -l.H / 100, l.H / 100);
                    Console.Write($"{shift}\n");
                }
                Image ls = LoadArgs.crop_image(l, (l.W - w) / 2, (l.H - h) / 2, w, h);
                Image rs = LoadArgs.crop_image(r, 105 + (r.W - w) / 2, (r.H - h) / 2 + shift, w, h);
                string buff = $"{prefix}_{count:05}_l";
                LoadArgs.save_image(ls, buff);
                buff = $"{prefix}_{count:05}_r";
                LoadArgs.save_image(rs, buff);
                ++count;
            }
        }

        public static void train_voxel(string cfgfile, string weightfile)
        {
            string train_images = "/Data.Data/imagenet/imagenet1k.train.list";
            string backup_directory = "/home/pjreddie/backup/";

            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data buffer = new Data.Data();

            string[] paths = Data.Data.GetPaths(train_images);

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Scale = 4;
            args.Paths = paths;
            args.N = imgs;
            args.M = paths.Length;
            args.D = buffer;
            args.Type = DataType.SuperData;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            var sw = new Stopwatch();
            //while(i*imgs < N*120){
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Start();
                load_thread.Join();
                var train = buffer;
                load_thread = Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0)
                {
                    string buffa = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buffa);
                }
                if (i % 100 == 0)
                {
                    string buffb = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buffb);
                }
            }
            string buff = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff);
        }

        public static void test_voxel(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);

            var sw = new Stopwatch();
            string input = "";
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Network.resize_network(net, im.W, im.H);
                Console.Write($"%d %d\n", im.W, im.H);

                float[] X = im.Data;
                sw.Start();
                Network.network_predict(net, X);
                Image outi = Network.get_network_image(net);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                LoadArgs.save_image(outi, "outf");

                if (!string.IsNullOrEmpty(filename)) break;
            }
        }


        public static void run_voxel(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "train") train_voxel(cfg, weights);
            else if (args[2] == "test") test_voxel(cfg, weights, filename);
            else if (args[2] == "extract") extract_voxel(args[3], args[4], args[5]);
        }

        #endregion

        #region SuperFile

        public static void train_super(string cfgfile, string weightfile)
        {
            string train_images = "/Data.Data/imagenet/imagenet1k.train.list";
            string backup_directory = "/home/pjreddie/backup/";

            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data buffer = new Data.Data();

            string[] paths = Data.Data.GetPaths(train_images);

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Scale = 4;
            args.Paths = paths;
            args.N = imgs;
            args.M = paths.Length;
            args.D = buffer;
            args.Type = DataType.SuperData;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            var sw = new Stopwatch();
            //while(i*imgs < N*120){
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Start();
                load_thread.Join();
                var train = buffer;
                load_thread = Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);
                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0)
                {
                    string buffa = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buffa);
                }
                if (i % 100 == 0)
                {
                    string buffb = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buffb);
                }
            }
            string buff = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff);
        }

        public static void test_super(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);

            var sw = new Stopwatch();
            string input = "";
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Network.resize_network(net, im.W, im.H);
                Console.Write($"%d %d\n", im.W, im.H);

                float[] X = im.Data;
                sw.Start();
                Network.network_predict(net, X);
                Image outi = Network.get_network_image(net);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                LoadArgs.save_image(outi, "outf");

                if (!string.IsNullOrEmpty(filename)) break;
            }
        }


        public static void run_super(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "train") train_super(cfg, weights);
            else if (args[2] == "test") test_super(cfg, weights, filename);
        }

        #endregion

        #region DetectorFile

        public static void train_detector(string datacfg, string cfgfile, string weightfile, bool clear)
        {
            var options = OptionList.read_data_cfg(datacfg);
            string train_images = OptionList.option_find_str(options, "train", "Data.Data/train.list");
            string backup_directory = OptionList.option_find_str(options, "backup", "/backup/");


            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network[] nets = new Network[1];


            int seed = Utils.Rand.Next();
            int i;
            for (i = 0; i < 1; ++i)
            {
                nets[i] = Parser.parse_network_cfg(cfgfile);
                if (string.IsNullOrEmpty(weightfile))
                {
                    Parser.load_weights(nets[i], weightfile);
                }
                if (clear) nets[i].Seen = 0;
                nets[i].LearningRate *= 1;
            }

            Network net = nets[0];

            int imgs = net.Batch * net.Subdivisions * 1;
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            Data.Data buffer = new Data.Data();

            Layer l = net.Layers[net.N - 1];

            int classes = l.Classes;
            float jitter = l.Jitter;

            string[] paths = Data.Data.GetPaths(train_images);

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.N = imgs;
            args.M = paths.Length;
            args.Classes = classes;
            args.Jitter = jitter;
            args.NumBoxes = l.MaxBoxes;
            args.D = buffer;
            args.Type = DataType.DetectionData;
            args.Threads = 8;

            args.Angle = net.Angle;
            args.Exposure = net.Exposure;
            args.Saturation = net.Saturation;
            args.Hue = net.Hue;

            Thread load_thread = Data.Data.load_data(args);
            var sw = new Stopwatch();
            int count = 0;
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                Data.Data train;
                if (l.Random && count++ % 10 == 0)
                {
                    Console.Write($"Resizing\n");
                    int dim = (Utils.Rand.Next() % 10 + 10) * 32;
                    if (Network.get_current_batch(net) + 100 > net.MaxBatches) dim = 544;
                    Console.Write($"%d\n", dim);
                    args.W = dim;
                    args.H = dim;

                    load_thread.Join();
                    load_thread = Data.Data.load_data(args);

                    for (i = 0; i < 1; ++i)
                    {
                        Network.resize_network(nets[i], dim, dim);
                    }
                    net = nets[0];
                }
                sw.Start();
                load_thread.Join();
                train = buffer;
                load_thread = Data.Data.load_data(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss;
                loss = Network.train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                i = Network.get_current_batch(net);
                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {

                    string buffa = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buffa);
                }
            }

            string buff = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff);
        }

        static void print_cocos(FileStream fp, string image_path, Box[] boxes, float[][] probs, int num_boxes, int classes, int w, int h)
        {
            int i, j;
            int image_id = get_coco_image_id(image_path);
            for (i = 0; i < num_boxes; ++i)
            {
                float xmin = boxes[i].X - boxes[i].W / 2.0f;
                float xmax = boxes[i].X + boxes[i].W / 2.0f;
                float ymin = boxes[i].Y - boxes[i].H / 2.0f;
                float ymax = boxes[i].Y + boxes[i].H / 2.0f;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                float bx = xmin;
                float by = ymin;
                float bw = xmax - xmin;
                float bh = ymax - ymin;

                for (j = 0; j < classes; ++j)
                {
                    if (probs[i].Length < j)
                    {
                        var temp = Encoding.UTF8.GetBytes($"{{\"image_id\":{image_id}, \"category_id\":{coco_ids[j]}, \"bbox\":[{bx}, {by}, {bw}, {bh}], \"score\":{probs[i][j]}}},\n");
                        fp.Write(temp, 0, temp.Length);
                    }
                }
            }
        }

        public static void print_detector_detections(FileStream[] fps, string id, Box[] boxes, float[][] probs, int total, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].X - boxes[i].W / 2.0f;
                float xmax = boxes[i].X + boxes[i].W / 2.0f;
                float ymin = boxes[i].Y - boxes[i].H / 2.0f;
                float ymax = boxes[i].Y + boxes[i].H / 2.0f;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                for (j = 0; j < classes; ++j)
                {
                    if (probs[i].Length > j)
                    {
                        var temp = Encoding.UTF8.GetBytes($"{id} {probs[i][j]} {xmin} {ymin} {xmax} {ymax}\n");
                        fps[j].Write(temp, 0, temp.Length);
                    }
                }
            }
        }

        public static void print_imagenet_detections(FileStream fp, int id, Box[] boxes, float[][] probs, int total, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].X - boxes[i].W / 2.0f;
                float xmax = boxes[i].X + boxes[i].W / 2.0f;
                float ymin = boxes[i].Y - boxes[i].H / 2.0f;
                float ymax = boxes[i].Y + boxes[i].H / 2.0f;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                for (j = 0; j < classes; ++j)
                {
                    if (probs[i].Length > j)
                    {
                        var temp = Encoding.UTF8.GetBytes($"{id} {j + 1} {probs[i][j]} {xmin} {ymin} {xmax} {ymax}\n");
                        fp.Write(temp, 0, temp.Length);
                    }
                }
            }
        }

        public static void validate_detector(string datacfg, string cfgfile, string weightfile)
        {
            int j;
            var options = OptionList.read_data_cfg(datacfg);
            string valid_images = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            string name_list = OptionList.option_find_str(options, "names", "Data.Data/names.list");
            string prefix = OptionList.option_find_str(options, "results", "results");
            string[] names = Data.Data.get_labels(name_list);
            string mapf = OptionList.option_find_str(options, "map", "");
            int[] map = new int[0];
            if (!string.IsNullOrEmpty(mapf)) map = Utils.read_map(mapf);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);


            string basec = "comp4_det_test_";
            string[] paths = Data.Data.GetPaths(valid_images);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            bool coco = false;
            var sw = new Stopwatch();

            string type = OptionList.option_find_str(options, "eval", "voc");
            FileStream fp = null;
            FileStream[] fps = new FileStream[0];
            try
            {
                bool imagenet = false;
                if (type == "coco")
                {
                    fp = File.OpenWrite($"{prefix}/coco_results.json");
                    var temp = Encoding.UTF8.GetBytes("[\n");
                    fp.Write(temp, 0, temp.Length);
                    coco = true;
                }
                else if (type == "imagenet")
                {
                    fp = File.OpenWrite($"{prefix}/imagenet-detection.txt");
                    imagenet = true;
                    classes = 200;
                }
                else
                {
                    fps = new FileStream[classes];
                    for (j = 0; j < classes; ++j)
                    {
                        fps[j] = File.OpenWrite($"{prefix}/{basec}{names[j]}.txt");
                    }
                }

                Box[] boxes = new Box[l.W * l.H * l.N];
                float[][] probs = new float[l.W * l.H * l.N][];
                for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = new float[classes];

                int m = paths.Length;
                int i = 0;
                int t;

                float thresh = .005f;
                float nms = .45f;

                int nthreads = 4;
                Image[] val = new Image[nthreads];
                Image[] val_resized = new Image[nthreads];
                Image[] buf = new Image[nthreads];
                Image[] buf_resized = new Image[nthreads];
                Thread[] thr = new Thread[nthreads];

                LoadArgs args = new LoadArgs();
                args.W = net.W;
                args.H = net.H;
                args.Type = DataType.ImageData;

                for (t = 0; t < nthreads; ++t)
                {
                    args.Path = paths[i + t];
                    args.Im = buf[t];
                    args.Resized = buf_resized[t];
                    thr[t] = Data.Data.load_data_in_thread(args);
                }

                sw.Start();
                for (i = nthreads; i < m + nthreads; i += nthreads)
                {
                    Console.Error.Write($"%d\n", i);
                    for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                    {
                        thr[t].Join();
                        val[t] = buf[t];
                        val_resized[t] = buf_resized[t];
                    }
                    for (t = 0; t < nthreads && i + t < m; ++t)
                    {
                        args.Path = paths[i + t];
                        args.Im = buf[t];
                        args.Resized = buf_resized[t];
                        thr[t] = Data.Data.load_data_in_thread(args);
                    }
                    for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                    {
                        string path = paths[i + t - nthreads];
                        string id = Utils.Basecfg(path);
                        float[] X = val_resized[t].Data;
                        Network.network_predict(net, X);
                        int w = val[t].W;
                        int h = val[t].H;
                        Layer.get_region_boxes(l, w, h, thresh, probs, boxes, false, map);
                        if (nms != 0) Box.do_nms_sort(boxes, probs, l.W * l.H * l.N, classes, nms);
                        if (coco)
                        {
                            print_cocos(fp, path, boxes, probs, l.W * l.H * l.N, classes, w, h);
                        }
                        else if (imagenet)
                        {
                            print_imagenet_detections(fp, i + t - nthreads + 1, boxes, probs, l.W * l.H * l.N, classes, w, h);
                        }
                        else
                        {
                            print_detector_detections(fps, id, boxes, probs, l.W * l.H * l.N, classes, w, h);
                        }
                    }
                }

            }
            finally
            {
                for (j = 0; j < classes; ++j)
                {
                    fps?[j]?.Close();
                }
                if (coco)
                {
                    fp?.Seek(-2, SeekOrigin.Current);
                    var temp = Encoding.UTF8.GetBytes("\n]\n");
                    fp?.Write(temp, 0, temp.Length);
                }
                fp?.Close();
            }

            sw.Stop();
            Console.Error.Write($"Total Detection Time: %f Seconds\n", sw.Elapsed.Seconds);
        }

        public static void validate_detector_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            string[] paths = Data.Data.GetPaths("Data.Data/voc.2007.test");

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;

            int j, k;
            Box[] boxes = new Box[l.W * l.H * l.N];
            float[][] probs = new float[l.W * l.H * l.N][];
            for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = new float[classes];

            int m = paths.Length;
            int i = 0;

            float thresh = .001f;
            float iou_thresh = .5f;
            float nms = .4f;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avg_iou = 0;

            for (i = 0; i < m; ++i)
            {
                string path = paths[i];
                Image orig = LoadArgs.load_image_color(path, 0, 0);
                Image sized = LoadArgs.resize_image(orig, net.W, net.H);
                string id = Utils.Basecfg(path);
                Network.network_predict(net, sized.Data);
                Layer.get_region_boxes(l, 1, 1, thresh, probs, boxes, true, new int[0]);
                if (nms != 0) Box.do_nms(boxes, probs, l.W * l.H * l.N, 1, nms);

                string labelpath;
                Utils.find_replace(path, "images", "labels", out labelpath);
                Utils.find_replace(labelpath, "JPEGImages", "labels", out labelpath);
                Utils.find_replace(labelpath, ".jpg", ".txt", out labelpath);
                Utils.find_replace(labelpath, ".JPEG", ".txt", out labelpath);

                int num_labels = 0;
                BoxLabel[] truth = Data.Data.read_boxes(labelpath, ref num_labels);
                for (k = 0; k < l.W * l.H * l.N; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < num_labels; ++j)
                {
                    ++total;
                    Box t = new Box(truth[j].X, truth[j].Y, truth[j].W, truth[j].H);
                    float best_iou = 0;
                    for (k = 0; k < l.W * l.H * l.N; ++k)
                    {
                        float iou = Box.box_iou(boxes[k], t);
                        if (probs[k][0] > thresh && iou > best_iou)
                        {
                            best_iou = iou;
                        }
                    }
                    avg_iou += best_iou;
                    if (best_iou > iou_thresh)
                    {
                        ++correct;
                    }
                }

                Console.Error.Write($"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100f * correct / total);
            }
        }

        public static void test_detector(string datacfg, string cfgfile, string weightfile, string filename, float thresh)
        {
            var options = OptionList.read_data_cfg(datacfg);
            string name_list = OptionList.option_find_str(options, "names", "Data.Data/names.list");
            string[] names = Data.Data.get_labels(name_list);

            Image[][] alphabet = LoadArgs.load_alphabet();
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);
            var sw = new Stopwatch();

            string input = "";
            int j;
            float nms = .4f;
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Image sized = LoadArgs.resize_image(im, net.W, net.H);
                Layer l = net.Layers[net.N - 1];

                Box[] boxes = new Box[l.W * l.H * l.N];
                float[][] probs = new float[l.W * l.H * l.N][];
                for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = new float[l.Classes];

                float[] X = sized.Data;
                sw.Start();
                Network.network_predict(net, X);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                Layer.get_region_boxes(l, 1, 1, thresh, probs, boxes, false, new int[0]);
                if (nms != 0) Box.do_nms_sort(boxes, probs, l.W * l.H * l.N, l.Classes, nms);
                LoadArgs.draw_detections(im, l.W * l.H * l.N, thresh, boxes, probs, names, alphabet, l.Classes);
                LoadArgs.save_image(im, "predictions");
                LoadArgs.show_image(im, "predictions");

                CvInvoke.WaitKey();
                CvInvoke.DestroyAllWindows();
                if (!string.IsNullOrEmpty(filename)) break;
            }
        }

        public static void run_detector(List<string> args)
        {
            string prefix = Utils.find_int_arg(args, "-prefix", "");
            float thresh = Utils.find_int_arg(args, "-thresh", .24f);
            int cam_index = Utils.find_int_arg(args, "-c", 0);
            int frame_skip = Utils.find_int_arg(args, "-s", 0);
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            bool clear = Utils.find_arg(args, "-clear");

            string datacfg = args[3];
            string cfg = args[4];
            string weights = (args.Count > 5) ? args[5] : "";
            string filename = (args.Count > 6) ? args[6] : "";
            if (args[2] == "test") test_detector(datacfg, cfg, weights, filename, thresh);
            else if (args[2] == "train") train_detector(datacfg, cfg, weights, clear);
            else if (args[2] == "valid") validate_detector(datacfg, cfg, weights);
            else if (args[2] == "recall") validate_detector_recall(cfg, weights);
            else if (args[2] == "demo")
            {
                var options = OptionList.read_data_cfg(datacfg);
                int classes = OptionList.option_find_int(options, "classes", 20);
                string name_list = OptionList.option_find_str(options, "names", "Data.Data/names.list");
                string[] names = Data.Data.get_labels(name_list);
                Demo.demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix);
            }
        }

        #endregion

        #region CiFarFile

        public static void train_cifar(string cfgfile, string weightfile)
        {

            float avg_loss = -1;
            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            string backup_directory = "/home/pjreddie/backup/";
            int classes = 10;
            int N = 50000;

            string[] labels = Data.Data.get_labels("Data.Data/cifar/labels.txt");
            int epoch = (net.Seen) / N;
            Data.Data train = Data.Data.load_all_cifar10();
            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();

                float loss = Network.train_network_sgd(net, train, 1);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95f + loss * .05f;
                sw.Stop();
                Console.Write(
                    $"{Network.get_current_batch(net)}, {net.Seen / N:.3}: {loss}, {avg_loss} avg, {Network.get_current_rate(net)} rate, {sw.Elapsed.Seconds} seconds, {net.Seen} images\n");
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;

                    string buff = $"{backup_directory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}.Weights";
            Parser.save_weights(net, buff2);
        }

        public static void train_cifar_distill(string cfgfile, string weightfile)
        {

            float avg_loss = -1;
            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            string backup_directory = "/home/pjreddie/backup/";
            int classes = 10;
            int N = 50000;

            string[] labels = Data.Data.get_labels("Data.Data/cifar/labels.txt");
            int epoch = (net.Seen) / N;

            Data.Data train = Data.Data.load_all_cifar10();
            Matrix soft = new Matrix("results/ensemble.csv");

            float weight = .9f;
            soft.scale_matrix(weight);
            train.Y.scale_matrix(1.0f - weight);
            Matrix.matrix_add_matrix(soft, train.Y);

            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();

                float loss = Network.train_network_sgd(net, train, 1);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95f + loss * .05f;
                sw.Stop();
                Console.Write(
                    $"{Network.get_current_batch(net)}, {net.Seen / N:.3}: {loss}, {avg_loss} avg, {Network.get_current_rate(net)} rate, {sw.Elapsed.Seconds} seconds, {net.Seen} images\n");
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;

                    string buff = $"{backup_directory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}.Weights";
            Parser.save_weights(net, buff2);
        }

        public static void test_cifar_multi(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);


            float avg_acc = 0;
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            int i;
            for (i = 0; i < test.X.Rows; ++i)
            {
                Image im = new Image(32, 32, 3, test.X.Vals[i]);

                float[] pred = new float[10];

                float[] p = Network.network_predict(net, im.Data);
                Blas.Axpy_cpu(10, 1, p, pred);
                LoadArgs.flip_image(im);
                p = Network.network_predict(net, im.Data);
                Blas.Axpy_cpu(10, 1, p, pred);

                int index = Utils.max_index(pred, 10);
                int sclass = Utils.max_index(test.Y.Vals[i], 10);
                if (index == sclass) avg_acc += 1;
                Console.Write($"%4d: %.2f%%\n", i, 100f * avg_acc / (i + 1));
            }
        }

        public static void test_cifar(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var sw = new Stopwatch();
            float avg_acc = 0;
            float avg_top5 = 0;
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            sw.Start();

            float[] acc = Network.network_accuracies(net, test, 2);
            avg_acc += acc[0];
            sw.Stop();
            Console.Write($"top1: %f, %lf seconds, %d images\n", avg_acc, sw.Elapsed.Seconds, test.X.Rows);
        }

        public static void extract_cifar()
        {
            string[] labels = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
            int i;
            Data.Data train = Data.Data.load_all_cifar10();
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");
            for (i = 0; i < train.X.Rows; ++i)
            {
                Image im = new Image(32, 32, 3, train.X.Vals[i]);
                int sclass = Utils.max_index(train.Y.Vals[i], 10);
                string buff = $"Data.Data/cifar/train/{i}_{labels[sclass]}";
                LoadArgs.save_image_png(im, buff);
            }
            for (i = 0; i < test.X.Rows; ++i)
            {
                Image im = new Image(32, 32, 3, test.X.Vals[i]);
                int sclass = Utils.max_index(test.Y.Vals[i], 10);
                string buff = $"Data.Data/cifar/test/{i}_{labels[sclass]}";
                LoadArgs.save_image_png(im, buff);
            }
        }

        public static void test_cifar_csv(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            Matrix pred = Network.network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.Rows; ++i)
            {
                Image im = new Image(32, 32, 3, test.X.Vals[i]);
                LoadArgs.flip_image(im);
            }
            Matrix pred2 = Network.network_predict_data(net, test);
            pred.scale_matrix(.5f);
            pred2.scale_matrix(.5f);
            Matrix.matrix_add_matrix(pred2, pred);

            pred.to_csv();
            Console.Error.Write($"Accuracy: {Matrix.matrix_topk_accuracy(test.Y, pred, 1)}\n");
        }

        public static void test_cifar_csvtrain(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            Data.Data test = Data.Data.load_all_cifar10();

            Matrix pred = Network.network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.Rows; ++i)
            {
                Image im = new Image(32, 32, 3, test.X.Vals[i]);
                LoadArgs.flip_image(im);
            }
            Matrix pred2 = Network.network_predict_data(net, test);
            pred.scale_matrix(.5f);
            pred2.scale_matrix(.5f);
            Matrix.matrix_add_matrix(pred2, pred);

            pred.to_csv();
            Console.Error.Write($"Accuracy: %f\n", Matrix.matrix_topk_accuracy(test.Y, pred, 1));
        }

        public static void eval_cifar_csv()
        {
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            Matrix pred = new Matrix("results/combined.csv");
            Console.Error.Write($"%d %d\n", pred.Rows, pred.Cols);

            Console.Error.Write($"Accuracy: %f\n", Matrix.matrix_topk_accuracy(test.Y, pred, 1));
        }


        public static void run_cifar(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_cifar(cfg, weights);
            else if (args[2] == "extract") extract_cifar();
            else if (args[2] == "distill") train_cifar_distill(cfg, weights);
            else if (args[2] == "test") test_cifar(cfg, weights);
            else if (args[2] == "multi") test_cifar_multi(cfg, weights);
            else if (args[2] == "csv") test_cifar_csv(cfg, weights);
            else if (args[2] == "csvtrain") test_cifar_csvtrain(cfg, weights);
            else if (args[2] == "eval") eval_cifar_csv();
        }

        #endregion

        #region RnnFile

        public static int[] read_tokenized_data(string filename, out int read)
        {
            var lines = File.ReadAllLines(filename);
            int n;
            List<int> ns = new List<int>();
            foreach (var line in lines)
            {
                var tokens = line.Split(' ');
                foreach (var token in tokens)
                {
                    if (int.TryParse(token, out n))
                    {
                        ns.Add(n);
                    }
                    else
                    {
                        read = ns.Count;
                        return ns.ToArray();
                    }
                }
            }
            read = ns.Count;
            return ns.ToArray();
        }

        public static string[] read_tokens(string filename, out int read)
        {
            var lines = File.ReadAllLines(filename);
            read = lines.Length;
            return lines;
        }

        public static FloatPair get_rnn_token_data(int[] tokens, int[] offsets, int characters, int len, int batch, int steps)
        {
            float[] x = new float[batch * steps * characters];
            float[] y = new float[batch * steps * characters];
            int i, j;
            for (i = 0; i < batch; ++i)
            {
                for (j = 0; j < steps; ++j)
                {
                    int curr = tokens[(offsets[i]) % len];
                    int next = tokens[(offsets[i] + 1) % len];

                    x[(j * batch + i) * characters + curr] = 1;
                    y[(j * batch + i) * characters + next] = 1;

                    offsets[i] = (offsets[i] + 1) % len;

                    if (curr >= characters || curr < 0 || next >= characters || next < 0)
                    {
                        Utils.Error("Bad char");
                    }
                }
            }
            FloatPair p = new FloatPair();
            p.X = x;
            p.Y = y;
            return p;
        }

        public static FloatPair get_rnn_data(byte[] text, int[] offsets, int characters, int len, int batch, int steps)
        {
            float[] x = new float[batch * steps * characters];
            float[] y = new float[batch * steps * characters];
            int i, j;
            for (i = 0; i < batch; ++i)
            {
                for (j = 0; j < steps; ++j)
                {
                    byte curr = text[(offsets[i]) % len];
                    byte next = text[(offsets[i] + 1) % len];

                    x[(j * batch + i) * characters + curr] = 1;
                    y[(j * batch + i) * characters + next] = 1;

                    offsets[i] = (offsets[i] + 1) % len;

                    if (curr > 255 || curr <= 0 || next > 255 || next <= 0)
                    {
                        Utils.Error("Bad char");
                    }
                }
            }
            FloatPair p = new FloatPair();
            p.X = x;
            p.Y = y;
            return p;
        }

        public static void reset_rnn_state(Network net, int b)
        {
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.StateGpu.Length != 0)
                {
                    Blas.fill_ongpu(l.Outputs, 0, l.StateGpu, 1, l.Outputs * b);
                }
            }
        }

        public static void train_char_rnn(string cfgfile, string weightfile, string filename, bool clear, bool tokenized)
        {

            byte[] text = new byte[0];
            int[] tokens = new int[0];
            int size;
            if (tokenized)
            {
                tokens = read_tokenized_data(filename, out size);
            }
            else
            {
                using (var fstream = File.OpenRead(filename))
                {
                    size = (int)fstream.Length;
                    text = new byte[size];
                    fstream.Read(text, 0, text.Length);
                }
            }

            string backup_directory = "/home/pjreddie/backup/";
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            int inputs = Network.get_network_input_size(net);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int batch = net.Batch;
            int steps = net.TimeSteps;
            if (clear) net.Seen = 0;
            int i = (net.Seen) / net.Batch;

            int streams = batch / steps;
            int[] offsets = new int[streams];
            int j;
            for (j = 0; j < streams; ++j)
            {
                offsets[j] = Utils.Rand.Next() % size;
            }

            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Reset();
                sw.Start();
                FloatPair p;
                if (tokenized)
                {
                    p = get_rnn_token_data(tokens, offsets, inputs, size, streams, steps);
                }
                else
                {
                    p = get_rnn_data(text, offsets, inputs, size, streams, steps);
                }

                float loss = Network.train_network_datum(net, p.X, p.Y) / (batch);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                int chars = Network.get_current_batch(net) * batch;
                sw.Stop();
                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, (float)chars / size);

                for (j = 0; j < streams; ++j)
                {
                    //Console.Write($"%d\n", j);
                    if (Utils.Rand.Next() % 10 == 0)
                    {
                        //Console.Error.Write($"Reset\n");
                        offsets[j] = Utils.Rand.Next() % size;
                        reset_rnn_state(net, j);
                    }
                }

                if (i % 1000 == 0)
                {

                    string buff = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (i % 10 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }

        public static void print_symbol(int n, string[] tokens)
        {
            if (!string.IsNullOrEmpty(tokens[n]))
            {
                Console.Write($"%s ", tokens[n]);
            }
            else
            {
                Console.Write($"%c", n);
            }
        }

        public static void test_char_rnn(string cfgfile, string weightfile, int num, string seed, float temp, int rseed, string token_file)
        {
            string[] tokens = new string[0];
            if (!string.IsNullOrEmpty(token_file))
            {
                int n;
                tokens = read_tokens(token_file, out n);
            }

            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].Temperature = temp;
            int c = 0;
            int len = seed.Length;
            float[] input = new float[inputs];


            for (i = 0; i < len - 1; ++i)
            {
                c = seed[i];
                input[c] = 1;
                Network.network_predict(net, input);
                input[c] = 0;
                print_symbol(c, tokens);
            }
            if (len != 0) c = seed[len - 1];
            print_symbol(c, tokens);
            for (i = 0; i < num; ++i)
            {
                input[c] = 1;
                float[] outf = Network.network_predict(net, input);
                input[c] = 0;
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                c = Utils.sample_array(outf, inputs);
                print_symbol(c, tokens);
            }
            Console.Write($"\n");
        }

        public static void test_tactic_rnn(string cfgfile, string weightfile, int num, float temp, int rseed, string token_file)
        {
            string[] tokens = new string[0];
            if (!string.IsNullOrEmpty(token_file))
            {
                int n;
                tokens = read_tokens(token_file, out n);
            }

            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].Temperature = temp;
            float[] input = new float[inputs];
            float[] outf = new float[0];
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);
            foreach (var c in readLine)
            {
                input[c] = 1;
                outf = Network.network_predict(net, input);
                input[c] = 0;
            }
            for (i = 0; i < num; ++i)
            {
                var c = readLine.Last();
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                int next = Utils.sample_array(outf, inputs);
                if (c == '.' && next == '\n') break;
                c = (char)next;
                print_symbol(c, tokens);

                input[c] = 1;
                outf = Network.network_predict(net, input);
                input[c] = 0;
            }
            Console.Write($"\n");
        }

        public static void valid_tactic_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int count = 0;
            int words = 1;
            int c;
            int len = seed.Length;
            float[] input = new float[inputs];

            for (var i = 0; i < len; ++i)
            {
                c = seed[i];
                input[(int)c] = 1;
                Network.network_predict(net, input);
                input[(int)c] = 0;
            }
            float sum = 0;
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);
            float log2 = (float)Math.Log(2);
            bool iIn = false;
            for (int i = 0; i < readLine.Length - 1; i++)
            {
                c = readLine[i];
                var next = readLine[i + 1];
                if (next < 0 || next >= 255) Utils.Error("Out of range character");
                input[c] = 1;
                float[] outf = Network.network_predict(net, input);
                input[c] = 0;

                if (c == '.' && next == '\n') iIn = false;
                if (!iIn)
                {
                    if (c == '>' && next == '>')
                    {
                        iIn = true;
                        ++words;
                    }
                    continue;
                }
                ++count;
                sum += (float)Math.Log(outf[next]) / log2;
                Console.Write($"%d %d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, words, (float)Math.Pow(2, -sum / count), (float)Math.Pow(2, -sum / words));
            }
        }

        public static void valid_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int count = 0;
            int words = 1;
            int c;
            int len = seed.Length;
            float[] input = new float[inputs];
            for (var i = 0; i < len; ++i)
            {
                c = seed[i];
                input[(int)c] = 1;
                Network.network_predict(net, input);
                input[(int)c] = 0;
            }
            float sum = 0;
            float log2 = (float)Math.Log(2);
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);

            for (var i = 0; i < readLine.Length - 1; i++)
            {
                c = readLine[i];
                var next = readLine[i + 1];
                if (next < 0 || next >= 255) Utils.Error("Out of range character");
                ++count;
                if (next == ' ' || next == '\n' || next == '\t') ++words;
                input[c] = 1;
                float[] outf = Network.network_predict(net, input);
                input[c] = 0;
                sum += (float)Math.Log(outf[next]) / log2;
                Console.Write($"%d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, (float)Math.Pow(2, -sum / count), (float)Math.Pow(2, -sum / words));
            }
        }

        public static void vec_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int c;
            int seed_len = seed.Length;
            float[] input = new float[inputs];
            int i;
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);
            foreach (var line in readLine.Split('\n'))
            {
                reset_rnn_state(net, 0);
                for (i = 0; i < seed_len; ++i)
                {
                    c = seed[i];
                    input[(int)c] = 1;
                    Network.network_predict(net, input);
                    input[(int)c] = 0;
                }
                Utils.Strip(line);
                int str_len = line.Length;
                for (i = 0; i < str_len; ++i)
                {
                    c = line[i];
                    input[(int)c] = 1;
                    Network.network_predict(net, input);
                    input[(int)c] = 0;
                }
                c = ' ';
                input[(int)c] = 1;
                Network.network_predict(net, input);
                input[(int)c] = 0;

                Layer l = net.Layers[0];
                Array.Copy(l.OutputGpu, l.Output, l.Outputs);
                Console.Write($"%s", line);
                for (i = 0; i < l.Outputs; ++i)
                {
                    Console.Write($",%g", l.Output[i]);
                }
                Console.Write($"\n");
            }
        }

        public static void run_char_rnn(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }
            string filename = Utils.find_int_arg(args, "-file", "Data.Data/shakespeare.txt");
            string seed = Utils.find_int_arg(args, "-seed", "\n\n");
            int len = Utils.find_int_arg(args, "-len", 1000);
            float temp = Utils.find_int_arg(args, "-temp", .7f);
            int rseed = Utils.find_int_arg(args, "-srand", (int)DateTime.Now.Ticks);
            bool clear = Utils.find_arg(args, "-clear");
            bool tokenized = Utils.find_arg(args, "-tokenized");
            string tokens = Utils.find_int_arg(args, "-tokens", "");

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_char_rnn(cfg, weights, filename, clear, tokenized);
            else if (args[2] == "valid") valid_char_rnn(cfg, weights, seed);
            else if (args[2] == "validtactic") valid_tactic_rnn(cfg, weights, seed);
            else if (args[2] == "vec") vec_char_rnn(cfg, weights, seed);
            else if (args[2] == "generate") test_char_rnn(cfg, weights, len, seed, temp, rseed, tokens);
            else if (args[2] == "generatetactic") test_tactic_rnn(cfg, weights, len, temp, rseed, tokens);
        }

        #endregion

        #region RnnVidFile


        public static FloatPair get_rnn_vid_data(Network net, string[] files, int n, int batch, int steps)
        {
            int b;
            Image out_im = Network.get_network_image(net);
            int output_size = out_im.W * out_im.H * out_im.C;
            Console.Write($"%d %d %d\n", out_im.W, out_im.H, out_im.C);
            float[] feats = new float[net.Batch * batch * output_size];
            for (b = 0; b < batch; ++b)
            {
                int input_size = net.W * net.H * net.C;
                float[] input = new float[input_size * net.Batch];
                string filename = files[Utils.Rand.Next() % n];
                using (VideoCapture cap = new VideoCapture(filename))
                {
                    int frames = (int)cap.GetCaptureProperty(CapProp.FrameCount);
                    int index = Utils.Rand.Next() % (frames - steps - 2);
                    if (frames < (steps + 4))
                    {
                        --b;
                        continue;
                    }

                    Console.Write($"frames: %d, index: %d\n", frames, index);
                    cap.SetCaptureProperty(CapProp.PosFrames, index);

                    int i;
                    for (i = 0; i < net.Batch; ++i)
                    {
                        Mat src = cap.QueryFrame();
                        Image im = new Image(src);

                        LoadArgs.rgbgr_image(im);
                        Image re = LoadArgs.resize_image(im, net.W, net.H);
                        Array.Copy(re.Data, 0, input, i * input_size, input_size);
                    }

                    float[] output = Network.network_predict(net, input);

                    for (i = 0; i < net.Batch; ++i)
                    {
                        Array.Copy(output, i * output_size, feats, (b + i * batch) * output_size, output_size);
                    }
                }
            }

            FloatPair p = new FloatPair();
            p.X = feats;
            p.Y = new float[feats.Length - output_size * batch];
            Array.Copy(feats, output_size * batch, p.Y, 0, p.Y.Length);

            return p;
        }


        public static void train_vid_rnn(string cfgfile, string weightfile)
        {
            string train_videos = "Data.Data/vid/train.txt";
            string backup_directory = "/home/pjreddie/backup/";

            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;

            string[] paths = Data.Data.GetPaths(train_videos);
            int N = paths.Length;
            var sw = new Stopwatch();
            int steps = net.TimeSteps;
            int batch = net.Batch / net.TimeSteps;

            Network extractor = Parser.parse_network_cfg("cfg/extractor.cfg");
            Parser.load_weights(extractor, "/home/pjreddie/trained/yolo-coco.conv");

            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Reset();
                sw.Start();
                FloatPair p = get_rnn_vid_data(extractor, paths, N, batch, steps);

                float loss = Network.train_network_datum(net, p.X, p.Y) / (net.Batch);

                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                sw.Stop();
                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds);
                if (i % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (i % 10 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }


        public static Image save_reconstruction(Network net, Image init, float[] feat, string name, int i)
        {
            Image recon;
            if (init != null)
            {
                recon = new Image(init);
            }
            else
            {
                recon = LoadArgs.make_random_image(net.W, net.H, 3);
            }

            Image update = new Image(net.W, net.H, 3);
            reconstruct_picture(net, feat, recon, update, .01f, .9f, .1f, 2, 50);

            string buff = $"{name}{i}";
            LoadArgs.save_image(recon, buff);
            return recon;
        }

        public static void generate_vid_rnn(string cfgfile, string weightfile)
        {
            Network extractor = Parser.parse_network_cfg("cfg/extractor.recon.cfg");
            Parser.load_weights(extractor, "/home/pjreddie/trained/yolo-coco.conv");

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(extractor, 1);
            Network.set_batch_network(net, 1);

            int i;
            VideoCapture cap = new VideoCapture("/extra/vid/ILSVRC2015/Data.Data/VID/snippets/val/ILSVRC2015_val_00007030.mp4");
            float[] feat;
            float[] next;
            next = null;
            Image last = null;
            for (i = 0; i < 25; ++i)
            {
                Image im = LoadArgs.get_image_from_stream(cap);
                Image re = LoadArgs.resize_image(im, extractor.W, extractor.H);
                feat = Network.network_predict(extractor, re.Data);
                if (i > 0)
                {
                    Console.Write($"%f %f\n", Utils.mean_array(feat, 14 * 14 * 512), Utils.variance_array(feat, 14 * 14 * 512));
                    Console.Write($"%f %f\n", Utils.mean_array(next, 14 * 14 * 512), Utils.variance_array(next, 14 * 14 * 512));
                    Console.Write($"%f\n", Utils.mse_array(feat, 14 * 14 * 512));
                    Blas.Axpy_cpu(14 * 14 * 512, -1, feat, next);
                    Console.Write($"%f\n", Utils.mse_array(next, 14 * 14 * 512));
                }
                next = Network.network_predict(net, feat);
                if (i == 24) last = new Image(re);
            }
            for (i = 0; i < 30; ++i)
            {
                next = Network.network_predict(net, next);
                Image newi = save_reconstruction(extractor, last, next, "new", i);
                last = newi;
            }
        }

        public static void run_vid_rnn(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_vid_rnn(cfg, weights);
            else if (args[2] == "generate") generate_vid_rnn(cfg, weights);
        }

        #endregion

        #region CocoFile

        public static string[] coco_classes = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        public static int[] coco_ids = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };

        public static void train_coco(string cfgfile, string weightfile)
        {
            //char *train_images = "/home/pjreddie/Data.Data/voc/test/train.txt";
            //char *train_images = "/home/pjreddie/Data.Data/coco/train.txt";
            string train_images = "Data.Data/coco.trainval.txt";
            //char *train_images = "Data.Data/bags.train.list";
            string backup_directory = "/home/pjreddie/backup/";

            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data buffer = new Data.Data();


            Layer l = net.Layers[net.N - 1];

            int side = l.Side;
            int classes = l.Classes;
            float jitter = l.Jitter;

            string[] paths = Data.Data.GetPaths(train_images);

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.N = imgs;
            args.M = paths.Length;
            args.Classes = classes;
            args.Jitter = jitter;
            args.NumBoxes = side;
            args.D = buffer;
            args.Type = DataType.RegionData;

            args.Angle = net.Angle;
            args.Exposure = net.Exposure;
            args.Saturation = net.Saturation;
            args.Hue = net.Hue;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Reset();
                sw.Start();
                load_thread.Join();
                var train = buffer;
                load_thread = Data.Data.load_data_in_thread(args);

                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Stop();
                float loss = Network.train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;

                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {

                    string buff = $"{backup_directory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (i % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }

        public static void print_cocos(FileStream fp, int image_id, Box[] boxes, float[][] probs, int num_boxes, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < num_boxes; ++i)
            {
                float xmin = boxes[i].X - boxes[i].W / 2;
                float xmax = boxes[i].X + boxes[i].W / 2;
                float ymin = boxes[i].Y - boxes[i].H / 2;
                float ymax = boxes[i].Y + boxes[i].H / 2;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                float bx = xmin;
                float by = ymin;
                float bw = xmax - xmin;
                float bh = ymax - ymin;

                for (j = 0; j < classes; ++j)
                {
                    if (probs[i].Length > j)
                    {
                        var temp = Encoding.UTF8.GetBytes(
                            $"{{\"image_id\":{image_id}, \"category_id\":{coco_ids[j]}, \"bbox\":[{bx}, {by}, {bw}, {bh}], \"score\":{probs[i][j]}}},\n");
                        fp.Write(temp, 0, temp.Length);
                    }
                }
            }
        }

        public static int get_coco_image_id(string filename)
        {
            var parts = filename.Split('_');
            return int.Parse(parts[parts.Length - 1]);
        }

        public static void validate_coco(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            Network.set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum,
                net.Decay);


            string basec = "results/";
            string[] paths = Data.Data.GetPaths("Data.Data/coco_val_5k.list");

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            int side = l.Side;

            int j;
            string buff = $"{basec}/coco_results.json";
            using (var fp = File.OpenWrite(buff))
            {
                var temp = Encoding.UTF8.GetBytes("[\n");
                fp.Write(temp, 0, temp.Length);

                Box[] boxes = new Box[side * side * l.N];
                float[][] probs = new float[side * side * l.N][];
                for (j = 0; j < side * side * l.N; ++j) probs[j] = new float[classes];

                int m = paths.Length;
                int i = 0;
                int t;

                float thresh = .01f;
                bool nms = true;
                float iou_thresh = .5f;

                int nthreads = 8;
                var val = new Image[nthreads];
                var val_resized = new Image[nthreads];
                var buf = new Image[nthreads];
                var buf_resized = new Image[nthreads];
                var thr = new Thread[nthreads];

                LoadArgs args = new LoadArgs();
                args.W = net.W;
                args.H = net.H;
                args.Type = DataType.ImageData;

                for (t = 0; t < nthreads; ++t)
                {
                    args.Path = paths[i + t];
                    args.Im = buf[t];
                    args.Resized = buf_resized[t];
                    thr[t] = Data.Data.load_data_in_thread(args);
                }

                var sw = new Stopwatch();
                sw.Start();
                for (i = nthreads; i < m + nthreads; i += nthreads)
                {
                    Console.Error.Write($"%d\n", i);
                    for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                    {
                        thr[t].Join();
                        val[t] = buf[t];
                        val_resized[t] = buf_resized[t];
                    }

                    for (t = 0; t < nthreads && i + t < m; ++t)
                    {
                        args.Path = paths[i + t];
                        args.Im = buf[t];
                        args.Resized = buf_resized[t];
                        thr[t] = Data.Data.load_data_in_thread(args);
                    }

                    for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                    {
                        string path = paths[i + t - nthreads];
                        int image_id = get_coco_image_id(path);
                        float[] X = val_resized[t].Data;
                        Network.network_predict(net, X);
                        int w = val[t].W;
                        int h = val[t].H;
                        l.get_detection_boxes( w, h, thresh, probs, boxes, false);
                        if (nms) Box.do_nms_sort(boxes, probs, side * side * l.N, classes, iou_thresh);
                        print_cocos(fp, image_id, boxes, probs, side * side * l.N, classes, w, h);
                    }
                }

                fp.Seek(-1, SeekOrigin.Current);
                temp = Encoding.UTF8.GetBytes("[\n]\n");
                fp.Write(temp, 0, temp.Length);
                sw.Stop();
                Console.Error.Write($"Total Detection Time: %f Seconds\n", sw.Elapsed.Seconds);
            }
        }

        public static void validate_coco_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);


            string basec = "results/comp4_det_test_";
            string[] paths = Data.Data.GetPaths("/home/pjreddie/Data.Data/voc/test/2007_test.txt");

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            int side = l.Side;

            int j, k;
            Box[] boxes = new Box[side * side * l.N];
            float[][] probs = new float[side * side * l.N][];
            for (j = 0; j < side * side * l.N; ++j) probs[j] = new float[classes];

            int m = paths.Length;
            int i = 0;

            float thresh = .001f;
            bool nms = false;
            float iou_thresh = .5f;
            float nms_thresh = .5f;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avg_iou = 0;

            for (i = 0; i < m; ++i)
            {
                string path = paths[i];
                Image orig = LoadArgs.load_image_color(path, 0, 0);
                Image sized = LoadArgs.resize_image(orig, net.W, net.H);
                string id = Utils.Basecfg(path);
                Network.network_predict(net, sized.Data);
                l.get_detection_boxes( 1, 1, thresh, probs, boxes, true);

                Utils.find_replace(path, "images", "labels", out var labelpath);
                Utils.find_replace(labelpath, "JPEGImages", "labels", out labelpath);
                Utils.find_replace(labelpath, ".jpg", ".txt", out labelpath);
                Utils.find_replace(labelpath, ".JPEG", ".txt", out labelpath);

                int num_labels = 0;
                BoxLabel[] truth = Data.Data.read_boxes(labelpath, ref num_labels);
                for (k = 0; k < side * side * l.N; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < num_labels; ++j)
                {
                    ++total;
                    Box t = new Box(truth[j].X, truth[j].Y, truth[j].W, truth[j].H );
                    float best_iou = 0;
                    for (k = 0; k < side * side * l.N; ++k)
                    {
                        float iou = Box.box_iou(boxes[k], t);
                        if (probs[k][0] > thresh && iou > best_iou)
                        {
                            best_iou = iou;
                        }
                    }
                    avg_iou += best_iou;
                    if (best_iou > iou_thresh)
                    {
                        ++correct;
                    }
                }

                Console.Error.Write(
                    $"{i:5} {correct:5} {total:5}\tRPs/Img: {proposals/(i+1):.2}\tIOU: {avg_iou * 100:.2}%\tRecall:{100.0 *correct / total:.2}%\n");
                
            }
        }

        public static void test_coco(string cfgfile, string weightfile, string filename, float thresh)
        {
            Image[][] alphabet = LoadArgs.load_alphabet();
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Layer l = net.Layers[net.N - 1];
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);
            float nms = .4f;
            var sw = new Stopwatch();

            int j;
            Box[] boxes = new Box[l.Side * l.Side * l.N];
            float[][] probs = new float[l.Side * l.Side * l.N][];
            for (j = 0; j < l.Side * l.Side * l.N; ++j) probs[j] = new float[l.Classes];
            while (true)
            {
                string input;
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Image sized = LoadArgs.resize_image(im, net.W, net.H);
                float[] X = sized.Data;
                sw.Reset();
                sw.Start();
                Network.network_predict(net, X);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                l.get_detection_boxes( 1, 1, thresh, probs, boxes, false);
                if (nms != 0) Box.do_nms_sort(boxes, probs, l.Side * l.Side * l.N, l.Classes, nms);
                LoadArgs.draw_detections(im, l.Side * l.Side * l.N, thresh, boxes, probs, coco_classes, alphabet, 80);
                LoadArgs.save_image(im, "prediction");
                LoadArgs.show_image(im, "predictions");
                CvInvoke.WaitKey();
                CvInvoke.DestroyAllWindows();
                if (!string.IsNullOrEmpty(filename)) break;
            }
        }

        public static void run_coco(List<string> args)
        {
            string prefix = Utils.find_int_arg(args, "-prefix", "");
            float thresh = Utils.find_int_arg(args, "-thresh", .2f);
            int cam_index = Utils.find_int_arg( args, "-c", 0);
            int frame_skip = Utils.find_int_arg( args, "-s", 0);

            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "test") test_coco(cfg, weights, filename, thresh);
            else if (args[2] == "train") train_coco(cfg, weights);
            else if (args[2] == "valid") validate_coco(cfg, weights);
            else if (args[2] == "recall") validate_coco_recall(cfg, weights);
            else if (args[2] == "demo") Demo.demo(cfg, weights, thresh, cam_index, filename, coco_classes, 80, frame_skip, prefix);
        }

        #endregion

        #region ClassifierFile


        float[] get_regression_values(string[] labels, int n)
        {
            float[] v = new float[n];
            int i;
            for (i = 0; i < n; ++i)
            {
                var p = labels[i].Split(' ');
                v[i] = float.Parse(p[1]);
            }
            return v;
        }

        public static void train_classifier(string datacfg, string cfgfile, string weightfile, int[] gpus, int ngpus, bool clear)
        {
            int i;

            float avg_loss = -1;
            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Console.Write($"%d\n", ngpus);
            Network[] nets = new Network[ngpus];


            int seed = Utils.Rand.Next();
            for (i = 0; i < ngpus; ++i)
            {
                nets[i] = Parser.parse_network_cfg(cfgfile);
                if (string.IsNullOrEmpty(weightfile))
                {
                    Parser.load_weights(nets[i], weightfile);
                }
                if (clear) nets[i].Seen = 0;
                nets[i].LearningRate *= ngpus;
            }

            Network net = nets[0];

            int imgs = net.Batch * net.Subdivisions * ngpus;

            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            var options = OptionList.read_data_cfg(datacfg);

            string backup_directory = OptionList.option_find_str(options, "backup", "/backup/");
            string label_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string train_list = OptionList.option_find_str(options, "train", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);

            string[] labels = Data.Data.get_labels(label_list);

            string[] paths = Data.Data.GetPaths(train_list);
            Console.Write($"%d\n", paths.Length);
            int N = paths.Length;
            var sw = new Stopwatch();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Threads = 32;
            args.Hierarchy = net.Hierarchy;

            args.Min = net.MinCrop;
            args.Max = net.MaxCrop;
            args.Angle = net.Angle;
            args.Aspect = net.Aspect;
            args.Exposure = net.Exposure;
            args.Saturation = net.Saturation;
            args.Hue = net.Hue;
            args.Size = net.W;

            args.Paths = paths;
            args.Classes = classes;
            args.N = imgs;
            args.M = N;
            args.Labels = labels;
            args.Type = DataType.ClassificationData;

            Data.Data train;
            Data.Data buffer = new Data.Data();
            Thread load_thread;
            args.D = buffer;
            load_thread = Data.Data.load_data(args);

            int epoch = (net.Seen) / N;
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();

                load_thread.Join();
                train = buffer;
                load_thread = Data.Data.load_data(args);

                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);
                sw.Reset();
                sw.Start();

                float loss = 0;
                if (ngpus == 1)
                {
                    loss = Network.train_network(net, train);
                }
                else
                {
                    loss = Network.train_networks(nets, ngpus, train, 4);
                }
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, net.Seen);
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;

                    string buff = $"{backup_directory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}.Weights";
            Parser.save_weights(net, buff2);
        }


        public static void validate_classifier_crop(string datacfg, string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string label_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(label_list);

            string[] paths = Data.Data.GetPaths(valid_list);
            int m = paths.Length;

            var sw = new Stopwatch();
            float avg_acc = 0;
            float avg_topk = 0;
            int splits = m / 1000;
            int num = (i + 1) * m / splits - i * m / splits;

            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;

            args.Paths = paths;
            args.Classes = classes;
            args.N = num;
            args.M = 0;
            args.Labels = labels;
            args.D = buffer;
            args.Type = DataType.OldClassificationData;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            for (i = 1; i <= splits; ++i)
            {
                sw.Reset();
                sw.Start();

                load_thread.Join();
                var val = buffer;

                num = (i + 1) * m / splits - i * m / splits;
                string[] part = new string[paths.Length - (i * m / splits)];
                Array.Copy(paths, i * m / splits, part, 0, part.Length);
                if (i != splits)
                {
                    args.Paths = part;
                    load_thread = Data.Data.load_data_in_thread(args);
                }
                sw.Stop();
                Console.Write($"Loaded: %d images ini %lf seconds\n", val.X.Rows, sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float[] acc = Network.network_accuracies(net, val, topk);
                avg_acc += acc[0];
                avg_topk += acc[1];
                sw.Stop();
                Console.Write($"%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc / i, topk, avg_topk / i, sw.Elapsed.Seconds, val.X.Rows);
            }
        }

        public static void validate_classifier_10(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string label_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(label_list);

            string[] paths = Data.Data.GetPaths(valid_list);
            int m = paths.Length;

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = new int[topk];

            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (path.Contains(labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                int w = net.W;
                int h = net.H;
                int shift = 32;
                Image im = LoadArgs.load_image_color(paths[i], w + shift, h + shift);
                Image[] images = new Image[10];
                images[0] = LoadArgs.crop_image(im, -shift, -shift, w, h);
                images[1] = LoadArgs.crop_image(im, shift, -shift, w, h);
                images[2] = LoadArgs.crop_image(im, 0, 0, w, h);
                images[3] = LoadArgs.crop_image(im, -shift, shift, w, h);
                images[4] = LoadArgs.crop_image(im, shift, shift, w, h);
                LoadArgs.flip_image(im);
                images[5] = LoadArgs.crop_image(im, -shift, -shift, w, h);
                images[6] = LoadArgs.crop_image(im, shift, -shift, w, h);
                images[7] = LoadArgs.crop_image(im, 0, 0, w, h);
                images[8] = LoadArgs.crop_image(im, -shift, shift, w, h);
                images[9] = LoadArgs.crop_image(im, shift, shift, w, h);
                float[] pred = new float[classes];
                for (j = 0; j < 10; ++j)
                {
                    float[] p = Network.network_predict(net, images[j].Data);
                    if (net.Hierarchy != null) net.Hierarchy.Hierarchy_predictions(p, 0, net.Outputs, true);
                    Blas.Axpy_cpu(classes, 1, p, pred);
                }
                Utils.top_k(pred, classes, topk, indexes);
                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        public static void validate_classifier_full(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string label_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(label_list);

            string[] paths = Data.Data.GetPaths(valid_list);
            int m = paths.Length;

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = new int[topk];

            int size = net.W;
            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (path.Contains(labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                Image resized = LoadArgs.resize_min(im, size);
                Network.resize_network(net, resized.W, resized.H);
                float[] pred = Network.network_predict(net, resized.Data);
                if (net.Hierarchy != null) net.Hierarchy.Hierarchy_predictions(pred, 0, net.Outputs, true);
                
                Utils.top_k(pred, classes, topk, indexes);

                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }


        public static void validate_classifier_single(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);


            var options = OptionList.read_data_cfg(datacfg);

            string label_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string leaf_list = OptionList.option_find_str(options, "leaves", "");
            if (!string.IsNullOrEmpty(leaf_list)) net.Hierarchy.Change_leaves( leaf_list);
            string valid_list = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(label_list);

            string[] paths = Data.Data.GetPaths(valid_list);
            int m = paths.Length;

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = new int[topk];

            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (path.Contains( labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                Image resized = LoadArgs.resize_min(im, net.W);
                Image crop = LoadArgs.crop_image(resized, (resized.W - net.W) / 2, (resized.H - net.H) / 2, net.W, net.H);
                float[] pred = Network.network_predict(net, crop.Data);
                if (net.Hierarchy != null) net.Hierarchy.Hierarchy_predictions(pred, 0, net.Outputs, false);
                
                Utils.top_k(pred, classes, topk, indexes);

                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        public static void validate_classifier_multi(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string label_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(label_list);
            int[] scales = { 224, 288, 320, 352, 384 };
            int nscales = scales.Length;

            string[] paths = Data.Data.GetPaths(valid_list);
            int m = paths.Length;

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = new int[topk];

            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (path.Contains( labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                float[] pred = new float[classes];
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                for (j = 0; j < nscales; ++j)
                {
                    Image r = LoadArgs.resize_min(im, scales[j]);
                    Network.resize_network(net, r.W, r.H);
                    float[] p = Network.network_predict(net, r.Data);
                    if (net.Hierarchy != null) net.Hierarchy.Hierarchy_predictions(p, 0, net.Outputs, true);
                    Blas.Axpy_cpu(classes, 1, p, pred);
                    LoadArgs.flip_image(r);
                    p = Network.network_predict(net, r.Data);
                    Blas.Axpy_cpu(classes, 1, p, pred);
                }
                Utils.top_k(pred, classes, topk, indexes);
                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        public static void try_classifier(string datacfg, string cfgfile, string weightfile, string filename, int layer_num)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);

            var options = OptionList.read_data_cfg(datacfg);

            string name_list = OptionList.option_find_str(options, "names", "");
            if (string.IsNullOrEmpty(name_list)) name_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            int top = OptionList.option_find_int(options, "top", 1);

            int i = 0;
            string[] names = Data.Data.get_labels(name_list);
            var sw = new Stopwatch();
            int[] indexes = new int[top];

            string input = "";
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image orig = LoadArgs.load_image_color(input, 0, 0);
                Image r = LoadArgs.resize_min(orig, 256);
                Image im = LoadArgs.crop_image(r, (r.W - 224 - 1) / 2 + 1, (r.H - 224 - 1) / 2 + 1, 224, 224);
                float[] mean = { 0.48263312050943f, 0.45230225481413f, 0.40099074308742f };
                float[] std = { 0.22590347483426f, 0.22120921437787f, 0.22103996251583f };
                float[] var = new float[3];
                var[0] = std[0] * std[0];
                var[1] = std[1] * std[1];
                var[2] = std[2] * std[2];

                Blas.Normalize_cpu(im.Data, mean, var, 1, 3, im.W * im.H);

                float[] X = im.Data;
                sw.Reset();
                sw.Start();
                float[] predictions = Network.network_predict(net, X);

                Layer l = net.Layers[layer_num];
                for (i = 0; i < l.C; ++i)
                {
                    if (l.RollingMean.Length > i) Console.Write($"%f %f %f\n", l.RollingMean[i], l.RollingVariance[i], l.Scales[i]);
                }
                Array.Copy(l.OutputGpu, l.Output, l.Outputs);
                for (i = 0; i < l.Outputs; ++i)
                {
                    Console.Write($"%f\n", l.Output[i]);
                }

                Network.top_predictions(net, top, indexes);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%s: %f\n", names[index], predictions[index]);
                }
                if (!string.IsNullOrEmpty(filename)) break;
            }
        }

        public static void predict_classifier(string datacfg, string cfgfile, string weightfile, string filename, int top)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);

            var options = OptionList.read_data_cfg(datacfg);

            string name_list = OptionList.option_find_str(options, "names", "");
            if (string.IsNullOrEmpty(name_list)) name_list = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            if (top == 0) top = OptionList.option_find_int(options, "top", 1);

            int i = 0;
            string[] names = Data.Data.get_labels(name_list);
            var sw = new Stopwatch();
            int[] indexes = new int[top];

            string input = "";
            int size = net.W;
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Image r = LoadArgs.resize_min(im, size);
                Network.resize_network(net, r.W, r.H);
                Console.Write($"%d %d\n", r.W, r.H);

                float[] X = r.Data;
                sw.Reset();
                sw.Start();
                float[] predictions = Network.network_predict(net, X);
                if (net.Hierarchy != null) net.Hierarchy.Hierarchy_predictions(predictions, 0, net.Outputs, false);
                Utils.top_k(predictions, net.Outputs, top, indexes);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    if (net.Hierarchy != null)
                    {
                        Console.Write($"%d, %s: %f, parent: %s \n", index, names[index], predictions[index], (net.Hierarchy.Parent[index] >= 0) ? names[net.Hierarchy.Parent[index]] : "Root");
                    }
                    else Console.Write($"%s: %f\n", names[index], predictions[index]);
                }
                if (!string.IsNullOrEmpty(filename)) break;
            }
        }


        public static void label_classifier(string datacfg, string filename, string weightfile)
        {
            int i;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string label_list = OptionList.option_find_str(options, "names", "Data.Data/labels.list");
            string test_list = OptionList.option_find_str(options, "test", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);

            string[] labels = Data.Data.get_labels(label_list);

            string[] paths = Data.Data.GetPaths(test_list);
            int m = paths.Length;

            for (i = 0; i < m; ++i)
            {
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                Image resized = LoadArgs.resize_min(im, net.W);
                Image crop = LoadArgs.crop_image(resized, (resized.W - net.W) / 2, (resized.H - net.H) / 2, net.W, net.H);
                float[] pred = Network.network_predict(net, crop.Data);

                int ind = Utils.max_index(pred, classes);

                Console.Write($"%s\n", labels[ind]);
            }
        }


        public static void test_classifier(string datacfg, string cfgfile, string weightfile, int target_layer)
        {
            int curr = 0;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string test_list = OptionList.option_find_str(options, "test", "Data.Data/test.list");
            int classes = OptionList.option_find_int(options, "classes", 2);

            string[] paths = Data.Data.GetPaths(test_list);
            int m = paths.Length;

            var sw = new Stopwatch();

            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.Classes = classes;
            args.N = net.Batch;
            args.M = 0;
            args.Labels = new string[0];
            args.D = buffer;
            args.Type = DataType.OldClassificationData;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            for (curr = net.Batch; curr < m; curr += net.Batch)
            {
                sw.Reset();
                sw.Start();

                load_thread.Join();
                var val = buffer;

                if (curr < m)
                {
                    args.Paths = new string[paths.Length - curr];
                    Array.Copy(paths, curr, args.Paths, 0, args.Paths.Length);
                    if (curr + net.Batch > m) args.N = m - curr;
                    load_thread = Data.Data.load_data_in_thread(args);
                }
                sw.Stop();
                Console.Error.Write($"Loaded: %d images ini %lf seconds\n", val.X.Rows, sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                Matrix pred = Network.network_predict_data(net, val);

                int i, j;

                for (i = 0; i < pred.Rows; ++i)
                {
                    Console.Write($"%s", paths[curr - net.Batch + i]);
                    for (j = 0; j < pred.Cols; ++j)
                    {
                        Console.Write($"\t%g", pred.Vals[i][j]);
                    }
                    Console.Write($"\n");
                }

                sw.Stop();
                Console.Error.Write($"%lf seconds, %d images, %d total\n", sw.Elapsed.Seconds, val.X.Rows, curr);
            }
        }


        public static void threat_classifier(string datacfg, string cfgfile, string weightfile, int cam_index, string filename)
        {
            float threat = 0;
            float roll = .2f;

            Console.Write($"Classifier Demo\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            var options = OptionList.read_data_cfg(datacfg);

            Utils.Rand = new Random(2222222);
            using (VideoCapture cap = !string.IsNullOrEmpty(filename)
                ? new VideoCapture(filename)
                : new VideoCapture(cam_index))
            {

                int top = OptionList.option_find_int(options, "top", 1);

                string name_list = OptionList.option_find_str(options, "names", "");
                string[] names = Data.Data.get_labels(name_list);

                int[] indexes = new int[top];

                if (!cap.IsOpened) Utils.Error("Couldn't connect to webcam.\n");
                float fps = 0;
                int i;

                int count = 0;

                while (true)
                {
                    ++count;
                    var sw = new Stopwatch();
                    sw.Start();

                    Image ini = LoadArgs.get_image_from_stream(cap);
                    if (ini.Data.Length == 0) break;
                    Image in_s = LoadArgs.resize_image(ini, net.W, net.H);

                    Image outo = ini;
                    int x1 = outo.W / 20;
                    int y1 = outo.H / 20;
                    int x2 = 2 * x1;
                    int y2 = outo.H - outo.H / 20;

                    int border = (int) (.01f * outo.H);
                    int h = y2 - y1 - 2 * border;
                    int w = x2 - x1 - 2 * border;

                    float[] predictions = Network.network_predict(net, in_s.Data);
                    float curr_threat = 0;
                    curr_threat = predictions[0] * 0f +
                                  predictions[1] * .6f +
                                  predictions[2];
                    threat = roll * curr_threat + (1 - roll) * threat;

                    LoadArgs.draw_box_width(outo, x2 + border, (int)(y1 + .02 * h), (int)(x2 + .5 * w), (int)(y1 + .02 * h + border), border, 0, 0,
                        0);
                    if (threat > .97)
                    {
                        LoadArgs.draw_box_width(outo, (int)(x2 + .5 * w + border),
                            (int)(y1 + .02 * h - 2 * border),
                            (int)(x2 + .5 * w + 6 * border),
                            (int)(y1 + .02 * h + 3 * border), 3 * border, 1, 0, 0);
                    }

                    LoadArgs.draw_box_width(outo, (int)(x2 + .5 * w + border),
                        (int)(y1 + .02 * h - 2 * border),
                            (int)(x2 + .5 * w + 6 * border),
                                (int)(y1 + .02 * h + 3 * border), (int)(.5 * border), 0, 0, 0);
                    LoadArgs.draw_box_width(outo, x2 + border, (int)(y1 + .42 * h), (int)(x2 + .5 * w), (int)(y1 + .42 * h + border), border, 0, 0,
                        0);
                    if (threat > .57)
                    {
                        LoadArgs.draw_box_width(outo, (int)(x2 + .5 * w + border),
                            (int)(y1 + .42 * h - 2 * border),
                            (int)(x2 + .5 * w + 6 * border),
                            (int)(y1 + .42 * h + 3 * border), (int)(3 * border), 1, 1, 0);
                    }

                    LoadArgs.draw_box_width(outo, (int)(x2 + .5 * w + border),
                        (int)(y1 + .42 * h - 2 * border),
                            (int)(x2 + .5 * w + 6 * border),
                                (int)(y1 + .42 * h + 3 * border), (int)(.5 * border), 0, 0, 0);

                    LoadArgs.draw_box_width(outo, x1, y1, x2, y2, border, 0, 0, 0);
                    for (i = 0; i < threat * h; ++i)
                    {
                        float ratio = (float) i / h;
                        float r = (ratio < .5f) ? (2 * (ratio)) : 1;
                        float g = (ratio < .5f) ? 1 : 1 - 2 * (ratio - .5f);
                        LoadArgs.draw_box_width(outo, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
                    }

                    Network.top_predictions(net, top, indexes);

                    string buff = $"/home/pjreddie/tmp/threat_{count:06}";

                    Console.Write($"\033[2J");
                    Console.Write($"\033[1;1H");
                    Console.Write($"\nFPS:%.0f\n", fps);

                    for (i = 0; i < top; ++i)
                    {
                        int index = indexes[i];
                        Console.Write($"%.1f%%: %s\n", predictions[index] * 100, names[index]);
                    }

                    LoadArgs.show_image(outo, "Threat");
                    CvInvoke.WaitKey(10);

                    sw.Stop();
                    float curr = 1000.0f / sw.ElapsedMilliseconds;
                    fps = .9f * fps + .1f * curr;
                }
            }
        }


        public static void gun_classifier(string datacfg, string cfgfile, string weightfile, int cam_index, string filename)
        {
            int[] bad_cats = { 218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697 };

            Console.Write($"Classifier Demo\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            var options = OptionList.read_data_cfg(datacfg);

            Utils.Rand = new Random(2222222);
            using (VideoCapture cap = !string.IsNullOrEmpty(filename)
                ?  new VideoCapture(filename)
                : new VideoCapture(cam_index))
            {

                int top = OptionList.option_find_int(options, "top", 1);

                string name_list = OptionList.option_find_str(options, "names", "");
                string[] names = Data.Data.get_labels(name_list);

                int[] indexes = new int[top];

                if (cap.IsOpened) Utils.Error("Couldn't connect to webcam.\n");

                float fps = 0;
                int i;

                while (true)
                {
                    var sw = new Stopwatch();
                    sw.Start();

                    Image ini = LoadArgs.get_image_from_stream(cap);
                    Image in_s = LoadArgs.resize_image(ini, net.W, net.H);
                    LoadArgs.show_image(ini, "Threat Detection");

                    float[] predictions = Network.network_predict(net, in_s.Data);
                    Network.top_predictions(net, top, indexes);

                    Console.Write($"\033[2J");
                    Console.Write($"\033[1;1H");

                    bool threat = false;
                    for (i = 0; i < bad_cats.Length; ++i)
                    {
                        int index = bad_cats[i];
                        if (predictions[index] > .01)
                        {
                            Console.Write($"Threat Detected!\n");
                            threat = true;
                            break;
                        }
                    }

                    if (threat) Console.Write($"Scanning...\n");
                    for (i = 0; i < bad_cats.Length; ++i)
                    {
                        int index = bad_cats[i];
                        if (predictions[index] > .01)
                        {
                            Console.Write($"%s\n", names[index]);
                        }
                    }

                    CvInvoke.WaitKey(10);

                    sw.Stop();
                    float curr = 1000.0f / sw.ElapsedMilliseconds;
                    fps = .9f * fps + .1f * curr;
                }
            }
        }

        public static void demo_classifier(string datacfg, string cfgfile, string weightfile, int cam_index, string filename)
        {
            Console.Write($"Classifier Demo\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            var options = OptionList.read_data_cfg(datacfg);

            Utils.Rand = new Random(2222222);
            using (VideoCapture cap = !string.IsNullOrEmpty(filename)
                ? new VideoCapture(filename)
                : new VideoCapture(cam_index))
            {
                int top = OptionList.option_find_int(options, "top", 1);

                string name_list = OptionList.option_find_str(options, "names", "");
                string[] names = Data.Data.get_labels(name_list);

                int[] indexes = new int[top];

                if (cap != null) Utils.Error("Couldn't connect to webcam.\n");

                float fps = 0;
                int i;

                while (true)
                {
                    var sw = new Stopwatch();
                    sw.Start();

                    Image ini = LoadArgs.get_image_from_stream(cap);
                    Image in_s = LoadArgs.resize_image(ini, net.W, net.H);
                    LoadArgs.show_image(ini, "Classifier");

                    float[] predictions = Network.network_predict(net, in_s.Data);
                    if (net.Hierarchy != null) net.Hierarchy.Hierarchy_predictions(predictions, 0, net.Outputs, true);
                    Network.top_predictions(net, top, indexes);

                    Console.Write($"\033[2J");
                    Console.Write($"\033[1;1H");
                    Console.Write($"\nFPS:%.0f\n", fps);

                    for (i = 0; i < top; ++i)
                    {
                        int index = indexes[i];
                        Console.Write($"%.1f%%: %s\n", predictions[index] * 100, names[index]);
                    }

                    CvInvoke.WaitKey(10);

                    sw.Stop();
                    float curr = 1000.0f / sw.ElapsedMilliseconds;
                    fps = .9f * fps + .1f * curr;
                }
            }
        }


        public static void run_classifier(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            int cam_index = Utils.find_int_arg(args, "-c", 0);
            int top = Utils.find_int_arg(args, "-t", 0);
            bool clear = Utils.find_arg(args, "-clear");
            string data = args[3];
            string cfg = args[4];
            string weights = (args.Count > 5) ? args[5] : "";
            string filename = (args.Count > 6) ? args[6] : "";
            string layer_s = (args.Count > 7) ? args[7] : "";
            int Layer = !string.IsNullOrEmpty(layer_s) ? int.Parse(layer_s) : -1;
            if (args[2] == "predict") predict_classifier(data, cfg, weights, filename, top);
            else if (args[2] == "try") try_classifier(data, cfg, weights, filename, int.Parse(layer_s));
            else if (args[2] == "train") train_classifier(data, cfg, weights, new int[0], 1, clear);
            else if (args[2] == "demo") demo_classifier(data, cfg, weights, cam_index, filename);
            else if (args[2] == "gun") gun_classifier(data, cfg, weights, cam_index, filename);
            else if (args[2] == "threat") threat_classifier(data, cfg, weights, cam_index, filename);
            else if (args[2] == "test") test_classifier(data, cfg, weights, Layer);
            else if (args[2] == "label") label_classifier(data, cfg, weights);
            else if (args[2] == "valid") validate_classifier_single(data, cfg, weights);
            else if (args[2] == "validmulti") validate_classifier_multi(data, cfg, weights);
            else if (args[2] == "valid10") validate_classifier_10(data, cfg, weights);
            else if (args[2] == "validcrop") validate_classifier_crop(data, cfg, weights);
            else if (args[2] == "validfull") validate_classifier_full(data, cfg, weights);
        }


        #endregion

        #region ArtFile


        public static void demo_art(string cfgfile, string weightfile, int cam_index)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);

            Utils.Rand = new Random(2222222);
            using (VideoCapture cap = new VideoCapture(cam_index))
            {
                string window = "ArtJudgementBot9000!!!";
                if (cap != null) Utils.Error("Couldn't connect to webcam.\n");
                int i;
                int[] idx = {37, 401, 434};
                int n = idx.Length;

                while (true)
                {
                    Image ini = LoadArgs.get_image_from_stream(cap);
                    Image in_s = LoadArgs.resize_image(ini, net.W, net.H);
                    LoadArgs.show_image(ini, window);

                    float[] p = Network.network_predict(net, in_s.Data);

                    Console.Write($"\033[2J");
                    Console.Write($"\033[1;1H");

                    float score = 0;
                    for (i = 0; i < n; ++i)
                    {
                        float s = p[idx[i]];
                        if (s > score) score = s;
                    }

                    Console.Write($"I APPRECIATE THIS ARTWORK: %10.7f%%\n", score * 100);
                    Console.Write($"[");
                    int upper = 30;
                    for (i = 0; i < upper; ++i)
                    {
                        Console.Write($"%c", ((i + .5) < score * upper) ? 219 : ' ');
                    }

                    Console.Write($"]\n");

                    CvInvoke.WaitKey(1);
                }
            }
        }


        public static void run_art(List<string> args)
        {
            int cam_index = Utils.find_int_arg(args, "-c", 0);
            string cfg = args[2];
            string weights = args[3];
            demo_art(cfg, weights, cam_index);
        }


        #endregion

        #region TagFile


        public static void train_tag(string cfgfile, string weightfile, bool clear)
        {

            float avg_loss = -1;
            string basec = Utils.Basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            if (clear) net.Seen = 0;
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            string[] paths = Data.Data.GetPaths("/home/pjreddie/tag/train.list");
            Console.Write($"%d\n", paths.Length);
            int N = paths.Length;
            var sw = new Stopwatch();
            Thread load_thread;
            Data.Data train;
            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;

            args.Min = net.W;
            args.Max = net.MaxCrop;
            args.Size = net.W;

            args.Paths = paths;
            args.Classes = net.Outputs;
            args.N = imgs;
            args.M = N;
            args.D = buffer;
            args.Type = DataType.TagData;

            args.Angle = net.Angle;
            args.Exposure = net.Exposure;
            args.Saturation = net.Saturation;
            args.Hue = net.Hue;

            Console.Error.Write($"%d classes\n", net.Outputs);

            load_thread = Data.Data.load_data_in_thread(args);
            int epoch = (net.Seen) / N;
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();
                load_thread.Join();
                train = buffer;

                load_thread = Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);
                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;
                Console.Write(
                    $"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, net.Seen);

                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;

                    string buff = $"{backup_directory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backup_directory}/{basec}.Weights";
            Parser.save_weights(net, buff2);

            load_thread.Join();
        }

        public static void test_tag(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);
            int i = 0;
            string[] names = Data.Data.get_labels("Data.Data/tags.txt");
            var sw = new Stopwatch();
            int[] indexes = new int[10];

            string input = "";
            int size = net.W;
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }
                Image im = LoadArgs.load_image_color(input, 0, 0);
                Image r = LoadArgs.resize_min(im, size);
                Network.resize_network(net, r.W, r.H);
                Console.Write($"%d %d\n", r.W, r.H);

                float[] X = r.Data;

                sw.Reset();
                sw.Start();
                float[] predictions = Network.network_predict(net, X);
                Network.top_predictions(net, 10, indexes);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                for (i = 0; i < 10; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }
                if (!string.IsNullOrEmpty(filename)) break;
            }
        }


        public static void run_tag(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            bool clear = Utils.find_arg(args, "-clear");
            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "train") train_tag(cfg, weights, clear);
            else if (args[2] == "test") test_tag(cfg, weights, filename);
        }


        #endregion

        #region CompareFile


        public static void train_compare(string cfgfile, string weightfile)
        {

            float avg_loss = -1;
            string basec = Utils.Basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            string[] paths = Data.Data.GetPaths("Data.Data/compare.train.list");
            int N = paths.Length;
            Console.Write($"%d\n", N);
            var sw = new Stopwatch();
            Thread load_thread;
            Data.Data train;
            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.Classes = 20;
            args.N = imgs;
            args.M = N;
            args.D = buffer;
            args.Type = DataType.CompareData;

            load_thread = Data.Data.load_data_in_thread(args);
            int epoch = net.Seen / N;
            int i = 0;
            while (true)
            {
                ++i;
                sw.Reset();
                sw.Start();
                load_thread.Join();
                train = buffer;

                load_thread = Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;
                sw.Stop();
                Console.Write($"%.3f: %f, %f avg, %lf seconds, %d images\n", (float)net.Seen / N, loss, avg_loss, sw.Elapsed.Seconds, net.Seen);
                if (i % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}_{epoch}_minor_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    i = 0;

                    string buff = $"{backup_directory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                    if (epoch % 22 == 0) net.LearningRate *= .1f;
                }
            }
            load_thread.Join();
        }

        public static void validate_compare(string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            string[] paths = Data.Data.GetPaths("Data.Data/compare.val.list");
            int N = paths.Length / 2;

            var sw = new Stopwatch();
            int correct = 0;
            int total = 0;
            int splits = 10;
            int num = (i + 1) * N / splits - i * N / splits;

            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.Classes = 20;
            args.N = num;
            args.M = 0;
            args.D = buffer;
            args.Type = DataType.CompareData;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            for (i = 1; i <= splits; ++i)
            {
                sw.Reset();
                sw.Start();

                load_thread.Join();
                var val = buffer;

                num = (i + 1) * N / splits - i * N / splits;
                string[] part = new string[paths.Length - (i * N / splits)];
                Array.Copy(paths, i * N / splits, part, 0, part.Length);
                if (i != splits)
                {
                    args.Paths = part;
                    load_thread = Data.Data.load_data_in_thread(args);
                }
                sw.Stop();
                Console.Write($"Loaded: %d images ini %lf seconds\n", val.X.Rows, sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                Matrix pred = Network.network_predict_data(net, val);
                int j, k;
                for (j = 0; j < val.Y.Rows; ++j)
                {
                    for (k = 0; k < 20; ++k)
                    {
                        if (val.Y.Vals[j][k * 2] != val.Y.Vals[j][k * 2 + 1])
                        {
                            ++total;
                            if ((val.Y.Vals[j][k * 2] < val.Y.Vals[j][k * 2 + 1]) == (pred.Vals[j][k * 2] < pred.Vals[j][k * 2 + 1]))
                            {
                                ++correct;
                            }
                        }
                    }
                }
                sw.Stop();
                Console.Write($"%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct / total, sw.Elapsed.Seconds, val.X.Rows);
            }
        }


        public static int total_compares = 0;
        public static int current_class = 0;

        public static int elo_comparator(SortableBbox a, SortableBbox b)
        {
            if (a.elos[current_class] == b.elos[current_class]) return 0;
            if (a.elos[current_class] > b.elos[current_class]) return -1;
            return 1;
        }

        public static int bbox_comparator(SortableBbox a, SortableBbox b)
        {
            ++total_compares;
            Network net = a.net;
            int sclass = a.sclass;

            Image im1 = LoadArgs.load_image_color(a.filename, net.W, net.H);
            Image im2 = LoadArgs.load_image_color(b.filename, net.W, net.H);
            float[] X = new float[net.W * net.H * net.C];
            Array.Copy(im1.Data, 0, X, 0, im1.Data.Length);
            Array.Copy(im2.Data, 0, X, im1.Data.Length, im2.Data.Length);
            float[] predictions = Network.network_predict(net, X);

            if (predictions[sclass * 2] > predictions[sclass * 2 + 1])
            {
                return 1;
            }
            return -1;
        }

        public static void bbox_update(SortableBbox a, SortableBbox b, int sclass, bool result)
        {
            int k = 32;
            float EA = 1.0f / (1 + (float)Math.Pow(10, (b.elos[sclass] - a.elos[sclass]) / 400f));
            float EB = 1.0f / (1 + (float)Math.Pow(10, (a.elos[sclass] - b.elos[sclass]) / 400f));
            float SA = result ? 1 : 0;
            float SB = result ? 0 : 1;
            a.elos[sclass] += k * (SA - EA);
            b.elos[sclass] += k * (SB - EB);
        }

        public static void bbox_fight(Network net, SortableBbox a, SortableBbox b, int classes, int sclass)
        {
            Image im1 = LoadArgs.load_image_color(a.filename, net.W, net.H);
            Image im2 = LoadArgs.load_image_color(b.filename, net.W, net.H);
            float[] X = new float[net.W * net.H * net.C];
            Array.Copy(im1.Data, 0, X, 0, im1.Data.Length);
            Array.Copy(im2.Data, 0, X, im1.Data.Length, im2.Data.Length);
            float[] predictions = Network.network_predict(net, X);
            ++total_compares;

            int i;
            for (i = 0; i < classes; ++i)
            {
                if (sclass < 0 || sclass == i)
                {
                    bool result = predictions[i * 2] > predictions[i * 2 + 1];
                    bbox_update(a, b, i, result);
                }
            }
        }

        public static void SortMaster3000(string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            Network.set_batch_network(net, 1);

            string[] paths = Data.Data.GetPaths("Data.Data/compare.sort.list");
            int N = paths.Length;
            SortableBbox[] boxes = new SortableBbox[N];
            Console.Write($"Sorting %d boxes...\n", N);
            for (i = 0; i < N; ++i)
            {
                boxes[i].filename = paths[i];
                boxes[i].net = net;
                boxes[i].sclass = 7;
                boxes[i].elo = 1500;
            }
            var sw = new Stopwatch();
            sw.Start();
            Array.Sort(boxes, bbox_comparator);
            for (i = 0; i < N; ++i)
            {
                Console.Write($"%s\n", boxes[i].filename);
            }
            sw.Stop();
            Console.Write($"Sorted ini %d compares, %f secs\n", total_compares, sw.Elapsed.Seconds);
        }

        public static void BattleRoyaleWithCheese(string filename, string weightfile)
        {
            int classes = 20;
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            Network.set_batch_network(net, 1);

            string[] paths = Data.Data.GetPaths("Data.Data/compare.sort.list");
            int N = paths.Length;
            int total = N;
            SortableBbox[] boxes = new SortableBbox[N];
            Console.Write($"Battling %d boxes...\n", N);
            for (i = 0; i < N; ++i)
            {
                boxes[i].filename = paths[i];
                boxes[i].net = net;
                boxes[i].classes = classes;
                boxes[i].elos = new float[classes];
                for (j = 0; j < classes; ++j)
                {
                    boxes[i].elos[j] = 1500;
                }
            }
            int round;
            var swTotal = new Stopwatch();
            swTotal.Start();
            for (round = 1; round <= 4; ++round)
            {
                var sw = new Stopwatch();
                sw.Start();
                Console.Write($"Round: %d\n", round);
                Utils.Shuffle(boxes);
                for (i = 0; i < N / 2; ++i)
                {
                    bbox_fight(net, boxes[i * 2], boxes[i * 2 + 1], classes, -1);
                }
                sw.Stop();
                Console.Write($"Round: %f secs, %d remaining\n", sw.Elapsed.Seconds, N);
            }

            for (var iClass = 0; iClass < classes; ++iClass)
            {

                N = total;
                current_class = iClass;
                Array.Sort(boxes, elo_comparator);
                N /= 2;

                for (round = 1; round <= 100; ++round)
                {
                    var sw = new Stopwatch();
                    sw.Start();
                    Console.Write($"Round: %d\n", round);

                    Utils.sorta_shuffle(boxes, 10);
                    for (i = 0; i < N / 2; ++i)
                    {
                        bbox_fight(net, boxes[ i * 2], boxes[ i * 2 + 1], classes, iClass);
                    }
                    Array.Sort(boxes, elo_comparator);
                    if (round <= 20) N = (N * 9 / 10) / 2 * 2;

                    sw.Stop();
                    Console.Write($"Round: %f secs, %d remaining\n", sw.Elapsed.Seconds, N);
                }

                string buff = $"results/battle_{iClass}.(float)Math.Log";
                var lines = new List<string>();
                for (i = 0; i < N; ++i)
                {
                    lines.Add($"{boxes[i].filename} {boxes[i].elos[iClass]}");
                }
                File.WriteAllLines(buff, lines);
            }
            swTotal.Stop();
            Console.Write($"Tournament ini %d compares, %f secs\n", total_compares, swTotal.Elapsed.Seconds);
        }

        public static void run_compare(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_compare(cfg, weights);
            else if (args[2] == "valid") validate_compare(cfg, weights);
            else if (args[2] == "sort") SortMaster3000(cfg, weights);
            else if (args[2] == "battle") BattleRoyaleWithCheese(cfg, weights);
        }


        #endregion

        #region WritingFile


        public static void train_writing(string cfgfile, string weightfile)
        {
            string backup_directory = "/home/pjreddie/backup/";

            float avg_loss = -1;
            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            string[] paths = Data.Data.GetPaths("figures.list");
            var sw = new Stopwatch();
            int N = paths.Length;
            Console.Write($"N: %d\n", N);
            Image outf = Network.get_network_image(net);

            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.OutW = outf.W;
            args.OutH = outf.H;
            args.Paths = paths;
            args.N = imgs;
            args.M = N;
            args.D = buffer;
            args.Type = DataType.WritingData;

            Thread load_thread = Data.Data.load_data_in_thread(args);
            int epoch = (net.Seen) / N;
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();
                load_thread.Join();
                var train = buffer;
                load_thread = Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);


                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9f + loss * .1f;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, Network.get_current_rate(net), sw.Elapsed.Seconds, net.Seen);
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backup_directory}/{basec}_batch_{Data.Network.get_current_batch(net)}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;

                    string buff = $"{backup_directory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
            }
        }

        public static void test_writing(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);
            var sw = new Stopwatch();

            string input = "";
            while (true)
            {
                if (!string.IsNullOrEmpty(filename))
                {
                    input = filename;
                }
                else
                {
                    Console.Write($"Enter Image Path: ");

                    input = Console.ReadLine();
                    if (string.IsNullOrEmpty(input)) return;
                    input = input.TrimEnd();
                }

                Image im = LoadArgs.load_image_color(input, 0, 0);
                Network.resize_network(net, im.W, im.H);
                Console.Write($"%d %d %d\n", im.H, im.W, im.C);
                float[] X = im.Data;
                sw.Reset();
                sw.Start();
                Network.network_predict(net, X);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                Image pred = Network.get_network_image(net);

                Image upsampled = LoadArgs.resize_image(pred, im.W, im.H);
                Image thresh = LoadArgs.threshold_image(upsampled, .5f);
                pred = thresh;

                LoadArgs.show_image(pred, "prediction");
                LoadArgs.show_image(im, "orig");
                CvInvoke.WaitKey();
                CvInvoke.DestroyAllWindows();

                if (!string.IsNullOrEmpty(filename)) break;
            }
        }

        public static void run_writing(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "train") train_writing(cfg, weights);
            else if (args[2] == "test") test_writing(cfg, weights, filename);
        }


        #endregion

        #region NightmareFile


        float abs_mean(float[] x, int n)
        {
            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                sum += (float)Math.Abs(x[i]);
            }
            return sum / n;
        }

        public static void calculate_loss(float[] output, float[] delta, int n, float thresh)
        {
            int i;
            float mean = Utils.mean_array(output, n);
            float var = Utils.variance_array(output, n);
            for (i = 0; i < n; ++i)
            {
                if (delta[i] > mean + thresh * (float)Math.Sqrt(var)) delta[i] = output[i];
                else delta[i] = 0;
            }
        }

        public static void optimize_picture(Network net, Image orig, int max_layer, float scale, float rate, float thresh, bool norm)
        {
            net.N = max_layer + 1;

            int dx = Utils.Rand.Next() % 16 - 8;
            int dy = Utils.Rand.Next() % 16 - 8;
            bool flip = Utils.Rand.Next() % 2 != 0;

            Image crop = LoadArgs.crop_image(orig, dx, dy, orig.W, orig.H);
            Image im = LoadArgs.resize_image(crop, (int)(orig.W * scale), (int)(orig.H * scale));
            if (flip) LoadArgs.flip_image(im);

            Network.resize_network(net, im.W, im.H);
            Layer last = net.Layers[net.N - 1];

            Image delta = new Image(im.W, im.H, im.C);

            NetworkState state = new NetworkState();
            state.Input = (float[])im.Data.Clone();
            state.Delta = (float[])im.Data.Clone();

            Network.forward_network_gpu(net, state);
            Blas.copy_ongpu(last.Outputs, last.OutputGpu, last.DeltaGpu);

            Array.Copy(last.DeltaGpu, last.Delta, last.Outputs);
            calculate_loss(last.Delta, last.Delta, last.Outputs, thresh);
            Array.Copy(last.Delta, last.DeltaGpu, last.Outputs);

            Network.backward_network_gpu(net, state);

            Array.Copy(state.Delta, delta.Data, im.W * im.H * im.C);


            if (flip) LoadArgs.flip_image(delta);

            Image resized = LoadArgs.resize_image(delta, orig.W, orig.H);
            Image outi = LoadArgs.crop_image(resized, -dx, -dy, orig.W, orig.H);

            if (norm) Utils.normalize_array(outi.Data, outi.W * outi.H * outi.C);
            Blas.Axpy_cpu(orig.W * orig.H * orig.C, rate, outi.Data, orig.Data);
            
            LoadArgs.constrain_image(orig);
        }

        public static void smooth(Image recon, Image update, float lambda, int num)
        {
            int i, j, k;
            int ii, jj;
            for (k = 0; k < recon.C; ++k)
            {
                for (j = 0; j < recon.H; ++j)
                {
                    for (i = 0; i < recon.W; ++i)
                    {
                        int out_index = i + recon.W * (j + recon.H * k);
                        for (jj = j - num; jj <= j + num && jj < recon.H; ++jj)
                        {
                            if (jj < 0) continue;
                            for (ii = i - num; ii <= i + num && ii < recon.W; ++ii)
                            {
                                if (ii < 0) continue;
                                int in_index = ii + recon.W * (jj + recon.H * k);
                                update.Data[out_index] += lambda * (recon.Data[in_index] - recon.Data[out_index]);
                            }
                        }
                    }
                }
            }
        }

        public static void reconstruct_picture(Network net, float[] features, Image recon, Image update, float rate, float momentum, float lambda, int smooth_size, int iters)
        {
            int iter = 0;
            for (iter = 0; iter < iters; ++iter)
            {
                Image delta = new Image(recon.W, recon.H, recon.C);

                NetworkState state = new NetworkState();
                state.Input = (float[])recon.Data.Clone();
                state.Delta = (float[])delta.Data.Clone();
                state.Truth = new float[Network.get_network_output_size(net)];
                Array.Copy(features, 0, state.Truth, 0, state.Truth.Length);

                Network.forward_network_gpu(net, state);
                Network.backward_network_gpu(net, state);

                Array.Copy(state.Delta, delta.Data, delta.W * delta.H * delta.C);

                Blas.Axpy_cpu(recon.W * recon.H * recon.C, 1, delta.Data, update.Data);
                smooth(recon, update, lambda, smooth_size);

                Blas.Axpy_cpu(recon.W * recon.H * recon.C, rate, update.Data, recon.Data);
                Blas.Scal_cpu(recon.W * recon.H * recon.C, momentum, update.Data, 1);

                LoadArgs.constrain_image(recon);
            }
        }


        public static void run_nightmare(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [cfg] [weights] [Image] [Layer] [options! (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[2];
            string weights = args[3];
            string input = args[4];
            int max_layer = int.Parse(args[5]);

            int range = Utils.find_int_arg(args, "-range", 1);
            bool norm = Utils.find_int_arg(args, "-norm", 1) != 0;
            int rounds = Utils.find_int_arg(args, "-rounds", 1);
            int iters = Utils.find_int_arg(args, "-iters", 10);
            int octaves = Utils.find_int_arg(args, "-octaves", 4);
            float zoom = Utils.find_int_arg(args, "-zoom", 1);
            float rate = Utils.find_int_arg(args, "-rate", .04f);
            float thresh = Utils.find_int_arg(args, "-thresh", 1);
            float rotate = Utils.find_int_arg(args, "-rotate", 0);
            float momentum = Utils.find_int_arg(args, "-momentum", .9f);
            float lambda = Utils.find_int_arg(args, "-lambda", .01f);
            string prefix = Utils.find_int_arg(args, "-prefix", "");
            bool reconstruct = Utils.find_arg(args, "-reconstruct");
            int smooth_size = Utils.find_int_arg(args, "-smooth", 1);

            Network net = Parser.parse_network_cfg(cfg);
            Parser.load_weights(net, weights);
            string cfgbase = Utils.Basecfg(cfg);
            string imbase = Utils.Basecfg(input);

            Network.set_batch_network(net, 1);
            Image im = LoadArgs.load_image_color(input, 0, 0);
            
            float[] features = new float[0];
            Image update = null;
            if (reconstruct)
            {
                Network.resize_network(net, im.W, im.H);

                int zz = 0;
                Network.network_predict(net, im.Data);
                Image out_im = Network.get_network_image(net);
                Image crop = LoadArgs.crop_image(out_im, zz, zz, out_im.W - 2 * zz, out_im.H - 2 * zz);
                Image f_im = LoadArgs.resize_image(crop, out_im.W, out_im.H);
                Console.Write($"%d features\n", out_im.W * out_im.H * out_im.C);


                im = LoadArgs.resize_image(im, im.W, im.H);
                f_im = LoadArgs.resize_image(f_im, f_im.W, f_im.H);
                features = f_im.Data;

                int i;
                for (i = 0; i < 14 * 14 * 512; ++i)
                {
                    features[i] += Utils.rand_uniform(-.19f, .19f);
                }

                im = LoadArgs.make_random_image(im.W, im.H, im.C);
                update = new Image(im.W, im.H, im.C);

            }

            int e;
            int n;
            for (e = 0; e < rounds; ++e)
            {
                Console.Error.Write($"Iteration: ");
                for (n = 0; n < iters; ++n)
                {
                    Console.Error.Write($"%d, ", n);
                    if (reconstruct)
                    {
                        reconstruct_picture(net, features, im, update, rate, momentum, lambda, smooth_size, 1);
                        //if ((n+1)%30 == 0) rate *= .5;
                        LoadArgs.show_image(im, "reconstruction");
                        CvInvoke.WaitKey(10);
                    }
                    else
                    {
                        int Layer = max_layer + Utils.Rand.Next() % range - range / 2;
                        int octave = Utils.Rand.Next() % octaves;
                        optimize_picture(net, im, Layer, 1 / (float)Math.Pow(1.33333333, octave), rate, thresh, norm);
                    }
                }
                Console.Error.Write($"done\n");
                string buff;
                if (!string.IsNullOrEmpty(prefix))
                {
                    buff = $"{prefix}_{imbase}_{cfgbase}_{max_layer}_{e:06}%s/%s_%s_%d_%06d";
                }
                else
                {
                    buff = $"{imbase}_{cfgbase}_{max_layer}_{e:06}";
                }
                Console.Write($"%d %s\n", e, buff);
                LoadArgs.save_image(im, buff);
                //LoadArgs.show_image(im, buff);
                //CvInvoke.WaitKey();

                if (rotate != 0)
                {
                    Image rot = LoadArgs.rotate_image(im, rotate);
                    im = rot;
                }
                Image crop = LoadArgs.crop_image(im, (int)(im.W * (1f - zoom) / 2f), (int)(im.H * (1f - zoom) / 2f), (int)(im.W * zoom), (int)(im.H * zoom));
                Image resized = LoadArgs.resize_image(crop, im.W, im.H);
                im = resized;
            }
        }


        #endregion
    }
}
