using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
                case "go":
                    run_go(args);
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
                case "dice":
                    run_dice(args);
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
                case "captcha":
                    run_captcha(args);
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
                        Blas.Axpy_cpu(l.N, 1, l.Biases, outl.Biases);
                        Blas.Axpy_cpu(num, 1, l.Weights, outl.Weights);
                        if (l.BatchNormalize)
                        {
                            Blas.Axpy_cpu(l.N, 1, l.Scales, outl.Scales);
                            Blas.Axpy_cpu(l.N, 1, l.RollingMean, outl.RollingMean);
                            Blas.Axpy_cpu(l.N, 1, l.RollingVariance, outl.RollingVariance);
                        }
                    }
                    if (l.LayerType == LayerType.Connected)
                    {
                        Blas.Axpy_cpu(l.Outputs, 1, l.Biases, outl.Biases);
                        Blas.Axpy_cpu(l.Outputs * l.Inputs, 1, l.Weights, outl.Weights);
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
                    Blas.Scal_cpu(l.N, 1.0f/ n, l.Biases, 1);
                    Blas.Scal_cpu(num, 1.0f/ n, l.Weights, 1);
                    if (l.BatchNormalize)
                    {
                        Blas.Scal_cpu(l.N, 1.0f / n, l.Scales, 1);
                        Blas.Scal_cpu(l.N, 1.0f / n, l.RollingMean, 1);
                        Blas.Scal_cpu(l.N, 1.0f / n, l.RollingVariance, 1);
                    }
                }
                if (l.LayerType == LayerType.Connected)
                {
                    Blas.Scal_cpu(l.Outputs, 1.0f / n, l.Biases, 1);
                    Blas.Scal_cpu(l.Outputs * l.Inputs, 1.0f / n, l.Weights, 1);
                }
            }
            save_weights(sum, outfile);
        }

        public static void speed(string cfgfile, int tics)
        {
            if (tics == 0) tics = 1000;
            Network net = Parser.parse_network_cfg(cfgfile);
            set_batch_network(net, 1);
            int i;
            var sw = new Stopwatch();
            sw.Start();
            Image im = new Image(net.W, net.H, net.C);
            for (i = 0; i < tics; ++i)
            {
                network_predict(net, im.Data.Data);
            }
            sw.Stop();
            var t = sw.Elapsed.Seconds;
            Console.Write($"\n%d evals, %f Seconds\n", tics, t);
            Console.Write($"Speed: %f sec/eval\n", t / tics);
            Console.Write($"Speed: %f Hz\n", tics / t);
        }

        public static void operations(string cfgfile)
        {
            CudaUtils.UseGpu = -1;
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
            Console.Write($"Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.);
        }

        public static void oneoff(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            int oldn = net.Layers[net.N - 2].N;
            int c = net.Layers[net.N - 2].C;
            net.Layers[net.N - 2].N = 9372;
            net.Layers[net.N - 2].Biases += 5;
            net.Layers[net.N - 2].Weights += 5 * c;
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            net.Layers[net.N - 2].Biases -= 5;
            net.Layers[net.N - 2].Weights -= 5 * c;
            net.Layers[net.N - 2].N = oldn;
            Console.Write($"%d\n", oldn);
            Layer l = net.Layers[net.N - 2];
            Blas.Copy_cpu(l.N / 3, l.Biases, 1, l.Biases + l.N / 3, 1);
            Blas.Copy_cpu(l.N / 3, l.Biases, 1, l.Biases + 2 * l.N / 3, 1);
            Blas.Copy_cpu(l.N / 3 * l.C, l.Weights, 1, l.Weights + l.N / 3 * l.C, 1);
            Blas.Copy_cpu(l.N / 3 * l.C, l.Weights, 1, l.Weights + 2 * l.N / 3 * l.C, 1);
            net.Seen = 0;
            save_weights(net, outfile);
        }

        public static void partial(string cfgfile, string weightfile, string outfile, int max)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights_upto(net, weightfile, max);
            }
            net.Seen = 0;
            save_weights_upto(net, outfile, max);
        }

        public static void rescale_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    rescale_weights(l, 2, -.5);
                    break;
                }
            }
            save_weights(net, outfile);
        }

        public static void rgbgr_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional)
                {
                    rgbgr_weights(l);
                    break;
                }
            }
            save_weights(net, outfile);
        }

        public static void reset_normalize_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.LayerType == LayerType.Convolutional && l.BatchNormalize)
                {
                    denormalize_convolutional_layer(l);
                }
                if (l.LayerType == LayerType.Connected && l.BatchNormalize)
                {
                    denormalize_connected_layer(l);
                }
                if (l.LayerType == LayerType.Gru && l.BatchNormalize)
                {
                    denormalize_connected_layer(l.InputZLayer);
                    denormalize_connected_layer(l.InputRLayer);
                    denormalize_connected_layer(l.InputHLayer);
                    denormalize_connected_layer(l.StateZLayer);
                    denormalize_connected_layer(l.StateRLayer);
                    denormalize_connected_layer(l.StateHLayer);
                }
            }
            save_weights(net, outfile);
        }

        public static Layer normalize_layer(Layer l, int n)
        {
            int j;
            l.BatchNormalize = true;
            l.Scales = calloc(n, sizeof(float));
            for (j = 0; j < n; ++j)
            {
                l.Scales[j] = 1;
            }
            l.RollingMean = calloc(n, sizeof(float));
            l.RollingVariance = calloc(n, sizeof(float));
            return l;
        }

        public static void normalize_net(string cfgfile, string weightfile, string outfile)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
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
            save_weights(net, outfile);
        }

        public static void statistics_net(string cfgfile, string weightfile)
        {
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
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
            CudaUtils.UseGpu = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
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
            save_weights(net, outfile);
        }

        public static void visualize(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            visualize_network(net);
            cvWaitKey(0);
        }

        #region yolo-file

        string[] voc_names = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

        void train_yolo(string cfgfile, string weightfile)
        {
            string train_images = "/Data.Data/voc/train.txt";
            string backup_directory = "/home/pjreddie/backup/";
            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data train, buffer;


            Layer l = net.Layers[net.N - 1];

            int side = l.Side;
            int classes = l.Classes;
            float jitter = l.Jitter;

            list* plist = get_paths(train_images);
            //int N = plist.Size;
            string[] paths = (string[])list_to_array(plist);

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.N = imgs;
            args.m = plist.Size;
            args.Classes = classes;
            args.Jitter = jitter;
            args.num_boxes = side;
            args.d = &buffer;
            args.LayerType = REGION_DATA;

            args.angle = net.angle;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;

            pthread_t load_thread = load_data_in_thread(args);
            clock_t time;
            //while(i*imgs < N*120){
            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data_in_thread(args);

                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void print_yolo_detections(FILE** fps, string id, box* boxes, float[]* probs, int total, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].x - boxes[i].W / 2.;
                float xmax = boxes[i].x + boxes[i].W / 2.;
                float ymin = boxes[i].y - boxes[i].H / 2.;
                float ymax = boxes[i].y + boxes[i].H / 2.;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                for (j = 0; j < classes; ++j)
                {
                    if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                            xmin, ymin, xmax, ymax);
                }
            }
        }

        void validate_yolo(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            

            string basec = "results/comp4_det_test_";
            //list *plist = get_paths("Data.Data/voc.2007.test");
            list* plist = get_paths("/home/pjreddie/Data.Data/voc/2007_test.txt");
            //list *plist = get_paths("Data.Data/voc.2012.test");
            string[] paths = (string[])list_to_array(plist);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;

            int j;
            FILE** fps = (FILE**)calloc(classes, sizeof(FILE*));
            for (j = 0; j < classes; ++j)
            {
                char buff[1024];
                snprintf(buff, 1024, "%s%s.txt", basec, voc_names[j]);
                fps[j] = fopen(buff, "w");
            }
            box* boxes = (box*)calloc(l.Side * l.Side * l.N, sizeof(box));
            float[]*probs = (float[] *)calloc(l.Side * l.Side * l.N, sizeof(float[]));
            for (j = 0; j < l.Side * l.Side * l.N; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.Size;
            int i = 0;
            int t;

            float thresh = .001;
            int nms = 1;
            float iou_thresh = .5;

            int nthreads = 8;
            Image* val = (Image*)calloc(nthreads, sizeof(Image));
            Image* val_resized = (Image*)calloc(nthreads, sizeof(Image));
            Image* buf = (Image*)calloc(nthreads, sizeof(Image));
            Image* buf_resized = (Image*)calloc(nthreads, sizeof(Image));
            pthread_t* thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.LayerType = IMAGE_DATA;

            for (t = 0; t < nthreads; ++t)
            {
                args.path = paths[i + t];
                args.im = &buf[t];
                args.resized = &buf_resized[t];
                thr[t] = load_data_in_thread(args);
            }
            time_t start = time(0);
            for (i = nthreads; i < m + nthreads; i += nthreads)
            {
                Console.Error.Write($"%d\n", i);
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    pthread_join(thr[t], 0);
                    val[t] = buf[t];
                    val_resized[t] = buf_resized[t];
                }
                for (t = 0; t < nthreads && i + t < m; ++t)
                {
                    args.path = paths[i + t];
                    args.im = &buf[t];
                    args.resized = &buf_resized[t];
                    thr[t] = load_data_in_thread(args);
                }
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    string path = paths[i + t - nthreads];
                    string id = basecfg(path);
                    float[] X = val_resized[t].Data.Data;
                    network_predict(net, X);
                    int w = val[t].W;
                    int h = val[t].H;
                    get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
                    if (nms) do_nms_sort(boxes, probs, l.Side * l.Side * l.N, classes, iou_thresh);
                    print_yolo_detections(fps, id, boxes, probs, l.Side * l.Side * l.N, classes, w, h);
                    free(id);
                    free_image(val[t]);
                    free_image(val_resized[t]);
                }
            }
            Console.Error.Write($"Total Detection Time: %f Seconds\n", (double)(time(0) - start));
        }

        void validate_yolo_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            

            string basec = "results/comp4_det_test_";
            list* plist = get_paths("Data.Data/voc.2007.test");
            string[] paths = (string[])list_to_array(plist);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            int side = l.Side;

            int j, k;
            FILE** fps = (FILE**)calloc(classes, sizeof(FILE*));
            for (j = 0; j < classes; ++j)
            {
                char buff[1024];
                snprintf(buff, 1024, "%s%s.txt", basec, voc_names[j]);
                fps[j] = fopen(buff, "w");
            }
            box* boxes = (box*)calloc(side * side * l.N, sizeof(box));
            float[]*probs = (float[] *)calloc(side * side * l.N, sizeof(float[]));
            for (j = 0; j < side * side * l.N; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.Size;
            int i = 0;

            float thresh = .001;
            float iou_thresh = .5;
            float nms = 0;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avg_iou = 0;

            for (i = 0; i < m; ++i)
            {
                string path = paths[i];
                Image orig = load_image_color(path, 0, 0);
                Image sized = resize_image(orig, net.W, net.H);
                string id = basecfg(path);
                network_predict(net, sized.Data.Data);
                get_detection_boxes(l, orig.W, orig.H, thresh, probs, boxes, 1);
                if (nms) do_nms(boxes, probs, side * side * l.N, 1, nms);

                char labelpath[4096];
                find_replace(path, "images", "labels", labelpath);
                find_replace(labelpath, "JPEGImages", "labels", labelpath);
                find_replace(labelpath, ".jpg", ".txt", labelpath);
                find_replace(labelpath, ".JPEG", ".txt", labelpath);

                int num_labels = 0;
                box_label* truth = read_boxes(labelpath, &num_labels);
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
                    box t = { truth[j].x, truth[j].y, truth[j].W, truth[j].H };
                    float best_iou = 0;
                    for (k = 0; k < side * side * l.N; ++k)
                    {
                        float iou = box_iou(boxes[k], t);
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

                Console.Error.Write($"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.* correct / total);
                free(id);
                free_image(orig);
                free_image(sized);
            }
        }

        void test_yolo(string cfgfile, string weightfile, string filename, float thresh)
        {
            Image** alphabet = load_alphabet();
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            detection_layer l = net.Layers[net.N - 1];
            set_batch_network(net, 1);
            srand(2222222);
            clock_t time;
            char buff[256];
            string input = buff;
            int j;
            float nms = .4;
            box* boxes = (box*)calloc(l.Side * l.Side * l.N, sizeof(box));
            float[]*probs = (float[] *)calloc(l.Side * l.Side * l.N, sizeof(float[]));
            for (j = 0; j < l.Side * l.Side * l.N; ++j) probs[j] = (float[])calloc(l.Classes, sizeof(float[]));
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                Image sized = resize_image(im, net.W, net.H);
                float[] X = sized.Data.Data;
                time = clock();
                network_predict(net, X);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
                if (nms) do_nms_sort(boxes, probs, l.Side * l.Side * l.N, l.Classes, nms);
                //draw_detections(im, l.sidel.sidel.N, thresh, boxes, probs, voc_names, alphabet, 20);
                draw_detections(im, l.Side * l.Side * l.N, thresh, boxes, probs, voc_names, alphabet, 20);
                save_image(im, "predictions");
                show_image(im, "predictions");

                free_image(im);
                free_image(sized);
                cvWaitKey(0);
                cvDestroyAllWindows();
                if (filename) break;
            }
        }

        public static void run_yolo(List<string> args)
        {
            string prefix = find_char_arg(args.Count, args, "-prefix", 0);
            float thresh = find_float_arg(args.Count, args, "-thresh", .2);
            int cam_index = Utils.find_int_arg(args.Count, args, "-c", 0);
            int frame_skip = Utils.find_int_arg(args.Count, args, "-s", 0);
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "test")) test_yolo(cfg, weights, filename, thresh);
            else if (0 == strcmp(args[2], "train")) train_yolo(cfg, weights);
            else if (0 == strcmp(args[2], "valid")) validate_yolo(cfg, weights);
            else if (0 == strcmp(args[2], "recall")) validate_yolo_recall(cfg, weights);
            else if (0 == strcmp(args[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix);
        }

        #endregion

        #region voxelFile

        void extract_voxel(string lfile, string rfile, string prefix)
        {
            int w = 1920;
            int h = 1080;
            int shift = 0;
            int count = 0;
            CvCapture* lcap = cvCaptureFromFile(lfile);
            CvCapture* rcap = cvCaptureFromFile(rfile);
            while (1)
            {
                Image l = get_image_from_stream(lcap);
                Image r = get_image_from_stream(rcap);
                if (!l.W || !r.W) break;
                if (count % 100 == 0)
                {
                    shift = best_3d_shift_r(l, r, -l.H / 100, l.H / 100);
                    Console.Write($"%d\n", shift);
                }
                Image ls = crop_image(l, (l.W - w) / 2, (l.H - h) / 2, w, h);
                Image rs = crop_image(r, 105 + (r.W - w) / 2, (r.H - h) / 2 + shift, w, h);
                char buff[256];
                sprintf(buff, "%s_%05d_l", prefix, count);
                save_image(ls, buff);
                sprintf(buff, "%s_%05d_r", prefix, count);
                save_image(rs, buff);
                free_image(l);
                free_image(r);
                free_image(ls);
                free_image(rs);
                ++count;
            }
        }

        void train_voxel(string cfgfile, string weightfile)
        {
            string train_images = "/Data.Data/imagenet/imagenet1k.train.list";
            string backup_directory = "/home/pjreddie/backup/";
            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data train, buffer;


            list* plist = get_paths(train_images);
            //int N = plist.Size;
            string[] paths = (string[])list_to_array(plist);

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.scale = 4;
            args.paths = paths;
            args.N = imgs;
            args.m = plist.Size;
            args.d = &buffer;
            args.LayerType = SUPER_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            clock_t time;
            //while(i*imgs < N*120){
            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data_in_thread(args);

                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void test_voxel(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);

            clock_t time;
            char buff[256];
            string input = buff;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                resize_network(net, im.W, im.H);
                Console.Write($"%d %d\n", im.W, im.H);

                float[] X = im.Data.Data;
                time = clock();
                network_predict(net, X);
                Image outi = get_network_image(net);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                save_image(outi, "outf");

                free_image(im);
                if (filename) break;
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
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "train")) train_voxel(cfg, weights);
            else if (0 == strcmp(args[2], "test")) test_voxel(cfg, weights, filename);
            else if (0 == strcmp(args[2], "extract")) extract_voxel(args[3], args[4], args[5]);
        }

        #endregion

        #region SuperFile

        void train_super(string cfgfile, string weightfile)
        {
            string train_images = "/Data.Data/imagenet/imagenet1k.train.list";
            string backup_directory = "/home/pjreddie/backup/";
            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data train, buffer;


            list* plist = get_paths(train_images);
            //int N = plist.Size;
            string[] paths = (string[])list_to_array(plist);

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.scale = 4;
            args.paths = paths;
            args.N = imgs;
            args.m = plist.Size;
            args.d = &buffer;
            args.LayerType = SUPER_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            clock_t time;
            //while(i*imgs < N*120){
            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data_in_thread(args);

                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void test_super(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);

            clock_t time;
            char buff[256];
            string input = buff;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                resize_network(net, im.W, im.H);
                Console.Write($"%d %d\n", im.W, im.H);

                float[] X = im.Data.Data;
                time = clock();
                network_predict(net, X);
                Image outi = get_network_image(net);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                save_image(outi, "outf");

                free_image(im);
                if (filename) break;
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
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "train")) train_super(cfg, weights);
            else if (0 == strcmp(args[2], "test")) test_super(cfg, weights, filename);
        }

        #endregion

        #region DetectorFile

        static int[] coco_ids = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };

        void train_detector(string datacfg, string cfgfile, string weightfile, int[] gpus, int ngpus, int clear)
        {
            list* options = read_data_cfg(datacfg);
            string train_images = option_find_str(options, "train", "Data.Data/train.list");
            string backup_directory = option_find_str(options, "backup", "/backup/");

            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network* nets = calloc(ngpus, sizeof(Network));

            
            int seed = rand();
            int i;
            for (i = 0; i < ngpus; ++i)
            {
                srand(seed);
                cuda_set_device(gpus[i]);
                nets[i] = Parser.parse_network_cfg(cfgfile);
                if (weightfile)
                {
                    Parser.load_weights(&nets[i], weightfile);
                }
                if (clear) *nets[i].Seen = 0;
                nets[i].LearningRate *= ngpus;
            }
            
            Network net = nets[0];

            int imgs = net.Batch * net.Subdivisions * ngpus;
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            Data.Data train, buffer;

            Layer l = net.Layers[net.N - 1];

            int classes = l.Classes;
            float jitter = l.Jitter;

            list* plist = get_paths(train_images);
            //int N = plist.Size;
            string[] paths = (string[])list_to_array(plist);

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.N = imgs;
            args.m = plist.Size;
            args.Classes = classes;
            args.Jitter = jitter;
            args.num_boxes = l.max_boxes;
            args.d = &buffer;
            args.LayerType = DETECTION_DATA;
            args.threads = 8;

            args.angle = net.angle;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;

            pthread_t load_thread = load_data(args);
            clock_t time;
            int count = 0;
            //while(i*imgs < N*120){
            while (get_current_batch(net) < net.max_batches)
            {
                if (l.random && count++ % 10 == 0)
                {
                    Console.Write($"Resizing\n");
                    int dim = (rand() % 10 + 10) * 32;
                    if (get_current_batch(net) + 100 > net.max_batches) dim = 544;
                    //int dim = (rand() % 4 + 16) * 32;
                    Console.Write($"%d\n", dim);
                    args.W = dim;
                    args.H = dim;

                    pthread_join(load_thread, 0);
                    train = buffer;
                    free_data(train);
                    load_thread = load_data(args);

                    for (i = 0; i < ngpus; ++i)
                    {
                        resize_network(nets + i, dim, dim);
                    }
                    net = nets[0];
                }
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data(args);

                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = 0;
                if (ngpus == 1)
                {
                    loss = train_network(net, train);
                }
                else
                {
                    loss = train_networks(nets, ngpus, train, 4);
                }
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                i = get_current_batch(net);
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    if (ngpus != 1) sync_nets(nets, ngpus, 0);
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }


        static int get_coco_image_id(string filename)
        {
            string p = strrchr(filename, '_');
            return int.Parse(p + 1);
        }

        static void print_cocos(FILE* fp, string image_path, box* boxes, float[]* probs, int num_boxes, int classes, int w, int h)
        {
            int i, j;
            int image_id = get_coco_image_id(image_path);
            for (i = 0; i < num_boxes; ++i)
            {
                float xmin = boxes[i].x - boxes[i].W / 2.;
                float xmax = boxes[i].x + boxes[i].W / 2.;
                float ymin = boxes[i].y - boxes[i].H / 2.;
                float ymax = boxes[i].y + boxes[i].H / 2.;

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
                    if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
                }
            }
        }

        void print_detector_detections(FILE** fps, string id, box* boxes, float[]* probs, int total, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].x - boxes[i].W / 2.;
                float xmax = boxes[i].x + boxes[i].W / 2.;
                float ymin = boxes[i].y - boxes[i].H / 2.;
                float ymax = boxes[i].y + boxes[i].H / 2.;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                for (j = 0; j < classes; ++j)
                {
                    if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                            xmin, ymin, xmax, ymax);
                }
            }
        }

        void print_imagenet_detections(FILE* fp, int id, box* boxes, float[]* probs, int total, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].x - boxes[i].W / 2.;
                float xmax = boxes[i].x + boxes[i].W / 2.;
                float ymin = boxes[i].y - boxes[i].H / 2.;
                float ymax = boxes[i].y + boxes[i].H / 2.;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > w) xmax = w;
                if (ymax > h) ymax = h;

                for (j = 0; j < classes; ++j)
                {
                    int class2 = j;
                    if (probs[i][class2]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, probs[i][class2],
                            xmin, ymin, xmax, ymax);
                }
            }
        }

        void validate_detector(string datacfg, string cfgfile, string weightfile)
        {
            int j;
            list* options = read_data_cfg(datacfg);
            string valid_images = option_find_str(options, "valid", "Data.Data/train.list");
            string name_list = option_find_str(options, "names", "Data.Data/names.list");
            string prefix = option_find_str(options, "results", "results");
            string[] names = get_labels(name_list);
            string mapf = option_find_str(options, "map", 0);
            int[] map = 0;
            if (mapf) map = read_map(mapf);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            

            string basec = "comp4_det_test_";
            list* plist = get_paths(valid_images);
            string[] paths = (string[])list_to_array(plist);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;

            char buff[1024];
            string type = option_find_str(options, "eval", "voc");
            FILE* fp = 0;
            FILE** fps = 0;
            int coco = 0;
            int imagenet = 0;
            if (0 == strcmp(type, "coco"))
            {
                snprintf(buff, 1024, "%s/coco_results.json", prefix);
                fp = fopen(buff, "w");
                fprintf(fp, "[\n");
                coco = 1;
            }
            else if (0 == strcmp(type, "imagenet"))
            {
                snprintf(buff, 1024, "%s/imagenet-detection.txt", prefix);
                fp = fopen(buff, "w");
                imagenet = 1;
                classes = 200;
            }
            else
            {
                fps = calloc(classes, sizeof(FILE*));
                for (j = 0; j < classes; ++j)
                {
                    snprintf(buff, 1024, "%s/%s%s.txt", prefix, basec, names[j]);
                    fps[j] = fopen(buff, "w");
                }
            }


            box* boxes = calloc(l.W * l.H * l.N, sizeof(box));
            float[]*probs = calloc(l.W * l.H * l.N, sizeof(float[]));
            for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = calloc(classes, sizeof(float[]));

            int m = plist.Size;
            int i = 0;
            int t;

            float thresh = .005;
            float nms = .45;

            int nthreads = 4;
            Image* val = calloc(nthreads, sizeof(Image));
            Image* val_resized = calloc(nthreads, sizeof(Image));
            Image* buf = calloc(nthreads, sizeof(Image));
            Image* buf_resized = calloc(nthreads, sizeof(Image));
            pthread_t* thr = calloc(nthreads, sizeof(pthread_t));

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.LayerType = IMAGE_DATA;

            for (t = 0; t < nthreads; ++t)
            {
                args.path = paths[i + t];
                args.im = &buf[t];
                args.resized = &buf_resized[t];
                thr[t] = load_data_in_thread(args);
            }
            time_t start = time(0);
            for (i = nthreads; i < m + nthreads; i += nthreads)
            {
                Console.Error.Write($"%d\n", i);
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    pthread_join(thr[t], 0);
                    val[t] = buf[t];
                    val_resized[t] = buf_resized[t];
                }
                for (t = 0; t < nthreads && i + t < m; ++t)
                {
                    args.path = paths[i + t];
                    args.im = &buf[t];
                    args.resized = &buf_resized[t];
                    thr[t] = load_data_in_thread(args);
                }
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    string path = paths[i + t - nthreads];
                    string id = basecfg(path);
                    float[] X = val_resized[t].Data.Data;
                    network_predict(net, X);
                    int w = val[t].W;
                    int h = val[t].H;
                    get_region_boxes(l, w, h, thresh, probs, boxes, 0, map);
                    if (nms) do_nms_sort(boxes, probs, l.W * l.H * l.N, classes, nms);
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
                    free(id);
                    free_image(val[t]);
                    free_image(val_resized[t]);
                }
            }
            for (j = 0; j < classes; ++j)
            {
                if (fps) fclose(fps[j]);
            }
            if (coco)
            {
                fseek(fp, -2, SEEK_CUR);
                fprintf(fp, "\n]\n");
                fclose(fp);
            }
            Console.Error.Write($"Total Detection Time: %f Seconds\n", (double)(time(0) - start));
        }

        void validate_detector_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            

            list* plist = get_paths("Data.Data/voc.2007.test");
            string[] paths = (string[])list_to_array(plist);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;

            int j, k;
            box* boxes = calloc(l.W * l.H * l.N, sizeof(box));
            float[]*probs = calloc(l.W * l.H * l.N, sizeof(float[]));
            for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = calloc(classes, sizeof(float[]));

            int m = plist.Size;
            int i = 0;

            float thresh = .001;
            float iou_thresh = .5;
            float nms = .4;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avg_iou = 0;

            for (i = 0; i < m; ++i)
            {
                string path = paths[i];
                Image orig = load_image_color(path, 0, 0);
                Image sized = resize_image(orig, net.W, net.H);
                string id = basecfg(path);
                network_predict(net, sized.Data.Data);
                get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0);
                if (nms) do_nms(boxes, probs, l.W * l.H * l.N, 1, nms);

                char labelpath[4096];
                find_replace(path, "images", "labels", labelpath);
                find_replace(labelpath, "JPEGImages", "labels", labelpath);
                find_replace(labelpath, ".jpg", ".txt", labelpath);
                find_replace(labelpath, ".JPEG", ".txt", labelpath);

                int num_labels = 0;
                box_label* truth = read_boxes(labelpath, &num_labels);
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
                    box t = { truth[j].x, truth[j].y, truth[j].W, truth[j].H };
                    float best_iou = 0;
                    for (k = 0; k < l.W * l.H * l.N; ++k)
                    {
                        float iou = box_iou(boxes[k], t);
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

                Console.Error.Write($"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.* correct / total);
                free(id);
                free_image(orig);
                free_image(sized);
            }
        }

        public static void test_detector(string datacfg, string cfgfile, string weightfile, string filename, float thresh)
        {
            list* options = read_data_cfg(datacfg);
            string name_list = option_find_str(options, "names", "Data.Data/names.list");
            string[] names = get_labels(name_list);

            Image** alphabet = load_alphabet();
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);
            clock_t time;
            char buff[256];
            string input = buff;
            int j;
            float nms = .4;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                Image sized = resize_image(im, net.W, net.H);
                Layer l = net.Layers[net.N - 1];

                box* boxes = calloc(l.W * l.H * l.N, sizeof(box));
                float[]*probs = calloc(l.W * l.H * l.N, sizeof(float[]));
                for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = calloc(l.Classes, sizeof(float[]));

                float[] X = sized.Data.Data;
                time = clock();
                network_predict(net, X);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
                if (nms) do_nms_sort(boxes, probs, l.W * l.H * l.N, l.Classes, nms);
                draw_detections(im, l.W * l.H * l.N, thresh, boxes, probs, names, alphabet, l.Classes);
                save_image(im, "predictions");
                show_image(im, "predictions");

                free_image(im);
                free_image(sized);
                free(boxes);
                free_ptrs((void**)probs, l.W * l.H * l.N);
                cvWaitKey(0);
                cvDestroyAllWindows();
                if (filename) break;
            }
        }

        public static void run_detector(List<string> args)
        {
            string prefix = find_char_arg(args.Count, args, "-prefix", 0);
            float thresh = find_float_arg(args.Count, args, "-thresh", .24);
            int cam_index = Utils.find_int_arg(args.Count, args, "-c", 0);
            int frame_skip = Utils.find_int_arg(args.Count, args, "-s", 0);
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }
            string gpu_list = find_char_arg(args.Count, args, "-gpus", 0);
            int[] gpus = 0;
            int gpu = 0;
            int ngpus = 0;
            if (gpu_list)
            {
                Console.Write($"%s\n", gpu_list);
                int len = strlen(gpu_list);
                ngpus = 1;
                int i;
                for (i = 0; i < len; ++i)
                {
                    if (gpu_list[i] == ',') ++ngpus;
                }
                gpus = calloc(ngpus, sizeof(int));
                for (i = 0; i < ngpus; ++i)
                {
                    gpus[i] = int.Parse(gpu_list);
                    gpu_list = strchr(gpu_list, ',') + 1;
                }
            }
            else
            {
                gpu = CudaUtils.UseGpu;
                gpus = &gpu;
                ngpus = 1;
            }

            int clear = find_arg(args.Count, args, "-clear");

            string datacfg = args[3];
            string cfg = args[4];
            string weights = (args.Count > 5) ? args[5] : 0;
            string filename = (args.Count > 6) ? args[6] : 0;
            if (0 == strcmp(args[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh);
            else if (0 == strcmp(args[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
            else if (0 == strcmp(args[2], "valid")) validate_detector(datacfg, cfg, weights);
            else if (0 == strcmp(args[2], "recall")) validate_detector_recall(cfg, weights);
            else if (0 == strcmp(args[2], "demo"))
            {
                list* options = read_data_cfg(datacfg);
                int classes = option_find_int(options, "classes", 20);
                string name_list = option_find_str(options, "names", "Data.Data/names.list");
                string[] names = get_labels(name_list);
                demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix);
            }
        }

        #endregion

        #region CiFarFile

        void train_cifar(string cfgfile, string weightfile)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            string backup_directory = "/home/pjreddie/backup/";
            int classes = 10;
            int N = 50000;

            string[] labels = get_labels("Data.Data/cifar/labels.txt");
            int epoch = (net.Seen) / N;
            Data.Data train = load_all_cifar10();
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                clock_t time = clock();

                float loss = train_network_sgd(net, train, 1);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95 + loss * .05;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), net.Seen);
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                }
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
            }
            char buff[256];
            sprintf(buff, "%s/%s.Weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free_ptrs((void**)labels, classes);
            free(basec);
            free_data(train);
        }

        void train_cifar_distill(string cfgfile, string weightfile)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            string backup_directory = "/home/pjreddie/backup/";
            int classes = 10;
            int N = 50000;

            string[] labels = get_labels("Data.Data/cifar/labels.txt");
            int epoch = (net.Seen) / N;

            Data.Data train = load_all_cifar10();
            Matrix soft = csv_to_matrix("results/ensemble.csv");

            float weight = .9;
            scale_matrix(soft, weight);
            scale_matrix(train.y, 1. - weight);
            matrix_add_matrix(soft, train.y);

            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                clock_t time = clock();

                float loss = train_network_sgd(net, train, 1);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95 + loss * .05;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), net.Seen);
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                }
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
            }
            char buff[256];
            sprintf(buff, "%s/%s.Weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free_ptrs((void**)labels, classes);
            free(basec);
            free_data(train);
        }

        void test_cifar_multi(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            

            float avg_acc = 0;
            Data.Data test = load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            int i;
            for (i = 0; i < test.X.rows; ++i)
            {
                Image im = float_to_image(32, 32, 3, test.X.vals[i]);

                float pred[10] = { 0 };

                float[] p = network_predict(net, im.Data.Data);
                axpy_cpu(10, 1, p, 1, pred, 1);
                flip_image(im);
                p = network_predict(net, im.Data.Data);
                axpy_cpu(10, 1, p, 1, pred, 1);

                int index = max_index(pred, 10);
                int sclass = max_index(test.y.vals[i], 10);
                if (index == sclass) avg_acc += 1;
                free_image(im);
                Console.Write($"%4d: %.2f%%\n", i, 100.* avg_acc / (i + 1));
            }
        }

        void test_cifar(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            clock_t time;
            float avg_acc = 0;
            float avg_top5 = 0;
            Data.Data test = load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            time = clock();

            float[] acc = network_accuracies(net, test, 2);
            avg_acc += acc[0];
            avg_top5 += acc[1];
            Console.Write($"top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock() - time), test.X.rows);
            free_data(test);
        }

        void extract_cifar()
        {
            string labels[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
            int i;
            Data.Data train = load_all_cifar10();
            Data.Data test = load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");
            for (i = 0; i < train.X.rows; ++i)
            {
                Image im = float_to_image(32, 32, 3, train.X.vals[i]);
                int sclass = max_index(train.y.vals[i], 10);
                char buff[256];
                sprintf(buff, "Data.Data/cifar/train/%d_%s", i, labels[sclass]);
                save_image_png(im, buff);
            }
            for (i = 0; i < test.X.rows; ++i)
            {
                Image im = float_to_image(32, 32, 3, test.X.vals[i]);
                int sclass = max_index(test.y.vals[i], 10);
                char buff[256];
                sprintf(buff, "Data.Data/cifar/test/%d_%s", i, labels[sclass]);
                save_image_png(im, buff);
            }
        }

        void test_cifar_csv(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            Data.Data test = load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            Matrix pred = network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.rows; ++i)
            {
                Image im = float_to_image(32, 32, 3, test.X.vals[i]);
                flip_image(im);
            }
            Matrix pred2 = network_predict_data(net, test);
            scale_matrix(pred, .5);
            scale_matrix(pred2, .5);
            matrix_add_matrix(pred2, pred);

            matrix_to_csv(pred);
            Console.Error.Write($"Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
            free_data(test);
        }

        void test_cifar_csvtrain(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            Data.Data test = load_all_cifar10();

            Matrix pred = network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.rows; ++i)
            {
                Image im = float_to_image(32, 32, 3, test.X.vals[i]);
                flip_image(im);
            }
            Matrix pred2 = network_predict_data(net, test);
            scale_matrix(pred, .5);
            scale_matrix(pred2, .5);
            matrix_add_matrix(pred2, pred);

            matrix_to_csv(pred);
            Console.Error.Write($"Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
            free_data(test);
        }

        void eval_cifar_csv()
        {
            Data.Data test = load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            Matrix pred = csv_to_matrix("results/combined.csv");
            Console.Error.Write($"%d %d\n", pred.rows, pred.cols);

            Console.Error.Write($"Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
            free_data(test);
            free_matrix(pred);
        }


        public static void run_cifar(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            if (0 == strcmp(args[2], "train")) train_cifar(cfg, weights);
            else if (0 == strcmp(args[2], "extract")) extract_cifar();
            else if (0 == strcmp(args[2], "distill")) train_cifar_distill(cfg, weights);
            else if (0 == strcmp(args[2], "test")) test_cifar(cfg, weights);
            else if (0 == strcmp(args[2], "multi")) test_cifar_multi(cfg, weights);
            else if (0 == strcmp(args[2], "csv")) test_cifar_csv(cfg, weights);
            else if (0 == strcmp(args[2], "csvtrain")) test_cifar_csvtrain(cfg, weights);
            else if (0 == strcmp(args[2], "eval")) eval_cifar_csv();
        }

        #endregion

        #region GoFile

        static int inverted = 1;
        static int noi = 1;
        static int nind = 5;


        string fgetgo(FILE* fp)
        {
            if (feof(fp)) return 0;
            size_t size = 94;
            string line = malloc(size * sizeof(char));
            if (size != fread(line, sizeof(char), size, fp))
            {
                free(line);
                return 0;
            }

            return line;
        }

        moves load_go_moves(string filename)
        {
            moves m;
            m.N = 128;
            Data.Data.Data = calloc(128, sizeof(string));
            FILE* fp = fopen(filename, "rb");
            int count = 0;
            string line = 0;
            while ((line = fgetgo(fp)))
            {
                if (count >= m.N)
                {
                    m.N *= 2;
                    Data.Data.Data = realloc(Data.Data.Data, m.N * sizeof(string));
                }
                Data.Data.Data[count] = line;
                ++count;
            }
            Console.Write($"%d\n", count);
            m.N = count;
            Data.Data.Data = realloc(Data.Data.Data, count * sizeof(string));
            return m;
        }

        void string_to_board(string s, float[] board)
        {
            int i, j;
            //memset(board, 0, 1*19*19*sizeof(float));
            int count = 0;
            for (i = 0; i < 91; ++i)
            {
                char c = s[i];
                for (j = 0; j < 4; ++j)
                {
                    int me = (c >> (2 * j)) & 1;
                    int you = (c >> (2 * j + 1)) & 1;
                    if (me) board[count] = 1;
                    else if (you) board[count] = -1;
                    else board[count] = 0;
                    ++count;
                    if (count >= 19 * 19) break;
                }
            }
        }

        void board_to_string(string s, float[] board)
        {
            int i, j;
            memset(s, 0, (19 * 19 / 4 + 1) * sizeof(char));
            int count = 0;
            for (i = 0; i < 91; ++i)
            {
                for (j = 0; j < 4; ++j)
                {
                    int me = (board[count] == 1);
                    int you = (board[count] == -1);
                    if (me) s[i] = s[i] | (1 << (2 * j));
                    if (you) s[i] = s[i] | (1 << (2 * j + 1));
                    ++count;
                    if (count >= 19 * 19) break;
                }
            }
        }

        void random_go_moves(moves m, float[] boards, float[] labels, int n)
        {
            int i;
            memset(labels, 0, 19 * 19 * n * sizeof(float));
            for (i = 0; i < n; ++i)
            {
                string b = Data.Data.Data[rand() % m.N];
                int row = b[0];
                int col = b[1];
                labels[col + 19 * (row + i * 19)] = 1;
                string_to_board(b + 2, boards + i * 19 * 19);
                boards[col + 19 * (row + i * 19)] = 0;

                int flip = rand() % 2;
                int rotate = rand() % 4;
                Image ini = float_to_image(19, 19, 1, boards + i * 19 * 19);
                Image outi = float_to_image(19, 19, 1, labels + i * 19 * 19);
                if (flip)
                {
                    flip_image(ini);
                    flip_image(outi);
                }
                rotate_image_cw(ini, rotate);
                rotate_image_cw(outi, rotate);
            }
        }


        void train_go(string cfgfile, string weightfile)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            string backup_directory = "/home/pjreddie/backup/";

            char buff[256];
            float[] board = calloc(19 * 19 * net.Batch, sizeof(float));
            float[] move = calloc(19 * 19 * net.Batch, sizeof(float));
            moves m = load_go_moves("/home/pjreddie/backup/go.train");
            //moves m = load_go_moves("games.txt");

            int N = m.N;
            int epoch = (net.Seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                clock_t time = clock();

                random_go_moves(m, board, move, net.Batch);
                float loss = train_network_datum(net, board, move) / net.Batch;
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95 + loss * .05;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), net.Seen);
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);

                }
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
                if (get_current_batch(net) % 10000 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.backup", backup_directory, basec, get_current_batch(net));
                    save_weights(net, buff);
                }
            }
            sprintf(buff, "%s/%s.Weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free(basec);
        }

        void propagate_liberty(float[] board, int[] lib, int[] visited, int row, int col, int side)
        {
            if (row < 0 || row > 18 || col < 0 || col > 18) return;
            int index = row * 19 + col;
            if (board[index] != side) return;
            if (visited[index]) return;
            visited[index] = 1;
            lib[index] += 1;
            propagate_liberty(board, lib, visited, row + 1, col, side);
            propagate_liberty(board, lib, visited, row - 1, col, side);
            propagate_liberty(board, lib, visited, row, col + 1, side);
            propagate_liberty(board, lib, visited, row, col - 1, side);
        }


        int[] calculate_liberties(float[] board)
        {
            int[] lib = calloc(19 * 19, sizeof(int));
            int visited[361];
            int i, j;
            for (j = 0; j < 19; ++j)
            {
                for (i = 0; i < 19; ++i)
                {
                    memset(visited, 0, 19 * 19 * sizeof(int));
                    int index = j * 19 + i;
                    if (board[index] == 0)
                    {
                        if ((i > 0) && board[index - 1]) propagate_liberty(board, lib, visited, j, i - 1, board[index - 1]);
                        if ((i < 18) && board[index + 1]) propagate_liberty(board, lib, visited, j, i + 1, board[index + 1]);
                        if ((j > 0) && board[index - 19]) propagate_liberty(board, lib, visited, j - 1, i, board[index - 19]);
                        if ((j < 18) && board[index + 19]) propagate_liberty(board, lib, visited, j + 1, i, board[index + 19]);
                    }
                }
            }
            return lib;
        }

        void print_board(float[] board, int swap, int[] indexes)
        {
            //FILE *stream = stdout;
            FILE* stream = stderr;
            int i, j, n;
            fprintf(stream, "\n\n");
            fprintf(stream, "   ");
            for (i = 0; i < 19; ++i)
            {
                fprintf(stream, "%c ", 'A' + i + 1 * (i > 7 && noi));
            }
            fprintf(stream, "\n");
            for (j = 0; j < 19; ++j)
            {
                fprintf(stream, "%2d", (inverted) ? 19 - j : j + 1);
                for (i = 0; i < 19; ++i)
                {
                    int index = j * 19 + i;
                    if (indexes)
                    {
                        int found = 0;
                        for (n = 0; n < nind; ++n)
                        {
                            if (index == indexes[n])
                            {
                                found = 1;
                                /*
                                if(n == 0) fprintf(stream, "\uff11");
                                else if(n == 1) fprintf(stream, "\uff12");
                                else if(n == 2) fprintf(stream, "\uff13");
                                else if(n == 3) fprintf(stream, "\uff14");
                                else if(n == 4) fprintf(stream, "\uff15");
                                */
                                if (n == 0) fprintf(stream, " 1");
                                else if (n == 1) fprintf(stream, " 2");
                                else if (n == 2) fprintf(stream, " 3");
                                else if (n == 3) fprintf(stream, " 4");
                                else if (n == 4) fprintf(stream, " 5");
                            }
                        }
                        if (found) continue;
                    }
                    //if(board[index]*-swap > 0) fprintf(stream, "\u25C9 ");
                    //else if(board[index]*-swap < 0) fprintf(stream, "\u25EF ");
                    if (board[index] * -swap > 0) fprintf(stream, " O");
                    else if (board[index] * -swap < 0) fprintf(stream, " X");
                    else fprintf(stream, "  ");
                }
                fprintf(stream, "\n");
            }
        }

        void flip_board(float[] board)
        {
            int i;
            for (i = 0; i < 19 * 19; ++i)
            {
                board[i] = -board[i];
            }
        }

        void predict_move(Network net, float[] board, float[] move, int multi)
        {
            float[] output = network_predict(net, board);
            Blas.Copy_cpu(19 * 19, output, 1, move, 1);
            int i;
            if (multi)
            {
                Image bim = float_to_image(19, 19, 1, board);
                for (i = 1; i < 8; ++i)
                {
                    rotate_image_cw(bim, i);
                    if (i >= 4) flip_image(bim);

                    float[] output = network_predict(net, board);
                    Image oim = float_to_image(19, 19, 1, output);

                    if (i >= 4) flip_image(oim);
                    rotate_image_cw(oim, -i);

                    axpy_cpu(19 * 19, 1, output, 1, move, 1);

                    if (i >= 4) flip_image(bim);
                    rotate_image_cw(bim, -i);
                }
                Blas.Scal_cpu(19 * 19, 1.0f / 8., move, 1);
            }
            for (i = 0; i < 19 * 19; ++i)
            {
                if (board[i]) move[i] = 0;
            }
        }

        void remove_connected(float[] b, int[] lib, int p, int r, int c)
        {
            if (r < 0 || r >= 19 || c < 0 || c >= 19) return;
            if (b[r * 19 + c] != p) return;
            if (lib[r * 19 + c] != 1) return;
            b[r * 19 + c] = 0;
            remove_connected(b, lib, p, r + 1, c);
            remove_connected(b, lib, p, r - 1, c);
            remove_connected(b, lib, p, r, c + 1);
            remove_connected(b, lib, p, r, c - 1);
        }


        void move_go(float[] b, int p, int r, int c)
        {
            int[] l = calculate_liberties(b);
            b[r * 19 + c] = p;
            remove_connected(b, l, -p, r + 1, c);
            remove_connected(b, l, -p, r - 1, c);
            remove_connected(b, l, -p, r, c + 1);
            remove_connected(b, l, -p, r, c - 1);
            free(l);
        }

        int makes_safe_go(float[] b, int[] lib, int p, int r, int c)
        {
            if (r < 0 || r >= 19 || c < 0 || c >= 19) return 0;
            if (b[r * 19 + c] == -p)
            {
                if (lib[r * 19 + c] > 1) return 0;
                else return 1;
            }
            if (b[r * 19 + c] == 0) return 1;
            if (lib[r * 19 + c] > 1) return 1;
            return 0;
        }

        int suicide_go(float[] b, int p, int r, int c)
        {
            int[] l = calculate_liberties(b);
            int safe = 0;
            safe = safe || makes_safe_go(b, l, p, r + 1, c);
            safe = safe || makes_safe_go(b, l, p, r - 1, c);
            safe = safe || makes_safe_go(b, l, p, r, c + 1);
            safe = safe || makes_safe_go(b, l, p, r, c - 1);
            free(l);
            return !safe;
        }

        int legal_go(float[] b, string ko, int p, int r, int c)
        {
            if (b[r * 19 + c]) return 0;
            char curr[91];
            char next[91];
            board_to_string(curr, b);
            move_go(b, p, r, c);
            board_to_string(next, b);
            string_to_board(curr, b);
            if (memcmp(next, ko, 91) == 0) return 0;
            return 1;
        }

        int generate_move(Network net, int player, float[] board, int multi, float thresh, float temp, string ko, int print)
        {
            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].temperature = temp;

            float move[361];
            if (player < 0) flip_board(board);
            predict_move(net, board, move, multi);
            if (player < 0) flip_board(board);


            for (i = 0; i < 19; ++i)
            {
                for (j = 0; j < 19; ++j)
                {
                    if (!legal_go(board, ko, player, i, j)) move[i * 19 + j] = 0;
                }
            }

            int indexes[nind];
            top_k(move, 19 * 19, nind, indexes);
            if (thresh > move[indexes[0]]) thresh = move[indexes[nind - 1]];

            for (i = 0; i < 19; ++i)
            {
                for (j = 0; j < 19; ++j)
                {
                    if (move[i * 19 + j] < thresh) move[i * 19 + j] = 0;
                }
            }


            int max = max_index(move, 19 * 19);
            int row = max / 19;
            int col = max % 19;
            int index = sample_array(move, 19 * 19);

            if (print)
            {
                top_k(move, 19 * 19, nind, indexes);
                for (i = 0; i < nind; ++i)
                {
                    if (!move[indexes[i]]) indexes[i] = -1;
                }
                print_board(board, player, indexes);
                for (i = 0; i < nind; ++i)
                {
                    Console.Error.Write($"%d: %f\n", i + 1, move[indexes[i]]);
                }
            }

            if (suicide_go(board, player, row, col))
            {
                return -1;
            }
            if (suicide_go(board, player, index / 19, index % 19)) index = max;
            return index;
        }

        void valid_go(string cfgfile, string weightfile, int multi)
        {
            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);

            float[] board = calloc(19 * 19, sizeof(float));
            float[] move = calloc(19 * 19, sizeof(float));
            moves m = load_go_moves("/home/pjreddie/backup/go.test");

            int N = m.N;
            int i;
            int correct = 0;
            for (i = 0; i < N; ++i)
            {
                string b = Data.Data.Data[i];
                int row = b[0];
                int col = b[1];
                int truth = col + 19 * row;
                string_to_board(b + 2, board);
                predict_move(net, board, move, multi);
                int index = max_index(move, 19 * 19);
                if (index == truth) ++correct;
                Console.Write($"%d Accuracy %f\n", i, (float)correct / (i + 1));
            }
        }

        void engine_go(string filename, string weightfile, int multi)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            
            set_batch_network(net, 1);
            float[] board = calloc(19 * 19, sizeof(float));
            string one = calloc(91, sizeof(char));
            string two = calloc(91, sizeof(char));
            int passed = 0;
            while (1)
            {
                char buff[256];
                int id = 0;
                int has_id = (scanf("%d", &id) == 1);
                scanf("%s", buff);
                if (feof(stdin)) break;
                char ids[256];
                sprintf(ids, "%d", id);
                //Console.Error.Write($"%s\n", buff);
                if (!has_id) ids[0] = 0;
                if (!strcmp(buff, "protocol_version"))
                {
                    Console.Write($"=%s 2\n\n", ids);
                }
                else if (!strcmp(buff, "name"))
                {
                    Console.Write($"=%s DarkGo\n\n", ids);
                }
                else if (!strcmp(buff, "version"))
                {
                    Console.Write($"=%s 1.0\n\n", ids);
                }
                else if (!strcmp(buff, "known_command"))
                {
                    char comm[256];
                    scanf("%s", comm);
                    int known = (!strcmp(comm, "protocol_version") ||
                            !strcmp(comm, "name") ||
                            !strcmp(comm, "version") ||
                            !strcmp(comm, "known_command") ||
                            !strcmp(comm, "list_commands") ||
                            !strcmp(comm, "quit") ||
                            !strcmp(comm, "boardsize") ||
                            !strcmp(comm, "clear_board") ||
                            !strcmp(comm, "komi") ||
                            !strcmp(comm, "final_status_list") ||
                            !strcmp(comm, "play") ||
                            !strcmp(comm, "genmove"));
                    if (known) Console.Write($"=%s true\n\n", ids);
                    else Console.Write($"=%s false\n\n", ids);
                }
                else if (!strcmp(buff, "list_commands"))
                {
                    Console.Write($"=%s protocol_version\nname\nversion\nknown_command\nlist_commands\nquit\nboardsize\nclear_board\nkomi\nplay\ngenmove\nfinal_status_list\n\n", ids);
                }
                else if (!strcmp(buff, "quit"))
                {
                    break;
                }
                else if (!strcmp(buff, "boardsize"))
                {
                    int boardsize = 0;
                    scanf("%d", &boardsize);
                    //Console.Error.Write($"%d\n", boardsize);
                    if (boardsize != 19)
                    {
                        Console.Write($"?%s unacceptable size\n\n", ids);
                    }
                    else
                    {
                        Console.Write($"=%s \n\n", ids);
                    }
                }
                else if (!strcmp(buff, "clear_board"))
                {
                    passed = 0;
                    memset(board, 0, 19 * 19 * sizeof(float));
                    Console.Write($"=%s \n\n", ids);
                }
                else if (!strcmp(buff, "komi"))
                {
                    float komi = 0;
                    scanf("%f", &komi);
                    Console.Write($"=%s \n\n", ids);
                }
                else if (!strcmp(buff, "play"))
                {
                    char color[256];
                    scanf("%s ", color);
                    char c;
                    int r;
                    int count = scanf("%c%d", &c, &r);
                    int player = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
                    if (c == 'p' && count < 2)
                    {
                        passed = 1;
                        Console.Write($"=%s \n\n", ids);
                        string line = fgetl(stdin);
                        free(line);
                        fflush(stdout);
                        fflush(stderr);
                        continue;
                    }
                    else
                    {
                        passed = 0;
                    }
                    if (c >= 'A' && c <= 'Z') c = c - 'A';
                    if (c >= 'a' && c <= 'z') c = c - 'a';
                    if (c >= 8) --c;
                    r = 19 - r;
                    Console.Error.Write($"move: %d %d\n", r, c);

                    string swap = two;
                    two = one;
                    one = swap;
                    move_go(board, player, r, c);
                    board_to_string(one, board);

                    Console.Write($"=%s \n\n", ids);
                    print_board(board, 1, 0);
                }
                else if (!strcmp(buff, "genmove"))
                {
                    char color[256];
                    scanf("%s", color);
                    int player = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;

                    int index = generate_move(net, player, board, multi, .1, .7, two, 1);
                    if (passed || index < 0)
                    {
                        Console.Write($"=%s pass\n\n", ids);
                        passed = 0;
                    }
                    else
                    {
                        int row = index / 19;
                        int col = index % 19;

                        string swap = two;
                        two = one;
                        one = swap;

                        move_go(board, player, row, col);
                        board_to_string(one, board);
                        row = 19 - row;
                        if (col >= 8) ++col;
                        Console.Write($"=%s %c%d\n\n", ids, 'A' + col, row);
                        print_board(board, 1, 0);
                    }

                }
                else if (!strcmp(buff, "p"))
                {
                    //print_board(board, 1, 0);
                }
                else if (!strcmp(buff, "final_status_list"))
                {
                    char type[256];
                    scanf("%s", type);
                    Console.Error.Write($"final_status\n");
                    string line = fgetl(stdin);
                    free(line);
                    if (type[0] == 'd' || type[0] == 'D')
                    {
                        FILE* f = fopen("game.txt", "w");
                        int i, j;
                        int count = 2;
                        fprintf(f, "boardsize 19\n");
                        fprintf(f, "clear_board\n");
                        for (j = 0; j < 19; ++j)
                        {
                            for (i = 0; i < 19; ++i)
                            {
                                if (board[j * 19 + i] == 1) fprintf(f, "play black %c%d\n", 'A' + i + (i >= 8), 19 - j);
                                if (board[j * 19 + i] == -1) fprintf(f, "play white %c%d\n", 'A' + i + (i >= 8), 19 - j);
                                if (board[j * 19 + i]) ++count;
                            }
                        }
                        fprintf(f, "final_status_list dead\n");
                        fclose(f);
                        FILE* p = popen("./gnugo --mode gtp < game.txt", "r");
                        for (i = 0; i < count; ++i)
                        {
                            free(fgetl(p));
                            free(fgetl(p));
                        }
                        string l = 0;
                        while ((l = fgetl(p)))
                        {
                            Console.Write($"%s\n", l);
                            free(l);
                        }
                    }
                    else
                    {
                        Console.Write($"?%s unknown command\n\n", ids);
                    }
                }
                else
                {
                    string line = fgetl(stdin);
                    free(line);
                    Console.Write($"?%s unknown command\n\n", ids);
                }
                fflush(stdout);
                fflush(stderr);
            }
        }

        void test_go(string cfg, string weights, int multi)
        {
            Network net = Parser.parse_network_cfg(cfg);
            if (weights)
            {
                Parser.load_weights(net, weights);
            }
            
            set_batch_network(net, 1);
            float[] board = calloc(19 * 19, sizeof(float));
            float[] move = calloc(19 * 19, sizeof(float));
            int color = 1;
            while (1)
            {
                float[] output = network_predict(net, board);
                Blas.Copy_cpu(19 * 19, output, 1, move, 1);
                int i;
                if (multi)
                {
                    Image bim = float_to_image(19, 19, 1, board);
                    for (i = 1; i < 8; ++i)
                    {
                        rotate_image_cw(bim, i);
                        if (i >= 4) flip_image(bim);

                        float[] output = network_predict(net, board);
                        Image oim = float_to_image(19, 19, 1, output);

                        if (i >= 4) flip_image(oim);
                        rotate_image_cw(oim, -i);

                        axpy_cpu(19 * 19, 1, output, 1, move, 1);

                        if (i >= 4) flip_image(bim);
                        rotate_image_cw(bim, -i);
                    }
                    Blas.Scal_cpu(19 * 19, 1.0f / 8., move, 1);
                }
                for (i = 0; i < 19 * 19; ++i)
                {
                    if (board[i]) move[i] = 0;
                }

                int indexes[nind];
                int row, col;
                top_k(move, 19 * 19, nind, indexes);
                print_board(board, color, indexes);
                for (i = 0; i < nind; ++i)
                {
                    int index = indexes[i];
                    row = index / 19;
                    col = index % 19;
                    Console.Write($"%d: %c %d, %.2f%%\n", i + 1, col + 'A' + 1 * (col > 7 && noi), (inverted) ? 19 - row : row + 1, move[index] * 100);
                }
                //if(color == 1) Console.Write($"\u25EF Enter move: ");
                //else Console.Write($"\u25C9 Enter move: ");
                if (color == 1) Console.Write($"X Enter move: ");
                else Console.Write($"O Enter move: ");

                char c;
                string line = fgetl(stdin);
                int picked = 1;
                int dnum = sscanf(line, "%d", &picked);
                int cnum = sscanf(line, "%c", &c);
                if (strlen(line) == 0 || dnum)
                {
                    --picked;
                    if (picked < nind)
                    {
                        int index = indexes[picked];
                        row = index / 19;
                        col = index % 19;
                        board[row * 19 + col] = 1;
                    }
                }
                else if (cnum)
                {
                    if (c <= 'T' && c >= 'A')
                    {
                        int num = sscanf(line, "%c %d", &c, &row);
                        row = (inverted) ? 19 - row : row - 1;
                        col = c - 'A';
                        if (col > 7 && noi) col -= 1;
                        if (num == 2) board[row * 19 + col] = 1;
                    }
                    else if (c == 'p')
                    {
                        // Pass
                    }
                    else if (c == 'b' || c == 'w')
                    {
                        char g;
                        int num = sscanf(line, "%c %c %d", &g, &c, &row);
                        row = (inverted) ? 19 - row : row - 1;
                        col = c - 'A';
                        if (col > 7 && noi) col -= 1;
                        if (num == 3) board[row * 19 + col] = (g == 'b') ? color : -color;
                    }
                    else if (c == 'c')
                    {
                        char g;
                        int num = sscanf(line, "%c %c %d", &g, &c, &row);
                        row = (inverted) ? 19 - row : row - 1;
                        col = c - 'A';
                        if (col > 7 && noi) col -= 1;
                        if (num == 3) board[row * 19 + col] = 0;
                    }
                }
                free(line);
                flip_board(board);
                color = -color;
            }
        }

        float score_game(float[] board)
        {
            FILE* f = fopen("game.txt", "w");
            int i, j;
            int count = 3;
            fprintf(f, "komi 6.5\n");
            fprintf(f, "boardsize 19\n");
            fprintf(f, "clear_board\n");
            for (j = 0; j < 19; ++j)
            {
                for (i = 0; i < 19; ++i)
                {
                    if (board[j * 19 + i] == 1) fprintf(f, "play black %c%d\n", 'A' + i + (i >= 8), 19 - j);
                    if (board[j * 19 + i] == -1) fprintf(f, "play white %c%d\n", 'A' + i + (i >= 8), 19 - j);
                    if (board[j * 19 + i]) ++count;
                }
            }
            fprintf(f, "final_score\n");
            fclose(f);
            FILE* p = popen("./gnugo --mode gtp < game.txt", "r");
            for (i = 0; i < count; ++i)
            {
                free(fgetl(p));
                free(fgetl(p));
            }
            string l = 0;
            float score = 0;
            char player = 0;
            while ((l = fgetl(p)))
            {
                Console.Error.Write($"%s  \t", l);
                int n = sscanf(l, "= %c+%f", &player, &score);
                free(l);
                if (n == 2) break;
            }
            if (player == 'W') score = -score;
            pclose(p);
            return score;
        }

        void self_go(string filename, string weightfile, string f2, string w2, int multi)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }

            Network net2 = net;
            if (f2)
            {
                net2 = Parser.parse_network_cfg(f2);
                if (w2)
                {
                    Parser.load_weights(&net2, w2);
                }
            }
            
            //char boards[300][93];
            char boards[300, 93];
            int count = 0;
            set_batch_network(net, 1);
            set_batch_network(&net2, 1);
            float[] board = calloc(19 * 19, sizeof(float));
            string one = calloc(91, sizeof(char));
            string two = calloc(91, sizeof(char));
            int done = 0;
            int player = 1;
            int p1 = 0;
            int p2 = 0;
            int total = 0;
            while (1)
            {
                if (done || count >= 300)
                {
                    float score = score_game(board);
                    int i = (score > 0) ? 0 : 1;
                    if ((score > 0) == (total % 2 == 0)) ++p1;
                    else ++p2;
                    ++total;
                    Console.Error.Write($"Total: %d, Player 1: %f, Player 2: %f\n", total, (float)p1 / total, (float)p2 / total);
                    int j;
                    for (; i < count; i += 2)
                    {
                        for (j = 0; j < 93; ++j)
                        {
                            Console.Write($"%c", boards[i][j]);
                        }
                        Console.Write($"\n");
                    }
                    memset(board, 0, 19 * 19 * sizeof(float));
                    player = 1;
                    done = 0;
                    count = 0;
                    fflush(stdout);
                    fflush(stderr);
                }
                //print_board(board, 1, 0);
                //sleep(1);
                Network use = ((total % 2 == 0) == (player == 1)) ? net : net2;
                int index = generate_move(use, player, board, multi, .1, .7, two, 0);
                if (index < 0)
                {
                    done = 1;
                    continue;
                }
                int row = index / 19;
                int col = index % 19;

                string swap = two;
                two = one;
                one = swap;

                if (player < 0) flip_board(board);
                boards[count][0] = row;
                boards[count][1] = col;
                board_to_string(boards[count] + 2, board);
                if (player < 0) flip_board(board);
                ++count;

                move_go(board, player, row, col);
                board_to_string(one, board);

                player = -player;
            }
        }

        public static void run_go(List<string> args)
        {
            //boards_go();
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            string c2 = (args.Count > 5) ? args[5] : 0;
            string w2 = (args.Count > 6) ? args[6] : 0;
            int multi = find_arg(args.Count, args, "-multi");
            if (0 == strcmp(args[2], "train")) train_go(cfg, weights);
            else if (0 == strcmp(args[2], "valid")) valid_go(cfg, weights, multi);
            else if (0 == strcmp(args[2], "self")) self_go(cfg, weights, c2, w2, multi);
            else if (0 == strcmp(args[2], "test")) test_go(cfg, weights, multi);
            else if (0 == strcmp(args[2], "engine")) engine_go(cfg, weights, multi);
        }



        #endregion

        #region RnnFile

        int[] read_tokenized_data(string filename, size_t* read)
        {
            size_t size = 512;
            size_t count = 0;
            FILE* fp = fopen(filename, "r");
            int[] d = calloc(size, sizeof(int));
            int n, one;
            one = fscanf(fp, "%d", &n);
            while (one == 1)
            {
                ++count;
                if (count > size)
                {
                    size = size * 2;
                    d = realloc(d, size * sizeof(int));
                }
                d[count - 1] = n;
                one = fscanf(fp, "%d", &n);
            }
            fclose(fp);
            d = realloc(d, count * sizeof(int));
            *read = count;
            return d;
        }

        string[] read_tokens(string filename, size_t* read)
        {
            size_t size = 512;
            size_t count = 0;
            FILE* fp = fopen(filename, "r");
            string[] d = calloc(size, sizeof(string));
            string line;
            while ((line = fgetl(fp)) != 0)
            {
                ++count;
                if (count > size)
                {
                    size = size * 2;
                    d = realloc(d, size * sizeof(string));
                }
                d[count - 1] = line;
            }
            fclose(fp);
            d = realloc(d, count * sizeof(string));
            *read = count;
            return d;
        }

        float_pair get_rnn_token_data(int[] tokens, size_t* offsets, int characters, size_t len, int batch, int steps)
        {
            float[] x = calloc(batch * steps * characters, sizeof(float));
            float[] y = calloc(batch * steps * characters, sizeof(float));
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
                        error("Bad char");
                    }
                }
            }
            float_pair p;
            p.x = x;
            p.y = y;
            return p;
        }

        float_pair get_rnn_data(unsigned string text, size_t* offsets, int characters, size_t len, int batch, int steps)
        {
            float[] x = calloc(batch * steps * characters, sizeof(float));
            float[] y = calloc(batch * steps * characters, sizeof(float));
            int i, j;
            for (i = 0; i < batch; ++i)
            {
                for (j = 0; j < steps; ++j)
                {
                    unsigned char curr = text[(offsets[i]) % len];
                    unsigned char next = text[(offsets[i] + 1) % len];

                    x[(j * batch + i) * characters + curr] = 1;
                    y[(j * batch + i) * characters + next] = 1;

                    offsets[i] = (offsets[i] + 1) % len;

                    if (curr > 255 || curr <= 0 || next > 255 || next <= 0)
                    {
                        error("Bad char");
                    }
                }
            }
            float_pair p;
            p.x = x;
            p.y = y;
            return p;
        }

        void reset_rnn_state(Network net, int b)
        {
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.state_gpu)
                {
                    fill_ongpu(l.Outputs, 0, l.state_gpu + l.Outputs * b, 1);
                }
            }
        }

        void train_char_rnn(string cfgfile, string weightfile, string filename, int clear, int tokenized)
        {
            
            unsigned string text = 0;
            int[] tokens = 0;
            size_t size;
            if (tokenized)
            {
                tokens = read_tokenized_data(filename, &size);
            }
            else
            {
                FILE* fp = fopen(filename, "rb");

                fseek(fp, 0, SEEK_END);
                size = ftell(fp);
                fseek(fp, 0, SEEK_SET);

                text = calloc(size + 1, sizeof(char));
                fread(text, 1, size, fp);
                fclose(fp);
            }

            string backup_directory = "/home/pjreddie/backup/";
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }

            int inputs = get_network_input_size(net);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int batch = net.Batch;
            int steps = net.time_steps;
            if (clear) net.Seen = 0;
            int i = (net.Seen) / net.Batch;

            int streams = batch / steps;
            size_t* offsets = calloc(streams, sizeof(size_t));
            int j;
            for (j = 0; j < streams; ++j)
            {
                offsets[j] = rand_size_t() % size;
            }

            clock_t time;
            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                float_pair p;
                if (tokenized)
                {
                    p = get_rnn_token_data(tokens, offsets, inputs, size, streams, steps);
                }
                else
                {
                    p = get_rnn_data(text, offsets, inputs, size, streams, steps);
                }

                float loss = train_network_datum(net, p.x, p.y) / (batch);
                free(p.x);
                free(p.y);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                int chars = get_current_batch(net) * batch;
                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), (float)chars / size);

                for (j = 0; j < streams; ++j)
                {
                    //Console.Write($"%d\n", j);
                    if (rand() % 10 == 0)
                    {
                        //Console.Error.Write($"Reset\n");
                        offsets[j] = rand_size_t() % size;
                        reset_rnn_state(net, j);
                    }
                }

                if (i % 1000 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                if (i % 10 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void print_symbol(int n, string[] tokens)
        {
            if (tokens)
            {
                Console.Write($"%s ", tokens[n]);
            }
            else
            {
                Console.Write($"%c", n);
            }
        }

        void test_char_rnn(string cfgfile, string weightfile, int num, string seed, float temp, int rseed, string token_file)
        {
            string[] tokens = 0;
            if (token_file)
            {
                size_t n;
                tokens = read_tokens(token_file, &n);
            }

            srand(rseed);
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].temperature = temp;
            int c = 0;
            int len = strlen(seed);
            float[] input = calloc(inputs, sizeof(float));


            for (i = 0; i < len - 1; ++i)
            {
                c = seed[i];
                input[c] = 1;
                network_predict(net, input);
                input[c] = 0;
                print_symbol(c, tokens);
            }
            if (len) c = seed[len - 1];
            print_symbol(c, tokens);
            for (i = 0; i < num; ++i)
            {
                input[c] = 1;
                float[] outf = network_predict(net, input);
                input[c] = 0;
                for (j = 32; j < 127; ++j)
                {
                    //Console.Write($"%d %c %f\n",j, j, outf[j]);
                }
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                c = sample_array(outf, inputs);
                print_symbol(c, tokens);
            }
            Console.Write($"\n");
        }

        void test_tactic_rnn(string cfgfile, string weightfile, int num, float temp, int rseed, string token_file)
        {
            string[] tokens = 0;
            if (token_file)
            {
                size_t n;
                tokens = read_tokens(token_file, &n);
            }

            srand(rseed);
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].temperature = temp;
            int c = 0;
            float[] input = calloc(inputs, sizeof(float));
            float[] outf = 0;

            while ((c = getc(stdin)) != EOF)
            {
                input[c] = 1;
                outf = network_predict(net, input);
                input[c] = 0;
            }
            for (i = 0; i < num; ++i)
            {
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                int next = sample_array(outf, inputs);
                if (c == '.' && next == '\n') break;
                c = next;
                print_symbol(c, tokens);

                input[c] = 1;
                outf = network_predict(net, input);
                input[c] = 0;
            }
            Console.Write($"\n");
        }

        void valid_tactic_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int count = 0;
            int words = 1;
            int c;
            int len = strlen(seed);
            float[] input = calloc(inputs, sizeof(float));
            int i;
            for (i = 0; i < len; ++i)
            {
                c = seed[i];
                input[(int)c] = 1;
                network_predict(net, input);
                input[(int)c] = 0;
            }
            float sum = 0;
            c = getc(stdin);
            float log2 = log(2);
            int ini = 0;
            while (c != EOF)
            {
                int next = getc(stdin);
                if (next == EOF) break;
                if (next < 0 || next >= 255) error("Out of range character");

                input[c] = 1;
                float[] outf = network_predict(net, input);
                input[c] = 0;

                if (c == '.' && next == '\n') ini = 0;
                if (!ini)
                {
                    if (c == '>' && next == '>')
                    {
                        ini = 1;
                        ++words;
                    }
                    c = next;
                    continue;
                }
                ++count;
                sum += log(outf[next]) / log2;
                c = next;
                Console.Write($"%d %d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, words, pow(2, -sum / count), pow(2, -sum / words));
            }
        }

        void valid_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int count = 0;
            int words = 1;
            int c;
            int len = strlen(seed);
            float[] input = calloc(inputs, sizeof(float));
            int i;
            for (i = 0; i < len; ++i)
            {
                c = seed[i];
                input[(int)c] = 1;
                network_predict(net, input);
                input[(int)c] = 0;
            }
            float sum = 0;
            c = getc(stdin);
            float log2 = log(2);
            while (c != EOF)
            {
                int next = getc(stdin);
                if (next == EOF) break;
                if (next < 0 || next >= 255) error("Out of range character");
                ++count;
                if (next == ' ' || next == '\n' || next == '\t') ++words;
                input[c] = 1;
                float[] outf = network_predict(net, input);
                input[c] = 0;
                sum += log(outf[next]) / log2;
                c = next;
                Console.Write($"%d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, pow(2, -sum / count), pow(2, -sum / words));
            }
        }

        void vec_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int c;
            int seed_len = strlen(seed);
            float[] input = calloc(inputs, sizeof(float));
            int i;
            string line;
            while ((line = fgetl(stdin)) != 0)
            {
                reset_rnn_state(net, 0);
                for (i = 0; i < seed_len; ++i)
                {
                    c = seed[i];
                    input[(int)c] = 1;
                    network_predict(net, input);
                    input[(int)c] = 0;
                }
                strip(line);
                int str_len = strlen(line);
                for (i = 0; i < str_len; ++i)
                {
                    c = line[i];
                    input[(int)c] = 1;
                    network_predict(net, input);
                    input[(int)c] = 0;
                }
                c = ' ';
                input[(int)c] = 1;
                network_predict(net, input);
                input[(int)c] = 0;

                Layer l = net.Layers[0];
                cuda_pull_array(l.output_gpu, l.output, l.Outputs);
                Console.Write($"%s", line);
                for (i = 0; i < l.Outputs; ++i)
                {
                    Console.Write($",%g", l.output[i]);
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
            string filename = find_char_arg(args.Count, args, "-file", "Data.Data/shakespeare.txt");
            string seed = find_char_arg(args.Count, args, "-seed", "\n\n");
            int len = Utils.find_int_arg(args.Count, args, "-len", 1000);
            float temp = find_float_arg(args.Count, args, "-temp", .7);
            int rseed = Utils.find_int_arg(args.Count, args, "-srand", time(0));
            int clear = find_arg(args.Count, args, "-clear");
            int tokenized = find_arg(args.Count, args, "-tokenized");
            string tokens = find_char_arg(args.Count, args, "-tokens", 0);

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            if (0 == strcmp(args[2], "train")) train_char_rnn(cfg, weights, filename, clear, tokenized);
            else if (0 == strcmp(args[2], "valid")) valid_char_rnn(cfg, weights, seed);
            else if (0 == strcmp(args[2], "validtactic")) valid_tactic_rnn(cfg, weights, seed);
            else if (0 == strcmp(args[2], "vec")) vec_char_rnn(cfg, weights, seed);
            else if (0 == strcmp(args[2], "generate")) test_char_rnn(cfg, weights, len, seed, temp, rseed, tokens);
            else if (0 == strcmp(args[2], "generatetactic")) test_tactic_rnn(cfg, weights, len, temp, rseed, tokens);
        }

        #endregion

        #region RnnVidFile


        float_pair get_rnn_vid_data(Network net, string[] files, int n, int batch, int steps)
        {
            int b;
            assert(net.Batch == steps + 1);
            Image out_im = get_network_image(net);
            int output_size = out_im.W * out_im.H * out_im.C;
            Console.Write($"%d %d %d\n", out_im.W, out_im.H, out_im.C);
            float[] feats = calloc(net.Batch * batch * output_size, sizeof(float));
            for (b = 0; b < batch; ++b)
            {
                int input_size = net.W * net.H * net.C;
                float[] input = calloc(input_size * net.Batch, sizeof(float));
                string filename = files[rand() % n];
                CvCapture* cap = cvCaptureFromFile(filename);
                int frames = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
                int index = rand() % (frames - steps - 2);
                if (frames < (steps + 4))
                {
                    --b;
                    free(input);
                    continue;
                }

                Console.Write($"frames: %d, index: %d\n", frames, index);
                cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, index);

                int i;
                for (i = 0; i < net.Batch; ++i)
                {
                    IplImage* src = cvQueryFrame(cap);
                    Image im = ipl_to_image(src);
                    rgbgr_image(im);
                    Image re = resize_image(im, net.W, net.H);
                    //show_image(re, "loaded");
                    //cvWaitKey(10);
                    memcpy(input + i * input_size, re.Data.Data, input_size * sizeof(float));
                    free_image(im);
                    free_image(re);
                }
                float[] output = network_predict(net, input);

                free(input);

                for (i = 0; i < net.Batch; ++i)
                {
                    memcpy(feats + (b + i * batch) * output_size, output + i * output_size, output_size * sizeof(float));
                }

                cvReleaseCapture(&cap);
            }

            //Console.Write($"%d %d %d\n", out_im.W, out_im.H, out_im.C);
            float_pair p = { 0 };
            p.x = feats;
            p.y = feats + output_size * batch; //+ out_im.W*out_im.H*out_im.C;

            return p;
        }


        void train_vid_rnn(string cfgfile, string weightfile)
        {
            string train_videos = "Data.Data/vid/train.txt";
            string backup_directory = "/home/pjreddie/backup/";
            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;

            list* plist = get_paths(train_videos);
            int N = plist.Size;
            string[] paths = (string[])list_to_array(plist);
            clock_t time;
            int steps = net.time_steps;
            int batch = net.Batch / net.time_steps;

            Network extractor = Parser.parse_network_cfg("cfg/extractor.cfg");
            Parser.load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                float_pair p = get_rnn_vid_data(extractor, paths, N, batch, steps);

                float loss = train_network_datum(net, p.x, p.y) / (net.Batch);


                free(p.x);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time));
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                if (i % 10 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }


        Image save_reconstruction(Network net, Image* init, float[] feat, string name, int i)
        {
            Image recon;
            if (init)
            {
                recon = copy_image(*init);
            }
            else
            {
                recon = make_random_image(net.W, net.H, 3);
            }

            Image update = new Image(net.W, net.H, 3);
            reconstruct_picture(net, feat, recon, update, .01, .9, .1, 2, 50);
            char buff[256];
            sprintf(buff, "%s%d", name, i);
            save_image(recon, buff);
            free_image(update);
            return recon;
        }

        void generate_vid_rnn(string cfgfile, string weightfile)
        {
            Network extractor = Parser.parse_network_cfg("cfg/extractor.recon.cfg");
            Parser.load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(&extractor, 1);
            set_batch_network(net, 1);

            int i;
            CvCapture* cap = cvCaptureFromFile("/extra/vid/ILSVRC2015/Data.Data/VID/snippets/val/ILSVRC2015_val_00007030.mp4");
            float[] feat;
            float[] next;
            next = NULL;
            Image last;
            for (i = 0; i < 25; ++i)
            {
                Image im = get_image_from_stream(cap);
                Image re = resize_image(im, extractor.W, extractor.H);
                feat = network_predict(extractor, re.Data.Data);
                if (i > 0)
                {
                    Console.Write($"%f %f\n", mean_array(feat, 14 * 14 * 512), variance_array(feat, 14 * 14 * 512));
                    Console.Write($"%f %f\n", mean_array(next, 14 * 14 * 512), variance_array(next, 14 * 14 * 512));
                    Console.Write($"%f\n", mse_array(feat, 14 * 14 * 512));
                    axpy_cpu(14 * 14 * 512, -1, feat, 1, next, 1);
                    Console.Write($"%f\n", mse_array(next, 14 * 14 * 512));
                }
                next = network_predict(net, feat);

                free_image(im);

                free_image(save_reconstruction(extractor, 0, feat, "feat", i));
                free_image(save_reconstruction(extractor, 0, next, "next", i));
                if (i == 24) last = copy_image(re);
                free_image(re);
            }
            for (i = 0; i < 30; ++i)
            {
                next = network_predict(net, next);
                Image newi = save_reconstruction(extractor, &last, next, "new", i);
                free_image(last);
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
            string weights = (args.Count > 4) ? args[4] : 0;
            //char *filename = (args.Count > 5) ? args[5]: 0;
            if (0 == strcmp(args[2], "train")) train_vid_rnn(cfg, weights);
            else if (0 == strcmp(args[2], "generate")) generate_vid_rnn(cfg, weights);
        }

        #endregion

        #region CocoFile

        static string[] coco_classes = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        static int[] coco_ids = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };

        void train_coco(string cfgfile, string weightfile)
        {
            //char *train_images = "/home/pjreddie/Data.Data/voc/test/train.txt";
            //char *train_images = "/home/pjreddie/Data.Data/coco/train.txt";
            string train_images = "Data.Data/coco.trainval.txt";
            //char *train_images = "Data.Data/bags.train.list";
            string backup_directory = "/home/pjreddie/backup/";
            
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avg_loss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            int i = net.Seen / imgs;
            Data.Data train, buffer;


            Layer l = net.Layers[net.N - 1];

            int side = l.Side;
            int classes = l.Classes;
            float jitter = l.Jitter;

            list* plist = get_paths(train_images);
            //int N = plist.Size;
            string[] paths = (string[])list_to_array(plist);

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.N = imgs;
            args.m = plist.Size;
            args.Classes = classes;
            args.Jitter = jitter;
            args.num_boxes = side;
            args.d = &buffer;
            args.LayerType = REGION_DATA;

            args.angle = net.angle;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;

            pthread_t load_thread = load_data_in_thread(args);
            clock_t time;
            //while(i*imgs < N*120){
            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data_in_thread(args);

                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.Weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void print_cocos(FILE* fp, int image_id, box* boxes, float[]* probs, int num_boxes, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < num_boxes; ++i)
            {
                float xmin = boxes[i].x - boxes[i].W / 2.;
                float xmax = boxes[i].x + boxes[i].W / 2.;
                float ymin = boxes[i].y - boxes[i].H / 2.;
                float ymax = boxes[i].y + boxes[i].H / 2.;

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
                    if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
                }
            }
        }

        int get_coco_image_id(string filename)
        {
            string p = strrchr(filename, '_');
            return int.Parse(p + 1);
        }

        void validate_coco(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            

            string basec = "results/";
            list* plist = get_paths("Data.Data/coco_val_5k.list");
            //list *plist = get_paths("/home/pjreddie/Data.Data/people-art/test.txt");
            //list *plist = get_paths("/home/pjreddie/Data.Data/voc/test/2007_test.txt");
            string[] paths = (string[])list_to_array(plist);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            int side = l.Side;

            int j;
            char buff[1024];
            snprintf(buff, 1024, "%s/coco_results.json", basec);
            FILE* fp = fopen(buff, "w");
            fprintf(fp, "[\n");

            box* boxes = (box*)calloc(side * side * l.N, sizeof(box));
            float[]*probs = (float[] *)calloc(side * side * l.N, sizeof(float[]));
            for (j = 0; j < side * side * l.N; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.Size;
            int i = 0;
            int t;

            float thresh = .01;
            int nms = 1;
            float iou_thresh = .5;

            int nthreads = 8;
            Image* val = (Image*)calloc(nthreads, sizeof(Image));
            Image* val_resized = (Image*)calloc(nthreads, sizeof(Image));
            Image* buf = (Image*)calloc(nthreads, sizeof(Image));
            Image* buf_resized = (Image*)calloc(nthreads, sizeof(Image));
            pthread_t* thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.LayerType = IMAGE_DATA;

            for (t = 0; t < nthreads; ++t)
            {
                args.path = paths[i + t];
                args.im = &buf[t];
                args.resized = &buf_resized[t];
                thr[t] = load_data_in_thread(args);
            }
            time_t start = time(0);
            for (i = nthreads; i < m + nthreads; i += nthreads)
            {
                Console.Error.Write($"%d\n", i);
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    pthread_join(thr[t], 0);
                    val[t] = buf[t];
                    val_resized[t] = buf_resized[t];
                }
                for (t = 0; t < nthreads && i + t < m; ++t)
                {
                    args.path = paths[i + t];
                    args.im = &buf[t];
                    args.resized = &buf_resized[t];
                    thr[t] = load_data_in_thread(args);
                }
                for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                {
                    string path = paths[i + t - nthreads];
                    int image_id = get_coco_image_id(path);
                    float[] X = val_resized[t].Data.Data;
                    network_predict(net, X);
                    int w = val[t].W;
                    int h = val[t].H;
                    get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
                    if (nms) do_nms_sort(boxes, probs, side * side * l.N, classes, iou_thresh);
                    print_cocos(fp, image_id, boxes, probs, side * side * l.N, classes, w, h);
                    free_image(val[t]);
                    free_image(val_resized[t]);
                }
            }
            fseek(fp, -2, SEEK_CUR);
            fprintf(fp, "\n]\n");
            fclose(fp);

            Console.Error.Write($"Total Detection Time: %f Seconds\n", (double)(time(0) - start));
        }

        void validate_coco_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            

            string basec = "results/comp4_det_test_";
            list* plist = get_paths("/home/pjreddie/Data.Data/voc/test/2007_test.txt");
            string[] paths = (string[])list_to_array(plist);

            Layer l = net.Layers[net.N - 1];
            int classes = l.Classes;
            int side = l.Side;

            int j, k;
            FILE** fps = (FILE**)calloc(classes, sizeof(FILE*));
            for (j = 0; j < classes; ++j)
            {
                char buff[1024];
                snprintf(buff, 1024, "%s%s.txt", basec, coco_classes[j]);
                fps[j] = fopen(buff, "w");
            }
            box* boxes = (box*)calloc(side * side * l.N, sizeof(box));
            float[]*probs = (float[] *)calloc(side * side * l.N, sizeof(float[]));
            for (j = 0; j < side * side * l.N; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.Size;
            int i = 0;

            float thresh = .001;
            int nms = 0;
            float iou_thresh = .5;
            float nms_thresh = .5;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avg_iou = 0;

            for (i = 0; i < m; ++i)
            {
                string path = paths[i];
                Image orig = load_image_color(path, 0, 0);
                Image sized = resize_image(orig, net.W, net.H);
                string id = basecfg(path);
                network_predict(net, sized.Data.Data);
                get_detection_boxes(l, 1, 1, thresh, probs, boxes, 1);
                if (nms) do_nms(boxes, probs, side * side * l.N, 1, nms_thresh);

                char labelpath[4096];
                find_replace(path, "images", "labels", labelpath);
                find_replace(labelpath, "JPEGImages", "labels", labelpath);
                find_replace(labelpath, ".jpg", ".txt", labelpath);
                find_replace(labelpath, ".JPEG", ".txt", labelpath);

                int num_labels = 0;
                box_label* truth = read_boxes(labelpath, &num_labels);
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
                    box t = { truth[j].x, truth[j].y, truth[j].W, truth[j].H };
                    float best_iou = 0;
                    for (k = 0; k < side * side * l.N; ++k)
                    {
                        float iou = box_iou(boxes[k], t);
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

                Console.Error.Write($"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.* correct / total);
                free(id);
                free_image(orig);
                free_image(sized);
            }
        }

        void test_coco(string cfgfile, string weightfile, string filename, float thresh)
        {
            Image** alphabet = load_alphabet();
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            detection_layer l = net.Layers[net.N - 1];
            set_batch_network(net, 1);
            srand(2222222);
            float nms = .4;
            clock_t time;
            char buff[256];
            string input = buff;
            int j;
            box* boxes = (box*)calloc(l.Side * l.Side * l.N, sizeof(box));
            float[]*probs = (float[] *)calloc(l.Side * l.Side * l.N, sizeof(float[]));
            for (j = 0; j < l.Side * l.Side * l.N; ++j) probs[j] = (float[])calloc(l.Classes, sizeof(float[]));
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                Image sized = resize_image(im, net.W, net.H);
                float[] X = sized.Data.Data;
                time = clock();
                network_predict(net, X);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
                if (nms) do_nms_sort(boxes, probs, l.Side * l.Side * l.N, l.Classes, nms);
                draw_detections(im, l.Side * l.Side * l.N, thresh, boxes, probs, coco_classes, alphabet, 80);
                save_image(im, "prediction");
                show_image(im, "predictions");
                free_image(im);
                free_image(sized);
                cvWaitKey(0);
                cvDestroyAllWindows();
                if (filename) break;
            }
        }

        public static void run_coco(List<string> args)
        {
            string prefix = find_char_arg(args.Count, args, "-prefix", 0);
            float thresh = find_float_arg(args.Count, args, "-thresh", .2);
            int cam_index = Utils.find_int_arg(args.Count, args, "-c", 0);
            int frame_skip = Utils.find_int_arg(args.Count, args, "-s", 0);

            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "test")) test_coco(cfg, weights, filename, thresh);
            else if (0 == strcmp(args[2], "train")) train_coco(cfg, weights);
            else if (0 == strcmp(args[2], "valid")) validate_coco(cfg, weights);
            else if (0 == strcmp(args[2], "recall")) validate_coco_recall(cfg, weights);
            else if (0 == strcmp(args[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, coco_classes, 80, frame_skip, prefix);
        }

        #endregion

        #region ClassifierFile


        float[] get_regression_values(string[] labels, int n)
        {
            float[] v = calloc(n, sizeof(float));
            int i;
            for (i = 0; i < n; ++i)
            {
                string p = strchr(labels[i], ' ');
                *p = 0;
                v[i] = float.Parse(p + 1);
            }
            return v;
        }

        void train_classifier(string datacfg, string cfgfile, string weightfile, int[] gpus, int ngpus, int clear)
        {
            int i;

            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Console.Write($"%d\n", ngpus);
            Network* nets = calloc(ngpus, sizeof(Network));

            
            int seed = rand();
            for (i = 0; i < ngpus; ++i)
            {
                srand(seed);
# ifdef GPU
                cuda_set_device(gpus[i]);
#endif
                nets[i] = Parser.parse_network_cfg(cfgfile);
                if (weightfile)
                {
                    Parser.load_weights(&nets[i], weightfile);
                }
                if (clear) *nets[i].Seen = 0;
                nets[i].LearningRate *= ngpus;
            }
            
            Network net = nets[0];

            int imgs = net.Batch * net.Subdivisions * ngpus;

            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            list* options = read_data_cfg(datacfg);

            string backup_directory = option_find_str(options, "backup", "/backup/");
            string label_list = option_find_str(options, "labels", "Data.Data/labels.list");
            string train_list = option_find_str(options, "train", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(train_list);
            string[] paths = (string[])list_to_array(plist);
            Console.Write($"%d\n", plist.Size);
            int N = plist.Size;
            clock_t time;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.threads = 32;
            args.hierarchy = net.hierarchy;

            args.min = net.min_crop;
            args.max = net.max_crop;
            args.angle = net.angle;
            args.aspect = net.aspect;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;
            args.Size = net.W;

            args.paths = paths;
            args.Classes = classes;
            args.N = imgs;
            args.m = N;
            args.labels = labels;
            args.LayerType = CLASSIFICATION_DATA;

            Data.Data train;
            Data.Data buffer;
            pthread_t load_thread;
            args.d = &buffer;
            load_thread = load_data(args);

            int epoch = (net.Seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                time = clock();

                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data(args);

                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();

                float loss = 0;
# ifdef GPU
                if (ngpus == 1)
                {
                    loss = train_network(net, train);
                }
                else
                {
                    loss = train_networks(nets, ngpus, train, 4);
                }
#else
                loss = train_network(net, train);
#endif
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), net.Seen);
                free_data(train);
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                }
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
            }
            char buff[256];
            sprintf(buff, "%s/%s.Weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free_ptrs((void**)labels, classes);
            free_ptrs((void**)paths, plist.Size);
            free_list(plist);
            free(basec);
        }


        void validate_classifier_crop(string datacfg, string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = option_find_str(options, "valid", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            clock_t time;
            float avg_acc = 0;
            float avg_topk = 0;
            int splits = m / 1000;
            int num = (i + 1) * m / splits - i * m / splits;

            Data.Data val, buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;

            args.paths = paths;
            args.Classes = classes;
            args.N = num;
            args.m = 0;
            args.labels = labels;
            args.d = &buffer;
            args.LayerType = OLD_CLASSIFICATION_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            for (i = 1; i <= splits; ++i)
            {
                time = clock();

                pthread_join(load_thread, 0);
                val = buffer;

                num = (i + 1) * m / splits - i * m / splits;
                string[] part = paths + (i * m / splits);
                if (i != splits)
                {
                    args.paths = part;
                    load_thread = load_data_in_thread(args);
                }
                Console.Write($"Loaded: %d images ini %lf seconds\n", val.X.rows, sec(clock() - time));

                time = clock();
                float[] acc = network_accuracies(net, val, topk);
                avg_acc += acc[0];
                avg_topk += acc[1];
                Console.Write($"%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc / i, topk, avg_topk / i, sec(clock() - time), val.X.rows);
                free_data(val);
            }
        }

        void validate_classifier_10(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            set_batch_network(net, 1);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = option_find_str(options, "valid", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = calloc(topk, sizeof(int));

            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (strstr(path, labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                int w = net.W;
                int h = net.H;
                int shift = 32;
                Image im = load_image_color(paths[i], w + shift, h + shift);
                Image images[10];
                images[0] = crop_image(im, -shift, -shift, w, h);
                images[1] = crop_image(im, shift, -shift, w, h);
                images[2] = crop_image(im, 0, 0, w, h);
                images[3] = crop_image(im, -shift, shift, w, h);
                images[4] = crop_image(im, shift, shift, w, h);
                flip_image(im);
                images[5] = crop_image(im, -shift, -shift, w, h);
                images[6] = crop_image(im, shift, -shift, w, h);
                images[7] = crop_image(im, 0, 0, w, h);
                images[8] = crop_image(im, -shift, shift, w, h);
                images[9] = crop_image(im, shift, shift, w, h);
                float[] pred = calloc(classes, sizeof(float));
                for (j = 0; j < 10; ++j)
                {
                    float[] p = network_predict(net, Data.Data.Data);
                    if (net.hierarchy) hierarchy_predictions(p, net.Outputs, net.hierarchy, 1);
                    axpy_cpu(classes, 1, p, 1, pred, 1);
                    free_image(images[j]);
                }
                free_image(im);
                top_k(pred, classes, topk, indexes);
                free(pred);
                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        void validate_classifier_full(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            set_batch_network(net, 1);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = option_find_str(options, "valid", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = calloc(topk, sizeof(int));

            int size = net.W;
            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (strstr(path, labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                Image im = load_image_color(paths[i], 0, 0);
                Image resized = resize_min(im, size);
                resize_network(net, resized.W, resized.H);
                float[] pred = network_predict(net, resized.Data.Data);
                if (net.hierarchy) hierarchy_predictions(pred, net.Outputs, net.hierarchy, 1);

                free_image(im);
                free_image(resized);
                top_k(pred, classes, topk, indexes);

                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }


        void validate_classifier_single(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "Data.Data/labels.list");
            string leaf_list = option_find_str(options, "leaves", 0);
            if (leaf_list) change_leaves(net.hierarchy, leaf_list);
            string valid_list = option_find_str(options, "valid", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = calloc(topk, sizeof(int));

            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (strstr(path, labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                Image im = load_image_color(paths[i], 0, 0);
                Image resized = resize_min(im, net.W);
                Image crop = crop_image(resized, (resized.W - net.W) / 2, (resized.H - net.H) / 2, net.W, net.H);
                //show_image(im, "orig");
                //show_image(crop, "cropped");
                //cvWaitKey(0);
                float[] pred = network_predict(net, crop.Data.Data);
                if (net.hierarchy) hierarchy_predictions(pred, net.Outputs, net.hierarchy, 1);

                if (resized.Data.Data != im.Data.Data) free_image(resized);
                free_image(im);
                free_image(crop);
                top_k(pred, classes, topk, indexes);

                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        void validate_classifier_multi(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            set_batch_network(net, 1);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "Data.Data/labels.list");
            string valid_list = option_find_str(options, "valid", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);
            int scales[] = { 224, 288, 320, 352, 384 };
            int nscales = sizeof(scales) / sizeof(scales[0]);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = calloc(topk, sizeof(int));

            for (i = 0; i < m; ++i)
            {
                int class2 = -1;
                string path = paths[i];
                for (j = 0; j < classes; ++j)
                {
                    if (strstr(path, labels[j]))
                    {
                        class2 = j;
                        break;
                    }
                }
                float[] pred = calloc(classes, sizeof(float));
                Image im = load_image_color(paths[i], 0, 0);
                for (j = 0; j < nscales; ++j)
                {
                    Image r = resize_min(im, scales[j]);
                    resize_network(net, r.W, r.H);
                    float[] p = network_predict(net, r.Data.Data);
                    if (net.hierarchy) hierarchy_predictions(p, net.Outputs, net.hierarchy, 1);
                    axpy_cpu(classes, 1, p, 1, pred, 1);
                    flip_image(r);
                    p = network_predict(net, r.Data.Data);
                    axpy_cpu(classes, 1, p, 1, pred, 1);
                    if (r.Data.Data != im.Data.Data) free_image(r);
                }
                free_image(im);
                top_k(pred, classes, topk, indexes);
                free(pred);
                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        void try_classifier(string datacfg, string cfgfile, string weightfile, string filename, int layer_num)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);

            list* options = read_data_cfg(datacfg);

            string name_list = option_find_str(options, "names", 0);
            if (!name_list) name_list = option_find_str(options, "labels", "Data.Data/labels.list");
            int top = option_find_int(options, "top", 1);

            int i = 0;
            string[] names = get_labels(name_list);
            clock_t time;
            int[] indexes = calloc(top, sizeof(int));
            char buff[256];
            string input = buff;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image orig = load_image_color(input, 0, 0);
                Image r = resize_min(orig, 256);
                Image im = crop_image(r, (r.W - 224 - 1) / 2 + 1, (r.H - 224 - 1) / 2 + 1, 224, 224);
                float mean[] = { 0.48263312050943, 0.45230225481413, 0.40099074308742 };
                float std[] = { 0.22590347483426, 0.22120921437787, 0.22103996251583 };
                float var[3];
                var[0] = std[0] * std[0];
                var[1] = std[1] * std[1];
                var[2] = std[2] * std[2];

                normalize_cpu(im.Data.Data, mean, var, 1, 3, im.W * im.H);

                float[] X = im.Data.Data;
                time = clock();
                float[] predictions = network_predict(net, X);

                Layer l = net.Layers[layer_num];
                for (i = 0; i < l.C; ++i)
                {
                    if (l.RollingMean) Console.Write($"%f %f %f\n", l.RollingMean[i], l.RollingVariance[i], l.Scales[i]);
                }
                cuda_pull_array(l.output_gpu, l.output, l.Outputs);
                for (i = 0; i < l.Outputs; ++i)
                {
                    Console.Write($"%f\n", l.output[i]);
                }

                top_predictions(net, top, indexes);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%s: %f\n", names[index], predictions[index]);
                }
                free_image(im);
                if (filename) break;
            }
        }

        public static void predict_classifier(string datacfg, string cfgfile, string weightfile, string filename, int top)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);

            list* options = read_data_cfg(datacfg);

            string name_list = option_find_str(options, "names", 0);
            if (!name_list) name_list = option_find_str(options, "labels", "Data.Data/labels.list");
            if (top == 0) top = option_find_int(options, "top", 1);

            int i = 0;
            string[] names = get_labels(name_list);
            clock_t time;
            int[] indexes = calloc(top, sizeof(int));
            char buff[256];
            string input = buff;
            int size = net.W;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                Image r = resize_min(im, size);
                resize_network(net, r.W, r.H);
                Console.Write($"%d %d\n", r.W, r.H);

                float[] X = r.Data.Data;
                time = clock();
                float[] predictions = network_predict(net, X);
                if (net.hierarchy) hierarchy_predictions(predictions, net.Outputs, net.hierarchy, 0);
                top_k(predictions, net.Outputs, top, indexes);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    if (net.hierarchy) Console.Write($"%d, %s: %f, parent: %s \n", index, names[index], predictions[index], (net.hierarchy.parent[index] >= 0) ? names[net.hierarchy.parent[index]] : "Root");
                    else Console.Write($"%s: %f\n", names[index], predictions[index]);
                }
                if (r.Data.Data != im.Data.Data) free_image(r);
                free_image(im);
                if (filename) break;
            }
        }


        void label_classifier(string datacfg, string filename, string weightfile)
        {
            int i;
            Network net = Parser.parse_network_cfg(filename);
            set_batch_network(net, 1);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "names", "Data.Data/labels.list");
            string test_list = option_find_str(options, "test", "Data.Data/train.list");
            int classes = option_find_int(options, "classes", 2);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(test_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            for (i = 0; i < m; ++i)
            {
                Image im = load_image_color(paths[i], 0, 0);
                Image resized = resize_min(im, net.W);
                Image crop = crop_image(resized, (resized.W - net.W) / 2, (resized.H - net.H) / 2, net.W, net.H);
                float[] pred = network_predict(net, crop.Data.Data);

                if (resized.Data.Data != im.Data.Data) free_image(resized);
                free_image(im);
                free_image(crop);
                int ind = max_index(pred, classes);

                Console.Write($"%s\n", labels[ind]);
            }
        }


        void test_classifier(string datacfg, string cfgfile, string weightfile, int target_layer)
        {
            int curr = 0;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* options = read_data_cfg(datacfg);

            string test_list = option_find_str(options, "test", "Data.Data/test.list");
            int classes = option_find_int(options, "classes", 2);

            list* plist = get_paths(test_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            clock_t time;

            Data.Data val, buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.Classes = classes;
            args.N = net.Batch;
            args.m = 0;
            args.labels = 0;
            args.d = &buffer;
            args.LayerType = OLD_CLASSIFICATION_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            for (curr = net.Batch; curr < m; curr += net.Batch)
            {
                time = clock();

                pthread_join(load_thread, 0);
                val = buffer;

                if (curr < m)
                {
                    args.paths = paths + curr;
                    if (curr + net.Batch > m) args.N = m - curr;
                    load_thread = load_data_in_thread(args);
                }
                Console.Error.Write($"Loaded: %d images ini %lf seconds\n", val.X.rows, sec(clock() - time));

                time = clock();
                Matrix pred = network_predict_data(net, val);

                int i, j;

                for (i = 0; i < pred.rows; ++i)
                {
                    Console.Write($"%s", paths[curr - net.Batch + i]);
                    for (j = 0; j < pred.cols; ++j)
                    {
                        Console.Write($"\t%g", pred.vals[i][j]);
                    }
                    Console.Write($"\n");
                }

                free_matrix(pred);

                Console.Error.Write($"%lf seconds, %d images, %d total\n", sec(clock() - time), val.X.rows, curr);
                free_data(val);
            }
        }


        void threat_classifier(string datacfg, string cfgfile, string weightfile, int cam_index, string filename)
        {
            float threat = 0;
            float roll = .2;

            Console.Write($"Classifier Demo\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            list* options = read_data_cfg(datacfg);

            srand(2222222);
            CvCapture* cap;

            if (filename)
            {
                cap = cvCaptureFromFile(filename);
            }
            else
            {
                cap = cvCaptureFromCAM(cam_index);
            }

            int top = option_find_int(options, "top", 1);

            string name_list = option_find_str(options, "names", 0);
            string[] names = get_labels(name_list);

            int[] indexes = calloc(top, sizeof(int));

            if (!cap) error("Couldn't connect to webcam.\n");
            float fps = 0;
            int i;

            int count = 0;

            while (1)
            {
                ++count;
                timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);

                Image ini = get_image_from_stream(cap);
                if (!ini.Data.Data) break;
                Image in_s = resize_image(ini, net.W, net.H);

                Image outo = ini;
                int x1 = outo.W / 20;
                int y1 = outo.H / 20;
                int x2 = 2 * x1;
                int y2 = outo.H - outo.H / 20;

                int border = .01 * outo.H;
                int h = y2 - y1 - 2 * border;
                int w = x2 - x1 - 2 * border;

                float[] predictions = network_predict(net, in_s.Data.Data);
                float curr_threat = 0;
                curr_threat = predictions[0] * 0 +
                    predictions[1] * .6 +
                    predictions[2];
                threat = roll * curr_threat + (1 - roll) * threat;

                draw_box_width(outo, x2 + border, y1 + .02 * h, x2 + .5 * w, y1 + .02 * h + border, border, 0, 0, 0);
                if (threat > .97)
                {
                    draw_box_width(outo, x2 + .5 * w + border,
                            y1 + .02 * h - 2 * border,
                            x2 + .5 * w + 6 * border,
                            y1 + .02 * h + 3 * border, 3 * border, 1, 0, 0);
                }
                draw_box_width(outo, x2 + .5 * w + border,
                        y1 + .02 * h - 2 * border,
                        x2 + .5 * w + 6 * border,
                        y1 + .02 * h + 3 * border, .5 * border, 0, 0, 0);
                draw_box_width(outo, x2 + border, y1 + .42 * h, x2 + .5 * w, y1 + .42 * h + border, border, 0, 0, 0);
                if (threat > .57)
                {
                    draw_box_width(outo, x2 + .5 * w + border,
                            y1 + .42 * h - 2 * border,
                            x2 + .5 * w + 6 * border,
                            y1 + .42 * h + 3 * border, 3 * border, 1, 1, 0);
                }
                draw_box_width(outo, x2 + .5 * w + border,
                        y1 + .42 * h - 2 * border,
                        x2 + .5 * w + 6 * border,
                        y1 + .42 * h + 3 * border, .5 * border, 0, 0, 0);

                draw_box_width(outo, x1, y1, x2, y2, border, 0, 0, 0);
                for (i = 0; i < threat * h; ++i)
                {
                    float ratio = (float)i / h;
                    float r = (ratio < .5) ? (2 * (ratio)) : 1;
                    float g = (ratio < .5) ? 1 : 1 - 2 * (ratio - .5);
                    draw_box_width(outo, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
                }
                top_predictions(net, top, indexes);
                char buff[256];
                sprintf(buff, "/home/pjreddie/tmp/threat_%06d", count);
                //save_image(outo, buff);

                Console.Write($"\033[2J");
                Console.Write($"\033[1;1H");
                Console.Write($"\nFPS:%.0f\n", fps);

                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }

                show_image(outo, "Threat");
                cvWaitKey(10);
                free_image(in_s);
                free_image(ini);

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                float curr = 1000000.f / ((long int)tval_result.tv_usec);
                fps = .9 * fps + .1 * curr;
            }
        }


        void gun_classifier(string datacfg, string cfgfile, string weightfile, int cam_index, string filename)
        {
            int[] bad_cats = { 218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697 };

            Console.Write($"Classifier Demo\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            list* options = read_data_cfg(datacfg);

            srand(2222222);
            CvCapture* cap;

            if (filename)
            {
                cap = cvCaptureFromFile(filename);
            }
            else
            {
                cap = cvCaptureFromCAM(cam_index);
            }

            int top = option_find_int(options, "top", 1);

            string name_list = option_find_str(options, "names", 0);
            string[] names = get_labels(name_list);

            int[] indexes = calloc(top, sizeof(int));

            if (!cap) error("Couldn't connect to webcam.\n");
            cvNamedWindow("Threat Detection", CV_WINDOW_NORMAL);
            cvResizeWindow("Threat Detection", 512, 512);
            float fps = 0;
            int i;

            while (1)
            {
                timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);

                Image ini = get_image_from_stream(cap);
                Image in_s = resize_image(ini, net.W, net.H);
                show_image(ini, "Threat Detection");

                float[] predictions = network_predict(net, in_s.Data.Data);
                top_predictions(net, top, indexes);

                Console.Write($"\033[2J");
                Console.Write($"\033[1;1H");

                int threat = 0;
                for (i = 0; i < sizeof(bad_cats) / sizeof(bad_cats[0]); ++i)
                {
                    int index = bad_cats[i];
                    if (predictions[index] > .01)
                    {
                        Console.Write($"Threat Detected!\n");
                        threat = 1;
                        break;
                    }
                }
                if (!threat) Console.Write($"Scanning...\n");
                for (i = 0; i < sizeof(bad_cats) / sizeof(bad_cats[0]); ++i)
                {
                    int index = bad_cats[i];
                    if (predictions[index] > .01)
                    {
                        Console.Write($"%s\n", names[index]);
                    }
                }

                free_image(in_s);
                free_image(ini);

                cvWaitKey(10);

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                float curr = 1000000.f / ((long int)tval_result.tv_usec);
                fps = .9 * fps + .1 * curr;
            }
        }

        void demo_classifier(string datacfg, string cfgfile, string weightfile, int cam_index, string filename)
        {
            Console.Write($"Classifier Demo\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            list* options = read_data_cfg(datacfg);

            srand(2222222);
            CvCapture* cap;

            if (filename)
            {
                cap = cvCaptureFromFile(filename);
            }
            else
            {
                cap = cvCaptureFromCAM(cam_index);
            }

            int top = option_find_int(options, "top", 1);

            string name_list = option_find_str(options, "names", 0);
            string[] names = get_labels(name_list);

            int[] indexes = calloc(top, sizeof(int));

            if (!cap) error("Couldn't connect to webcam.\n");
            cvNamedWindow("Classifier", CV_WINDOW_NORMAL);
            cvResizeWindow("Classifier", 512, 512);
            float fps = 0;
            int i;

            while (1)
            {
                timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);

                Image ini = get_image_from_stream(cap);
                Image in_s = resize_image(ini, net.W, net.H);
                show_image(ini, "Classifier");

                float[] predictions = network_predict(net, in_s.Data.Data);
                if (net.hierarchy) hierarchy_predictions(predictions, net.Outputs, net.hierarchy, 1);
                top_predictions(net, top, indexes);

                Console.Write($"\033[2J");
                Console.Write($"\033[1;1H");
                Console.Write($"\nFPS:%.0f\n", fps);

                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }

                free_image(in_s);
                free_image(ini);

                cvWaitKey(10);

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                float curr = 1000000.f / ((long int)tval_result.tv_usec);
                fps = .9 * fps + .1 * curr;
            }
        }


        public static void run_classifier(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string gpu_list = find_char_arg(args.Count, args, "-gpus", 0);
            int[] gpus = 0;
            int gpu = 0;
            int ngpus = 0;
            if (gpu_list)
            {
                Console.Write($"%s\n", gpu_list);
                int len = strlen(gpu_list);
                ngpus = 1;
                int i;
                for (i = 0; i < len; ++i)
                {
                    if (gpu_list[i] == ',') ++ngpus;
                }
                gpus = calloc(ngpus, sizeof(int));
                for (i = 0; i < ngpus; ++i)
                {
                    gpus[i] = int.Parse(gpu_list);
                    gpu_list = strchr(gpu_list, ',') + 1;
                }
            }
            else
            {
                gpu = CudaUtils.UseGpu;
                gpus = &gpu;
                ngpus = 1;
            }

            int cam_index = Utils.find_int_arg(args.Count, args, "-c", 0);
            int top = Utils.find_int_arg(args.Count, args, "-t", 0);
            int clear = find_arg(args.Count, args, "-clear");
            string Data.Data = args[3];
            string cfg = args[4];
            string weights = (args.Count > 5) ? args[5] : 0;
            string filename = (args.Count > 6) ? args[6] : 0;
            string layer_s = (args.Count > 7) ? args[7] : 0;
            int Layer = layer_s ? int.Parse(layer_s) : -1;
            if (0 == strcmp(args[2], "predict")) predict_classifier(Data.Data, cfg, weights, filename, top);
            else if (0 == strcmp(args[2], "try")) try_classifier(Data.Data, cfg, weights, filename, int.Parse(layer_s));
            else if (0 == strcmp(args[2], "train")) train_classifier(Data.Data, cfg, weights, gpus, ngpus, clear);
            else if (0 == strcmp(args[2], "demo")) demo_classifier(Data.Data, cfg, weights, cam_index, filename);
            else if (0 == strcmp(args[2], "gun")) gun_classifier(Data.Data, cfg, weights, cam_index, filename);
            else if (0 == strcmp(args[2], "threat")) threat_classifier(Data.Data, cfg, weights, cam_index, filename);
            else if (0 == strcmp(args[2], "test")) test_classifier(Data.Data, cfg, weights, Layer);
            else if (0 == strcmp(args[2], "label")) label_classifier(Data.Data, cfg, weights);
            else if (0 == strcmp(args[2], "valid")) validate_classifier_single(Data.Data, cfg, weights);
            else if (0 == strcmp(args[2], "validmulti")) validate_classifier_multi(Data.Data, cfg, weights);
            else if (0 == strcmp(args[2], "valid10")) validate_classifier_10(Data.Data, cfg, weights);
            else if (0 == strcmp(args[2], "validcrop")) validate_classifier_crop(Data.Data, cfg, weights);
            else if (0 == strcmp(args[2], "validfull")) validate_classifier_full(Data.Data, cfg, weights);
        }


        #endregion

        #region ArtFile


        void demo_art(string cfgfile, string weightfile, int cam_index)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);

            srand(2222222);
            CvCapture* cap;

            cap = cvCaptureFromCAM(cam_index);

            string window = "ArtJudgementBot9000!!!";
            if (!cap) error("Couldn't connect to webcam.\n");
            cvNamedWindow(window, CV_WINDOW_NORMAL);
            cvResizeWindow(window, 512, 512);
            int i;
            int idx[] = { 37, 401, 434 };
            int n = sizeof(idx) / sizeof(idx[0]);

            while (1)
            {
                Image ini = get_image_from_stream(cap);
                Image in_s = resize_image(ini, net.W, net.H);
                show_image(ini, window);

                float[] p = network_predict(net, in_s.Data.Data);

                Console.Write($"\033[2J");
                Console.Write($"\033[1;1H");

                float score = 0;
                for (i = 0; i < n; ++i)
                {
                    float s = p[idx[i]];
                    if (s > score) score = s;
                }
                score = score;
                Console.Write($"I APPRECIATE THIS ARTWORK: %10.7f%%\n", score * 100);
                Console.Write($"[");
                int upper = 30;
                for (i = 0; i < upper; ++i)
                {
                    Console.Write($"%c", ((i + .5) < score * upper) ? 219 : ' ');
                }
                Console.Write($"]\n");

                free_image(in_s);
                free_image(ini);

                cvWaitKey(1);
            }
        }


        public static void run_art(List<string> args)
        {
            int cam_index = Utils.find_int_arg(args.Count, args, "-c", 0);
            string cfg = args[2];
            string weights = args[3];
            demo_art(cfg, weights, cam_index);
        }


        #endregion

        #region TagFile


        void train_tag(string cfgfile, string weightfile, int clear)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            if (clear) net.Seen = 0;
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            list* plist = get_paths("/home/pjreddie/tag/train.list");
            string[] paths = (string[])list_to_array(plist);
            Console.Write($"%d\n", plist.Size);
            int N = plist.Size;
            clock_t time;
            pthread_t load_thread;
            Data.Data train;
            Data.Data buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;

            args.min = net.W;
            args.max = net.max_crop;
            args.Size = net.W;

            args.paths = paths;
            args.Classes = net.Outputs;
            args.N = imgs;
            args.m = N;
            args.d = &buffer;
            args.LayerType = TAG_DATA;

            args.angle = net.angle;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;

            Console.Error.Write($"%d classes\n", net.Outputs);

            load_thread = load_data_in_thread(args);
            int epoch = (net.Seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;

                load_thread = load_data_in_thread(args);
                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), net.Seen);
                free_data(train);
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                }
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s.backup", backup_directory, basec);
                    save_weights(net, buff);
                }
            }
            char buff[256];
            sprintf(buff, "%s/%s.Weights", backup_directory, basec);
            save_weights(net, buff);

            pthread_join(load_thread, 0);
            free_data(buffer);
            free_network(net);
            free_ptrs((void**)paths, plist.Size);
            free_list(plist);
            free(basec);
        }

        void test_tag(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);
            int i = 0;
            string[] names = get_labels("Data.Data/tags.txt");
            clock_t time;
            int indexes[10];
            char buff[256];
            string input = buff;
            int size = net.W;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, 0, 0);
                Image r = resize_min(im, size);
                resize_network(net, r.W, r.H);
                Console.Write($"%d %d\n", r.W, r.H);

                float[] X = r.Data.Data;
                time = clock();
                float[] predictions = network_predict(net, X);
                top_predictions(net, 10, indexes);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                for (i = 0; i < 10; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }
                if (r.Data.Data != im.Data.Data) free_image(r);
                free_image(im);
                if (filename) break;
            }
        }


        public static void run_tag(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            int clear = find_arg(args.Count, args, "-clear");
            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "train")) train_tag(cfg, weights, clear);
            else if (0 == strcmp(args[2], "test")) test_tag(cfg, weights, filename);
        }


        #endregion

        #region CompareFile


        void train_compare(string cfgfile, string weightfile)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            list* plist = get_paths("Data.Data/compare.train.list");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.Size;
            Console.Write($"%d\n", N);
            clock_t time;
            pthread_t load_thread;
            Data.Data train;
            Data.Data.Data buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.Classes = 20;
            args.N = imgs;
            args.m = N;
            args.d = &buffer;
            args.LayerType = COMPARE_DATA;

            load_thread = load_data_in_thread(args);
            int epoch = net.Seen / N;
            int i = 0;
            while (1)
            {
                ++i;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;

                load_thread = load_data_in_thread(args);
                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                Console.Write($"%.3f: %f, %f avg, %lf seconds, %d images\n", (float)net.Seen / N, loss, avg_loss, sec(clock() - time), net.Seen);
                free_data(train);
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d_minor_%d.Weights", backup_directory, basec, epoch, i);
                    save_weights(net, buff);
                }
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    i = 0;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                    if (epoch % 22 == 0) net.LearningRate *= .1;
                }
            }
            pthread_join(load_thread, 0);
            free_data(buffer);
            free_network(net);
            free_ptrs((void**)paths, plist.Size);
            free_list(plist);
            free(basec);
        }

        void validate_compare(string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            list* plist = get_paths("Data.Data/compare.val.list");
            //list *plist = get_paths("Data.Data/compare.val.old");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.Size / 2;
            free_list(plist);

            clock_t time;
            int correct = 0;
            int total = 0;
            int splits = 10;
            int num = (i + 1) * N / splits - i * N / splits;

            Data.Data.Data val, buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.Classes = 20;
            args.N = num;
            args.m = 0;
            args.d = &buffer;
            args.LayerType = COMPARE_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            for (i = 1; i <= splits; ++i)
            {
                time = clock();

                pthread_join(load_thread, 0);
                val = buffer;

                num = (i + 1) * N / splits - i * N / splits;
                string[] part = paths + (i * N / splits);
                if (i != splits)
                {
                    args.paths = part;
                    load_thread = load_data_in_thread(args);
                }
                Console.Write($"Loaded: %d images ini %lf seconds\n", val.X.rows, sec(clock() - time));

                time = clock();
                Matrix pred = network_predict_data(net, val);
                int j, k;
                for (j = 0; j < val.y.rows; ++j)
                {
                    for (k = 0; k < 20; ++k)
                    {
                        if (val.y.vals[j][k * 2] != val.y.vals[j][k * 2 + 1])
                        {
                            ++total;
                            if ((val.y.vals[j][k * 2] < val.y.vals[j][k * 2 + 1]) == (pred.vals[j][k * 2] < pred.vals[j][k * 2 + 1]))
                            {
                                ++correct;
                            }
                        }
                    }
                }
                free_matrix(pred);
                Console.Write($"%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct / total, sec(clock() - time), val.X.rows);
                free_data(val);
            }
        }


        int total_compares = 0;
        int current_class = 0;

        int elo_comparator(void* a, void* b)
        {
            SortableBbox box1 = *(SortableBbox*)a;
            SortableBbox box2 = *(SortableBbox*)b;
            if (box1.elos[current_class] == box2.elos[current_class]) return 0;
            if (box1.elos[current_class] > box2.elos[current_class]) return -1;
            return 1;
        }

        int bbox_comparator(void* a, void* b)
        {
            ++total_compares;
            SortableBbox box1 = *(SortableBbox*)a;
            SortableBbox box2 = *(SortableBbox*)b;
            Network net = box1.net;
            int sclass = box1.sclass;

            Image im1 = load_image_color(box1.filename, net.W, net.H);
            Image im2 = load_image_color(box2.filename, net.W, net.H);
            float[] X = (float[])calloc(net.W * net.H * net.C, sizeof(float));
            memcpy(X, im1.Data.Data, im1.W * im1.H * im1.C * sizeof(float));
            memcpy(X + im1.W * im1.H * im1.C, im2.Data.Data, im2.W * im2.H * im2.C * sizeof(float));
            float[] predictions = network_predict(net, X);

            free_image(im1);
            free_image(im2);
            free(X);
            if (predictions[sclass * 2] > predictions[sclass * 2 + 1])
            {
                return 1;
            }
            return -1;
        }

        void bbox_update(SortableBbox* a, SortableBbox* b, int sclass, int result)
        {
            int k = 32;
            float EA = 1.0f / (1 + pow(10, (b.elos[sclass] - a.elos[sclass]) / 400.));
            float EB = 1.0f / (1 + pow(10, (a.elos[sclass] - b.elos[sclass]) / 400.));
            float SA = result ? 1 : 0;
            float SB = result ? 0 : 1;
            a.elos[sclass] += k * (SA - EA);
            b.elos[sclass] += k * (SB - EB);
        }

        void bbox_fight(Network net, SortableBbox* a, SortableBbox* b, int classes, int sclass)
        {
            Image im1 = load_image_color(a.filename, net.W, net.H);
            Image im2 = load_image_color(b.filename, net.W, net.H);
            float[] X = (float[])calloc(net.W * net.H * net.C, sizeof(float));
            memcpy(X, im1.Data.Data, im1.W * im1.H * im1.C * sizeof(float));
            memcpy(X + im1.W * im1.H * im1.C, im2.Data.Data, im2.W * im2.H * im2.C * sizeof(float));
            float[] predictions = network_predict(net, X);
            ++total_compares;

            int i;
            for (i = 0; i < classes; ++i)
            {
                if (sclass < 0 || sclass == i)
                {
                    int result = predictions[i * 2] > predictions[i * 2 + 1];
                    bbox_update(a, b, i, result);
                }
            }

            free_image(im1);
            free_image(im2);
            free(X);
        }

        void SortMaster3000(string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            
            set_batch_network(net, 1);

            list* plist = get_paths("Data.Data/compare.sort.list");
            //list *plist = get_paths("Data.Data/compare.val.old");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.Size;
            free_list(plist);
            SortableBbox* boxes = (SortableBbox*)calloc(N, sizeof(SortableBbox));
            Console.Write($"Sorting %d boxes...\n", N);
            for (i = 0; i < N; ++i)
            {
                boxes[i].filename = paths[i];
                boxes[i].net = net;
                boxes[i].sclass = 7;
                boxes[i].elo = 1500;
            }
            clock_t time = clock();
            qsort(boxes, N, sizeof(SortableBbox), bbox_comparator);
            for (i = 0; i < N; ++i)
            {
                Console.Write($"%s\n", boxes[i].filename);
            }
            Console.Write($"Sorted ini %d compares, %f secs\n", total_compares, sec(clock() - time));
        }

        void BattleRoyaleWithCheese(string filename, string weightfile)
        {
            int classes = 20;
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            
            set_batch_network(net, 1);

            list* plist = get_paths("Data.Data/compare.sort.list");
            //list *plist = get_paths("Data.Data/compare.small.list");
            //list *plist = get_paths("Data.Data/compare.cat.list");
            //list *plist = get_paths("Data.Data/compare.val.old");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.Size;
            int total = N;
            free_list(plist);
            SortableBbox* boxes = (SortableBbox*)calloc(N, sizeof(SortableBbox));
            Console.Write($"Battling %d boxes...\n", N);
            for (i = 0; i < N; ++i)
            {
                boxes[i].filename = paths[i];
                boxes[i].net = net;
                boxes[i].Classes = classes;
                boxes[i].elos = (float[])calloc(classes, sizeof(float)); ;
                for (j = 0; j < classes; ++j)
                {
                    boxes[i].elos[j] = 1500;
                }
            }
            int round;
            clock_t time = clock();
            for (round = 1; round <= 4; ++round)
            {
                clock_t round_time = clock();
                Console.Write($"Round: %d\n", round);
                shuffle(boxes, N, sizeof(SortableBbox));
                for (i = 0; i < N / 2; ++i)
                {
                    bbox_fight(net, boxes + i * 2, boxes + i * 2 + 1, classes, -1);
                }
                Console.Write($"Round: %f secs, %d remaining\n", sec(clock() - round_time), N);
            }

            int sclass;

            for (sclass = 0; sclass < classes; ++sclass)
            {

                N = total;
                current_class = sclass;
                qsort(boxes, N, sizeof(SortableBbox), elo_comparator);
                N /= 2;

                for (round = 1; round <= 100; ++round)
                {
                    clock_t round_time = clock();
                    Console.Write($"Round: %d\n", round);

                    sorta_shuffle(boxes, N, sizeof(SortableBbox), 10);
                    for (i = 0; i < N / 2; ++i)
                    {
                        bbox_fight(net, boxes + i * 2, boxes + i * 2 + 1, classes, sclass);
                    }
                    qsort(boxes, N, sizeof(SortableBbox), elo_comparator);
                    if (round <= 20) N = (N * 9 / 10) / 2 * 2;

                    Console.Write($"Round: %f secs, %d remaining\n", sec(clock() - round_time), N);
                }
                char buff[256];
                sprintf(buff, "results/battle_%d.log", sclass);
                FILE* outfp = fopen(buff, "w");
                for (i = 0; i < N; ++i)
                {
                    fprintf(outfp, "%s %f\n", boxes[i].filename, boxes[i].elos[sclass]);
                }
                fclose(outfp);
            }
            Console.Write($"Tournament ini %d compares, %f secs\n", total_compares, sec(clock() - time));
        }

        public static void run_compare(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            //char *filename = (args.Count > 5) ? args[5]: 0;
            if (0 == strcmp(args[2], "train")) train_compare(cfg, weights);
            else if (0 == strcmp(args[2], "valid")) validate_compare(cfg, weights);
            else if (0 == strcmp(args[2], "sort")) SortMaster3000(cfg, weights);
            else if (0 == strcmp(args[2], "battle")) BattleRoyaleWithCheese(cfg, weights);
        }


        #endregion

        #region DiceFile


        char[] dice_labels = { "face1", "face2", "face3", "face4", "face5", "face6" };

        void train_dice(string cfgfile, string weightfile)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            int i = net.Seen / imgs;
            string[] labels = dice_labels;
            list* plist = get_paths("Data.Data/dice/dice.train.list");
            string[] paths = (string[])list_to_array(plist);
            Console.Write($"%d\n", plist.Size);
            clock_t time;
            while (1)
            {
                ++i;
                time = clock();
                Data.Data train = load_data_old(paths, imgs, plist.Size, labels, 6, net.W, net.H);
                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                Console.Write($"%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock() - time), net.Seen);
                free_data(train);
                if ((i % 100) == 0) net.LearningRate *= .1;
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
            }
        }

        void validate_dice(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            

            string[] labels = dice_labels;
            list* plist = get_paths("Data.Data/dice/dice.val.list");

            string[] paths = (string[])list_to_array(plist);
            int m = plist.Size;
            free_list(plist);

            Data.Data val = load_data_old(paths, m, 0, labels, 6, net.W, net.H);
            float[] acc = network_accuracies(net, val, 2);
            Console.Write($"Validation Accuracy: %f, %d images\n", acc[0], m);
            free_data(val);
        }

        void test_dice(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);
            int i = 0;
            string[] names = dice_labels;
            char buff[256];
            string input = buff;
            int indexes[6];
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, net.W, net.H);
                float[] X = im.Data.Data;
                float[] predictions = network_predict(net, X);
                top_predictions(net, 6, indexes);
                for (i = 0; i < 6; ++i)
                {
                    int index = indexes[i];
                    Console.Write($"%s: %f\n", names[index], predictions[index]);
                }
                free_image(im);
                if (filename) break;
            }
        }

        public static void run_dice(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "test")) test_dice(cfg, weights, filename);
            else if (0 == strcmp(args[2], "train")) train_dice(cfg, weights);
            else if (0 == strcmp(args[2], "valid")) validate_dice(cfg, weights);
        }



        #endregion

        #region WritingFile


        void train_writing(string cfgfile, string weightfile)
        {
            string backup_directory = "/home/pjreddie/backup/";
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = net.Batch * net.Subdivisions;
            list* plist = get_paths("figures.list");
            string[] paths = (string[])list_to_array(plist);
            clock_t time;
            int N = plist.Size;
            Console.Write($"N: %d\n", N);
            Image outf = get_network_image(net);

            Data.Data train, buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.OutW = outf.W;
            args.OutH = outf.H;
            args.paths = paths;
            args.N = imgs;
            args.m = N;
            args.d = &buffer;
            args.LayerType = WRITING_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            int epoch = (net.Seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data_in_thread(args);
                Console.Write($"Loaded %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);


                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(net.Seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), net.Seen);
                free_data(train);
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_batch_%d.Weights", backup_directory, basec, get_current_batch(net));
                    save_weights(net, buff);
                }
                if (net.Seen / N > epoch)
                {
                    epoch = net.Seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.Weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                }
            }
        }

        void test_writing(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);
            clock_t time;
            char buff[256];
            string input = buff;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    Console.Write($"Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }

                Image im = load_image_color(input, 0, 0);
                resize_network(net, im.W, im.H);
                Console.Write($"%d %d %d\n", im.H, im.W, im.C);
                float[] X = im.Data.Data;
                time = clock();
                network_predict(net, X);
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                Image pred = get_network_image(net);

                Image upsampled = resize_image(pred, im.W, im.H);
                Image thresh = threshold_image(upsampled, .5);
                pred = thresh;

                show_image(pred, "prediction");
                show_image(im, "orig");
                cvWaitKey(0);
                cvDestroyAllWindows();

                free_image(upsampled);
                free_image(thresh);
                free_image(im);
                if (filename) break;
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
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "train")) train_writing(cfg, weights);
            else if (0 == strcmp(args[2], "test")) test_writing(cfg, weights, filename);
        }


        #endregion

        #region CaptchaFile


        void fix_data_captcha(Data.Data d, int mask)
        {
            Matrix labels = d.y;
            int i, j;
            for (i = 0; i < d.y.rows; ++i)
            {
                for (j = 0; j < d.y.cols; j += 2)
                {
                    if (mask)
                    {
                        if (!labels.vals[i][j])
                        {
                            labels.vals[i][j] = SECRET_NUM;
                            labels.vals[i][j + 1] = SECRET_NUM;
                        }
                        else if (labels.vals[i][j + 1])
                        {
                            labels.vals[i][j] = 0;
                        }
                    }
                    else
                    {
                        if (labels.vals[i][j])
                        {
                            labels.vals[i][j + 1] = 0;
                        }
                        else
                        {
                            labels.vals[i][j + 1] = 1;
                        }
                    }
                }
            }
        }

        void train_captcha(string cfgfile, string weightfile)
        {
            
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            int i = net.Seen / imgs;
            int solved = 1;
            list* plist;
            string[] labels = get_labels("/Data.Data/captcha/reimgs.labels.list");
            if (solved)
            {
                plist = get_paths("/Data.Data/captcha/reimgs.solved.list");
            }
            else
            {
                plist = get_paths("/Data.Data/captcha/reimgs.raw.list");
            }
            string[] paths = (string[])list_to_array(plist);
            Console.Write($"%d\n", plist.Size);
            clock_t time;
            pthread_t load_thread;
            Data.Data train;
            Data.Data buffer;

            LoadArgs args = { 0 };
            args.W = net.W;
            args.H = net.H;
            args.paths = paths;
            args.Classes = 26;
            args.N = imgs;
            args.m = plist.Size;
            args.labels = labels;
            args.d = &buffer;
            args.LayerType = CLASSIFICATION_DATA;

            load_thread = load_data_in_thread(args);
            while (1)
            {
                ++i;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                fix_data_captcha(train, solved);

                load_thread = load_data_in_thread(args);
                Console.Write($"Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                Console.Write($"%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock() - time), net.Seen);
                free_data(train);
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.Weights", basec, i);
                    save_weights(net, buff);
                }
            }
        }

        void test_captcha(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            set_batch_network(net, 1);
            srand(2222222);
            int i = 0;
            string[] names = get_labels("/Data.Data/captcha/reimgs.labels.list");
            char buff[256];
            string input = buff;
            int indexes[26];
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    //Console.Write($"Enter Image Path: ");
                    //fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                Image im = load_image_color(input, net.W, net.H);
                float[] X = im.Data.Data;
                float[] predictions = network_predict(net, X);
                top_predictions(net, 26, indexes);
                //Console.Write($"%s: Predicted in %f seconds.\n", input, sec(clock()-time));
                for (i = 0; i < 26; ++i)
                {
                    int index = indexes[i];
                    if (i != 0) Console.Write($", ");
                    Console.Write($"%s %f", names[index], predictions[index]);
                }
                Console.Write($"\n");
                fflush(stdout);
                free_image(im);
                if (filename) break;
            }
        }

        void valid_captcha(string cfgfile, string weightfile, string filename)
        {
            string[] labels = get_labels("/Data.Data/captcha/reimgs.labels.list");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (weightfile)
            {
                Parser.load_weights(net, weightfile);
            }
            list* plist = get_paths("/Data.Data/captcha/reimgs.fg.list");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.Size;
            int outputs = net.Outputs;

            set_batch_network(net, 1);
            srand(2222222);
            int i, j;
            for (i = 0; i < N; ++i)
            {
                if (i % 100 == 0) Console.Error.Write($"%d\n", i);
                Image im = load_image_color(paths[i], net.W, net.H);
                float[] X = im.Data.Data;
                float[] predictions = network_predict(net, X);
                //Console.Write($"%s: Predicted in %f seconds.\n", input, sec(clock()-time));
                int truth = -1;
                for (j = 0; j < 13; ++j)
                {
                    if (strstr(paths[i], labels[j])) truth = j;
                }
                if (truth == -1)
                {
                    Console.Error.Write($"bad: %s\n", paths[i]);
                    return;
                }
                Console.Write($"%d, ", truth);
                for (j = 0; j < outputs; ++j)
                {
                    if (j != 0) Console.Write($", ");
                    Console.Write($"%f", predictions[j]);
                }
                Console.Write($"\n");
                fflush(stdout);
                free_image(im);
                if (filename) break;
            }
        }

        public static void run_captcha(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : 0;
            string filename = (args.Count > 5) ? args[5] : 0;
            if (0 == strcmp(args[2], "train")) train_captcha(cfg, weights);
            else if (0 == strcmp(args[2], "test")) test_captcha(cfg, weights, filename);
            else if (0 == strcmp(args[2], "valid")) valid_captcha(cfg, weights, filename);
        }


        #endregion

        #region NightmareFile


        float abs_mean(float[] x, int n)
        {
            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                sum += fabs(x[i]);
            }
            return sum / n;
        }

        void calculate_loss(float[] output, float[] delta, int n, float thresh)
        {
            int i;
            float mean = mean_array(output, n);
            float var = variance_array(output, n);
            for (i = 0; i < n; ++i)
            {
                if (delta[i] > mean + thresh * sqrt(var)) delta[i] = output[i];
                else delta[i] = 0;
            }
        }

        void optimize_picture(Network* net, Image orig, int max_layer, float scale, float rate, float thresh, int norm)
        {
            //scale_image(orig, 2);
            //translate_image(orig, -1);
            net.N = max_layer + 1;

            int dx = rand() % 16 - 8;
            int dy = rand() % 16 - 8;
            int flip = rand() % 2;

            Image crop = crop_image(orig, dx, dy, orig.W, orig.H);
            Image im = resize_image(crop, (int)(orig.W * scale), (int)(orig.H * scale));
            if (flip) flip_image(im);

            resize_network(net, im.W, im.H);
            Layer last = net.Layers[net.N - 1];
            //net.Layers[net.N - 1].activation = LINEAR;

            Image delta = new Image(im.W, im.H, im.C);

            network_state state = { 0 };
            i
            state.input = cuda_make_array(im.Data.Data, im.W * im.H * im.C);
            state.delta = cuda_make_array(im.Data.Data, im.W * im.H * im.C);

            forward_network_gpu(net, state);
            copy_ongpu(last.Outputs, last.output_gpu, 1, last.delta_gpu, 1);

            cuda_pull_array(last.delta_gpu, last.delta, last.Outputs);
            calculate_loss(last.delta, last.delta, last.Outputs, thresh);
            cuda_push_array(last.delta_gpu, last.delta, last.Outputs);

            backward_network_gpu(net, state);

            cuda_pull_array(state.delta, delta.Data.Data, im.W * im.H * im.C); i

            if (flip) flip_image(delta);
            //normalize_array(delta.Data.Data, delta.W*delta.H*delta.C);
            Image resized = resize_image(delta, orig.W, orig.H);
            Image outi = crop_image(resized, -dx, -dy, orig.W, orig.H);

            if (norm) normalize_array(outi.Data.Data, outi.W * outi.H * outi.C);
            axpy_cpu(orig.W * orig.H * orig.C, rate, outi.Data.Data, 1, orig.Data.Data, 1);


            constrain_image(orig);

            free_image(crop);
            free_image(im);
            free_image(delta);
            free_image(resized);
            free_image(outi);

        }

        void smooth(Image recon, Image update, float lambda, int num)
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
                                update.Data.Data[out_index] += lambda * (recon.Data.Data[in_index] - recon.Data.Data[out_index]);
                            }
                        }
                    }
                }
            }
        }

        void reconstruct_picture(Network net, float[] features, Image recon, Image update, float rate, float momentum, float lambda, int smooth_size, int iters)
        {
            int iter = 0;
            for (iter = 0; iter < iters; ++iter)
            {
                Image delta = new Image(recon.W, recon.H, recon.C);

                network_state state = { 0 };
                state.input = cuda_make_array(recon.Data.Data, recon.W * recon.H * recon.C);
                state.delta = cuda_make_array(delta.Data.Data, delta.W * delta.H * delta.C);
                state.truth = cuda_make_array(features, get_network_output_size(net));

                forward_network_gpu(net, state);
                backward_network_gpu(net, state);

                cuda_pull_array(state.delta, delta.Data.Data, delta.W * delta.H * delta.C);

                axpy_cpu(recon.W * recon.H * recon.C, 1, delta.Data.Data, 1, update.Data.Data, 1);
                smooth(recon, update, lambda, smooth_size);

                axpy_cpu(recon.W * recon.H * recon.C, rate, update.Data.Data, 1, recon.Data.Data, 1);
                Blas.Scal_cpu(recon.W * recon.H * recon.C, momentum, update.Data.Data, 1);

                //float mag = mag_array(recon.Data.Data, recon.W*recon.H*recon.C);
                //Blas.Scal_cpu(recon.W*recon.H*recon.C, 600/mag, recon.Data.Data, 1);

                constrain_image(recon);
                free_image(delta);
            }
        }


        public static void run_nightmare(List<string> args)
        {
            srand(0);
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [cfg] [weights] [Image] [Layer] [options! (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[2];
            string weights = args[3];
            string input = args[4];
            int max_layer = int.Parse(args[5]);

            int range = Utils.find_int_arg(args.Count, args, "-range", 1);
            int norm = Utils.find_int_arg(args.Count, args, "-norm", 1);
            int rounds = Utils.find_int_arg(args.Count, args, "-rounds", 1);
            int iters = Utils.find_int_arg(args.Count, args, "-iters", 10);
            int octaves = Utils.find_int_arg(args.Count, args, "-octaves", 4);
            float zoom = find_float_arg(args.Count, args, "-zoom", 1.);
            float rate = find_float_arg(args.Count, args, "-rate", .04);
            float thresh = find_float_arg(args.Count, args, "-thresh", 1.);
            float rotate = find_float_arg(args.Count, args, "-rotate", 0);
            float momentum = find_float_arg(args.Count, args, "-momentum", .9);
            float lambda = find_float_arg(args.Count, args, "-lambda", .01);
            string prefix = find_char_arg(args.Count, args, "-prefix", 0);
            int reconstruct = find_arg(args.Count, args, "-reconstruct");
            int smooth_size = Utils.find_int_arg(args.Count, args, "-smooth", 1);

            Network net = Parser.parse_network_cfg(cfg);
            Parser.load_weights(net, weights);
            string cfgbase = basecfg(cfg);
            string imbase = basecfg(input);

            set_batch_network(net, 1);
            Image im = load_image_color(input, 0, 0);
            if (0)
            {
                float scale = 1;
                if (im.W > 512 || im.H > 512)
                {
                    if (im.W > im.H) scale = 512.0 / im.W;
                    else scale = 512.0 / im.H;
                }
                Image resized = resize_image(im, scale * im.W, scale * im.H);
                free_image(im);
                im = resized;
            }

            float[] features = 0;
            Image update;
            if (reconstruct)
            {
                resize_network(net, im.W, im.H);

                int zz = 0;
                network_predict(net, im.Data.Data);
                Image out_im = get_network_image(net);
                Image crop = crop_image(out_im, zz, zz, out_im.W - 2 * zz, out_im.H - 2 * zz);
                //flip_image(crop);
                Image f_im = resize_image(crop, out_im.W, out_im.H);
                free_image(crop);
                Console.Write($"%d features\n", out_im.W * out_im.H * out_im.C);


                im = resize_image(im, im.W, im.H);
                f_im = resize_image(f_im, f_im.W, f_im.H);
                features = f_im.Data.Data;

                int i;
                for (i = 0; i < 14 * 14 * 512; ++i)
                {
                    features[i] += rand_uniform(-.19, .19);
                }

                free_image(im);
                im = make_random_image(im.W, im.H, im.C);
                update = new Image(im.W, im.H, im.C);

            }

            int e;
            int n;
            for (e = 0; e < rounds; ++e)
            {
                Console.Error.Write($"Iteration: ");
                fflush(stderr);
                for (n = 0; n < iters; ++n)
                {
                    Console.Error.Write($"%d, ", n);
                    fflush(stderr);
                    if (reconstruct)
                    {
                        reconstruct_picture(net, features, im, update, rate, momentum, lambda, smooth_size, 1);
                        //if ((n+1)%30 == 0) rate *= .5;
                        show_image(im, "reconstruction");
                        cvWaitKey(10);
                    }
                    else
                    {
                        int Layer = max_layer + rand() % range - range / 2;
                        int octave = rand() % octaves;
                        optimize_picture(net, im, Layer, 1 / pow(1.33333333, octave), rate, thresh, norm);
                    }
                }
                Console.Error.Write($"done\n");
                if (0)
                {
                    Image g = grayscale_image(im);
                    free_image(im);
                    im = g;
                }
                char buff[256];
                if (prefix)
                {
                    sprintf(buff, "%s/%s_%s_%d_%06d", prefix, imbase, cfgbase, max_layer, e);
                }
                else
                {
                    sprintf(buff, "%s_%s_%d_%06d", imbase, cfgbase, max_layer, e);
                }
                Console.Write($"%d %s\n", e, buff);
                save_image(im, buff);
                //show_image(im, buff);
                //cvWaitKey(0);

                if (rotate)
                {
                    Image rot = rotate_image(im, rotate);
                    free_image(im);
                    im = rot;
                }
                Image crop = crop_image(im, im.W * (1. - zoom) / 2., im.H * (1.- zoom) / 2., im.W * zoom, im.H * zoom);
                Image resized = resize_image(crop, im.W, im.H);
                free_image(im);
                free_image(crop);
                im = resized;
            }
        }


        #endregion
    }
}
