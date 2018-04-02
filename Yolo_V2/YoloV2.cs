﻿using System;
using System.Collections.Generic;
using System.Linq;
using Yolo_V2.Data;

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
                    test_detector("cfg/coco.data", args[2], args[3], filename, thresh);
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
                    predict_classifier("cfg/imagenet1k.data", args[2], args[3], args[4], 5);
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
                    composite_3d(args[2], args[3], args[4], (args.Count > 5) ? atof(args[5]) : 0);
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
                    speed(args[2], (args.Count > 3 && !string.IsNullOrEmpty(args[3])) ? atoi(args[3]) : 0);
                    break;
                case "oneoff":
                    oneoff(args[2], args[3], args[4]);
                    break;
                case "partial":
                    partial(args[2], args[3], args[4], atoi(args[5]));
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

            Network net = parse_network_cfg(cfgfile);
            Network sum = parse_network_cfg(cfgfile);

            string weightfile = args[4];
            load_weights(&sum, weightfile);

            int i, j;
            int n = args.Count - 5;
            for (i = 0; i < n; ++i)
            {
                weightfile = args[i + 5];
                load_weights(&net, weightfile);
                for (j = 0; j < net.n; ++j)
                {
                    Layer l = net.layers[j];
                    Layer outl = sum.layers[j];
                    if (l.type == CONVOLUTIONAL)
                    {
                        int num = l.n * l.c * l.size * l.size;
                        Blas.Axpy_cpu(l.n, 1, l.Biases, 1, outl.Biases, 1);
                        Blas.Axpy_cpu(num, 1, l.weights, 1, outl.weights, 1);
                        if (l.batch_normalize)
                        {
                            Blas.Axpy_cpu(l.n, 1, l.scales, 1, outl.scales, 1);
                            Blas.Axpy_cpu(l.n, 1, l.rolling_mean, 1, outl.rolling_mean, 1);
                            Blas.Axpy_cpu(l.n, 1, l.rolling_variance, 1, outl.rolling_variance, 1);
                        }
                    }
                    if (l.type == CONNECTED)
                    {
                        Blas.Axpy_cpu(l.outputs, 1, l.Biases, 1, outl.Biases, 1);
                        Blas.Axpy_cpu(l.outputs * l.inputs, 1, l.weights, 1, outl.weights, 1);
                    }
                }
            }
            n = n + 1;
            for (j = 0; j < net.n; ++j)
            {
                Layer l = sum.layers[j];
                if (l.type == CONVOLUTIONAL)
                {
                    int num = l.n * l.c * l.size * l.size;
                    scal_cpu(l.n, 1./ n, l.Biases, 1);
                    scal_cpu(num, 1./ n, l.weights, 1);
                    if (l.batch_normalize)
                    {
                        scal_cpu(l.n, 1./ n, l.scales, 1);
                        scal_cpu(l.n, 1./ n, l.rolling_mean, 1);
                        scal_cpu(l.n, 1./ n, l.rolling_variance, 1);
                    }
                }
                if (l.type == CONNECTED)
                {
                    scal_cpu(l.outputs, 1./ n, l.Biases, 1);
                    scal_cpu(l.outputs * l.inputs, 1./ n, l.weights, 1);
                }
            }
            save_weights(sum, outfile);
        }

        public static void speed(string cfgfile, int tics)
        {
            if (tics == 0) tics = 1000;
            Network net = parse_network_cfg(cfgfile);
            set_batch_network(&net, 1);
            int i;
            time_t start = time(0);
            image im = make_image(net.w, net.h, net.c);
            for (i = 0; i < tics; ++i)
            {
                network_predict(net, im.data);
            }
            double t = difftime(time(0), start);
            printf("\n%d evals, %f Seconds\n", tics, t);
            printf("Speed: %f sec/eval\n", t / tics);
            printf("Speed: %f Hz\n", tics / t);
        }

        public static void operations(string cfgfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            int i;
            long ops = 0;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL)
                {
                    ops += 2l * l.n * l.size * l.size * l.c * l.out_h * l.out_w;
                }
                else if (l.type == CONNECTED)
                {
                    ops += 2l * l.inputs * l.outputs;
                }
            }
            printf("Floating Point Operations: %ld\n", ops);
            printf("Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.);
        }

        public static void oneoff(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            int oldn = net.layers[net.n - 2].n;
            int c = net.layers[net.n - 2].c;
            net.layers[net.n - 2].n = 9372;
            net.layers[net.n - 2].Biases += 5;
            net.layers[net.n - 2].weights += 5 * c;
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            net.layers[net.n - 2].Biases -= 5;
            net.layers[net.n - 2].weights -= 5 * c;
            net.layers[net.n - 2].n = oldn;
            printf("%d\n", oldn);
            Layer l = net.layers[net.n - 2];
            copy_cpu(l.n / 3, l.Biases, 1, l.Biases + l.n / 3, 1);
            copy_cpu(l.n / 3, l.Biases, 1, l.Biases + 2 * l.n / 3, 1);
            copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + l.n / 3 * l.c, 1);
            copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + 2 * l.n / 3 * l.c, 1);
            *net.seen = 0;
            save_weights(net, outfile);
        }

        public static void partial(string cfgfile, string weightfile, string outfile, int max)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights_upto(&net, weightfile, max);
            }
            *net.seen = 0;
            save_weights_upto(net, outfile, max);
        }

        public static void rescale_net(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int i;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL)
                {
                    rescale_weights(l, 2, -.5);
                    break;
                }
            }
            save_weights(net, outfile);
        }

        public static void rgbgr_net(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int i;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL)
                {
                    rgbgr_weights(l);
                    break;
                }
            }
            save_weights(net, outfile);
        }

        public static void reset_normalize_net(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int i;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL && l.batch_normalize)
                {
                    denormalize_convolutional_layer(l);
                }
                if (l.type == CONNECTED && l.batch_normalize)
                {
                    denormalize_connected_layer(l);
                }
                if (l.type == GRU && l.batch_normalize)
                {
                    denormalize_connected_layer(*l.input_z_layer);
                    denormalize_connected_layer(*l.input_r_layer);
                    denormalize_connected_layer(*l.input_h_layer);
                    denormalize_connected_layer(*l.state_z_layer);
                    denormalize_connected_layer(*l.state_r_layer);
                    denormalize_connected_layer(*l.state_h_layer);
                }
            }
            save_weights(net, outfile);
        }

        private Layer normalize_layer(Layer l, int n)
        {
            int j;
            l.batch_normalize = 1;
            l.scales = calloc(n, sizeof(float));
            for (j = 0; j < n; ++j)
            {
                l.scales[j] = 1;
            }
            l.rolling_mean = calloc(n, sizeof(float));
            l.rolling_variance = calloc(n, sizeof(float));
            return l;
        }

        public static void normalize_net(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int i;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL && !l.batch_normalize)
                {
                    net.layers[i] = normalize_layer(l, l.n);
                }
                if (l.type == CONNECTED && !l.batch_normalize)
                {
                    net.layers[i] = normalize_layer(l, l.outputs);
                }
                if (l.type == GRU && l.batch_normalize)
                {
                    *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer.outputs);
                    *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer.outputs);
                    *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer.outputs);
                    *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer.outputs);
                    *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer.outputs);
                    *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer.outputs);
                    net.layers[i].batch_normalize = 1;
                }
            }
            save_weights(net, outfile);
        }

        public static void statistics_net(string cfgfile, string weightfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int i;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONNECTED && l.batch_normalize)
                {
                    printf("Connected Layer %d\n", i);
                    statistics_connected_layer(l);
                }
                if (l.type == GRU && l.batch_normalize)
                {
                    printf("GRU Layer %d\n", i);
                    printf("Input Z\n");
                    statistics_connected_layer(*l.input_z_layer);
                    printf("Input R\n");
                    statistics_connected_layer(*l.input_r_layer);
                    printf("Input H\n");
                    statistics_connected_layer(*l.input_h_layer);
                    printf("State Z\n");
                    statistics_connected_layer(*l.state_z_layer);
                    printf("State R\n");
                    statistics_connected_layer(*l.state_r_layer);
                    printf("State H\n");
                    statistics_connected_layer(*l.state_h_layer);
                }
                printf("\n");
            }
        }

        public static void denormalize_net(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int i;
            for (i = 0; i < net.n; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL && l.batch_normalize)
                {
                    denormalize_convolutional_layer(l);
                    net.layers[i].batch_normalize = 0;
                }
                if (l.type == CONNECTED && l.batch_normalize)
                {
                    denormalize_connected_layer(l);
                    net.layers[i].batch_normalize = 0;
                }
                if (l.type == GRU && l.batch_normalize)
                {
                    denormalize_connected_layer(*l.input_z_layer);
                    denormalize_connected_layer(*l.input_r_layer);
                    denormalize_connected_layer(*l.input_h_layer);
                    denormalize_connected_layer(*l.state_z_layer);
                    denormalize_connected_layer(*l.state_r_layer);
                    denormalize_connected_layer(*l.state_h_layer);
                    l.input_z_layer.batch_normalize = 0;
                    l.input_r_layer.batch_normalize = 0;
                    l.input_h_layer.batch_normalize = 0;
                    l.state_z_layer.batch_normalize = 0;
                    l.state_r_layer.batch_normalize = 0;
                    l.state_h_layer.batch_normalize = 0;
                    net.layers[i].batch_normalize = 0;
                }
            }
            save_weights(net, outfile);
        }

        public static void visualize(string cfgfile, string weightfile)
        {
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            visualize_network(net);
            cvWaitKey(0);
        }

        #region yolo-file

        string[] voc_names = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

        void train_yolo(string cfgfile, string weightfile)
        {
            string train_images = "/data/voc/train.txt";
            string backup_directory = "/home/pjreddie/backup/";
            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            float avg_loss = -1;
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = net.batch * net.subdivisions;
            int i = *net.seen / imgs;
            data train, buffer;


            layer l = net.layers[net.n - 1];

            int side = l.side;
            int classes = l.classes;
            float jitter = l.jitter;

            list* plist = get_paths(train_images);
            //int N = plist.size;
            string[] paths = (string[])list_to_array(plist);

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.n = imgs;
            args.m = plist.size;
            args.classes = classes;
            args.jitter = jitter;
            args.num_boxes = side;
            args.d = &buffer;
            args.type = REGION_DATA;

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

                printf("Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            char buff[256];
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void print_yolo_detections(FILE** fps, string id, box* boxes, float[]* probs, int total, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < total; ++i)
            {
                float xmin = boxes[i].x - boxes[i].w / 2.;
                float xmax = boxes[i].x + boxes[i].w / 2.;
                float ymin = boxes[i].y - boxes[i].h / 2.;
                float ymax = boxes[i].y + boxes[i].h / 2.;

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
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            srand(time(0));

            string basec = "results/comp4_det_test_";
            //list *plist = get_paths("data/voc.2007.test");
            list* plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
            //list *plist = get_paths("data/voc.2012.test");
            string[] paths = (string[])list_to_array(plist);

            layer l = net.layers[net.n - 1];
            int classes = l.classes;

            int j;
            FILE** fps = (FILE**)calloc(classes, sizeof(FILE*));
            for (j = 0; j < classes; ++j)
            {
                char buff[1024];
                snprintf(buff, 1024, "%s%s.txt", basec, voc_names[j]);
                fps[j] = fopen(buff, "w");
            }
            box* boxes = (box*)calloc(l.side * l.side * l.n, sizeof(box));
            float[]*probs = (float[] *)calloc(l.side * l.side * l.n, sizeof(float[]));
            for (j = 0; j < l.side * l.side * l.n; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.size;
            int i = 0;
            int t;

            float thresh = .001;
            int nms = 1;
            float iou_thresh = .5;

            int nthreads = 8;
            image* val = (image*)calloc(nthreads, sizeof(image));
            image* val_resized = (image*)calloc(nthreads, sizeof(image));
            image* buf = (image*)calloc(nthreads, sizeof(image));
            image* buf_resized = (image*)calloc(nthreads, sizeof(image));
            pthread_t* thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.type = IMAGE_DATA;

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
                    float[] X = val_resized[t].data;
                    network_predict(net, X);
                    int w = val[t].w;
                    int h = val[t].h;
                    get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
                    if (nms) do_nms_sort(boxes, probs, l.side * l.side * l.n, classes, iou_thresh);
                    print_yolo_detections(fps, id, boxes, probs, l.side * l.side * l.n, classes, w, h);
                    free(id);
                    free_image(val[t]);
                    free_image(val_resized[t]);
                }
            }
            Console.Error.Write($"Total Detection Time: %f Seconds\n", (double)(time(0) - start));
        }

        void validate_yolo_recall(string cfgfile, string weightfile)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            srand(time(0));

            string basec = "results/comp4_det_test_";
            list* plist = get_paths("data/voc.2007.test");
            string[] paths = (string[])list_to_array(plist);

            layer l = net.layers[net.n - 1];
            int classes = l.classes;
            int side = l.side;

            int j, k;
            FILE** fps = (FILE**)calloc(classes, sizeof(FILE*));
            for (j = 0; j < classes; ++j)
            {
                char buff[1024];
                snprintf(buff, 1024, "%s%s.txt", basec, voc_names[j]);
                fps[j] = fopen(buff, "w");
            }
            box* boxes = (box*)calloc(side * side * l.n, sizeof(box));
            float[]*probs = (float[] *)calloc(side * side * l.n, sizeof(float[]));
            for (j = 0; j < side * side * l.n; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.size;
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
                image orig = load_image_color(path, 0, 0);
                image sized = resize_image(orig, net.w, net.h);
                string id = basecfg(path);
                network_predict(net, sized.data);
                get_detection_boxes(l, orig.w, orig.h, thresh, probs, boxes, 1);
                if (nms) do_nms(boxes, probs, side * side * l.n, 1, nms);

                char labelpath[4096];
                find_replace(path, "images", "labels", labelpath);
                find_replace(labelpath, "JPEGImages", "labels", labelpath);
                find_replace(labelpath, ".jpg", ".txt", labelpath);
                find_replace(labelpath, ".JPEG", ".txt", labelpath);

                int num_labels = 0;
                box_label* truth = read_boxes(labelpath, &num_labels);
                for (k = 0; k < side * side * l.n; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < num_labels; ++j)
                {
                    ++total;
                    box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                    float best_iou = 0;
                    for (k = 0; k < side * side * l.n; ++k)
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
            image** alphabet = load_alphabet();
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            detection_layer l = net.layers[net.n - 1];
            set_batch_network(&net, 1);
            srand(2222222);
            clock_t time;
            char buff[256];
            string input = buff;
            int j;
            float nms = .4;
            box* boxes = (box*)calloc(l.side * l.side * l.n, sizeof(box));
            float[]*probs = (float[] *)calloc(l.side * l.side * l.n, sizeof(float[]));
            for (j = 0; j < l.side * l.side * l.n; ++j) probs[j] = (float[])calloc(l.classes, sizeof(float[]));
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image sized = resize_image(im, net.w, net.h);
                float[] X = sized.data;
                time = clock();
                network_predict(net, X);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
                if (nms) do_nms_sort(boxes, probs, l.side * l.side * l.n, l.classes, nms);
                //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
                draw_detections(im, l.side * l.side * l.n, thresh, boxes, probs, voc_names, alphabet, 20);
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
                image l = get_image_from_stream(lcap);
                image r = get_image_from_stream(rcap);
                if (!l.w || !r.w) break;
                if (count % 100 == 0)
                {
                    shift = best_3d_shift_r(l, r, -l.h / 100, l.h / 100);
                    printf("%d\n", shift);
                }
                image ls = crop_image(l, (l.w - w) / 2, (l.h - h) / 2, w, h);
                image rs = crop_image(r, 105 + (r.w - w) / 2, (r.h - h) / 2 + shift, w, h);
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
            string train_images = "/data/imagenet/imagenet1k.train.list";
            string backup_directory = "/home/pjreddie/backup/";
            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            float avg_loss = -1;
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = net.batch * net.subdivisions;
            int i = *net.seen / imgs;
            data train, buffer;


            list* plist = get_paths(train_images);
            //int N = plist.size;
            string[] paths = (string[])list_to_array(plist);

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.scale = 4;
            args.paths = paths;
            args.n = imgs;
            args.m = plist.size;
            args.d = &buffer;
            args.type = SUPER_DATA;

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

                printf("Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
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
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void test_voxel(string cfgfile, string weightfile, string filename)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                resize_network(&net, im.w, im.h);
                printf("%d %d\n", im.w, im.h);

                float[] X = im.data;
                time = clock();
                network_predict(net, X);
                image outi = get_network_image(net);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
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
            string train_images = "/data/imagenet/imagenet1k.train.list";
            string backup_directory = "/home/pjreddie/backup/";
            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            float avg_loss = -1;
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = net.batch * net.subdivisions;
            int i = *net.seen / imgs;
            data train, buffer;


            list* plist = get_paths(train_images);
            //int N = plist.size;
            string[] paths = (string[])list_to_array(plist);

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.scale = 4;
            args.paths = paths;
            args.n = imgs;
            args.m = plist.size;
            args.d = &buffer;
            args.type = SUPER_DATA;

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

                printf("Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
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
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void test_super(string cfgfile, string weightfile, string filename)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                resize_network(&net, im.w, im.h);
                printf("%d %d\n", im.w, im.h);

                float[] X = im.data;
                time = clock();
                network_predict(net, X);
                image outi = get_network_image(net);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
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
            string train_images = option_find_str(options, "train", "data/train.list");
            string backup_directory = option_find_str(options, "backup", "/backup/");

            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            float avg_loss = -1;
            network* nets = calloc(ngpus, sizeof(network));

            srand(time(0));
            int seed = rand();
            int i;
            for (i = 0; i < ngpus; ++i)
            {
                srand(seed);
                cuda_set_device(gpus[i]);
                nets[i] = parse_network_cfg(cfgfile);
                if (weightfile)
                {
                    load_weights(&nets[i], weightfile);
                }
                if (clear) *nets[i].seen = 0;
                nets[i].learning_rate *= ngpus;
            }
            srand(time(0));
            network net = nets[0];

            int imgs = net.batch * net.subdivisions * ngpus;
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            data train, buffer;

            layer l = net.layers[net.n - 1];

            int classes = l.classes;
            float jitter = l.jitter;

            list* plist = get_paths(train_images);
            //int N = plist.size;
            string[] paths = (string[])list_to_array(plist);

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.n = imgs;
            args.m = plist.size;
            args.classes = classes;
            args.jitter = jitter;
            args.num_boxes = l.max_boxes;
            args.d = &buffer;
            args.type = DETECTION_DATA;
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
                    printf("Resizing\n");
                    int dim = (rand() % 10 + 10) * 32;
                    if (get_current_batch(net) + 100 > net.max_batches) dim = 544;
                    //int dim = (rand() % 4 + 16) * 32;
                    printf("%d\n", dim);
                    args.w = dim;
                    args.h = dim;

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

                printf("Loaded: %lf seconds\n", sec(clock() - time));

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
                printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    if (ngpus != 1) sync_nets(nets, ngpus, 0);
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
                free_data(train);
            }
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
            char buff[256];
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }


        static int get_coco_image_id(string filename)
        {
            string p = strrchr(filename, '_');
            return atoi(p + 1);
        }

        static void print_cocos(FILE* fp, string image_path, box* boxes, float[]* probs, int num_boxes, int classes, int w, int h)
        {
            int i, j;
            int image_id = get_coco_image_id(image_path);
            for (i = 0; i < num_boxes; ++i)
            {
                float xmin = boxes[i].x - boxes[i].w / 2.;
                float xmax = boxes[i].x + boxes[i].w / 2.;
                float ymin = boxes[i].y - boxes[i].h / 2.;
                float ymax = boxes[i].y + boxes[i].h / 2.;

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
                float xmin = boxes[i].x - boxes[i].w / 2.;
                float xmax = boxes[i].x + boxes[i].w / 2.;
                float ymin = boxes[i].y - boxes[i].h / 2.;
                float ymax = boxes[i].y + boxes[i].h / 2.;

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
                float xmin = boxes[i].x - boxes[i].w / 2.;
                float xmax = boxes[i].x + boxes[i].w / 2.;
                float ymin = boxes[i].y - boxes[i].h / 2.;
                float ymax = boxes[i].y + boxes[i].h / 2.;

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
            string valid_images = option_find_str(options, "valid", "data/train.list");
            string name_list = option_find_str(options, "names", "data/names.list");
            string prefix = option_find_str(options, "results", "results");
            string[] names = get_labels(name_list);
            string mapf = option_find_str(options, "map", 0);
            int[] map = 0;
            if (mapf) map = read_map(mapf);

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            srand(time(0));

            string basec = "comp4_det_test_";
            list* plist = get_paths(valid_images);
            string[] paths = (string[])list_to_array(plist);

            layer l = net.layers[net.n - 1];
            int classes = l.classes;

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


            box* boxes = calloc(l.w * l.h * l.n, sizeof(box));
            float[]*probs = calloc(l.w * l.h * l.n, sizeof(float[]));
            for (j = 0; j < l.w * l.h * l.n; ++j) probs[j] = calloc(classes, sizeof(float[]));

            int m = plist.size;
            int i = 0;
            int t;

            float thresh = .005;
            float nms = .45;

            int nthreads = 4;
            image* val = calloc(nthreads, sizeof(image));
            image* val_resized = calloc(nthreads, sizeof(image));
            image* buf = calloc(nthreads, sizeof(image));
            image* buf_resized = calloc(nthreads, sizeof(image));
            pthread_t* thr = calloc(nthreads, sizeof(pthread_t));

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.type = IMAGE_DATA;

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
                    float[] X = val_resized[t].data;
                    network_predict(net, X);
                    int w = val[t].w;
                    int h = val[t].h;
                    get_region_boxes(l, w, h, thresh, probs, boxes, 0, map);
                    if (nms) do_nms_sort(boxes, probs, l.w * l.h * l.n, classes, nms);
                    if (coco)
                    {
                        print_cocos(fp, path, boxes, probs, l.w * l.h * l.n, classes, w, h);
                    }
                    else if (imagenet)
                    {
                        print_imagenet_detections(fp, i + t - nthreads + 1, boxes, probs, l.w * l.h * l.n, classes, w, h);
                    }
                    else
                    {
                        print_detector_detections(fps, id, boxes, probs, l.w * l.h * l.n, classes, w, h);
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
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            srand(time(0));

            list* plist = get_paths("data/voc.2007.test");
            string[] paths = (string[])list_to_array(plist);

            layer l = net.layers[net.n - 1];
            int classes = l.classes;

            int j, k;
            box* boxes = calloc(l.w * l.h * l.n, sizeof(box));
            float[]*probs = calloc(l.w * l.h * l.n, sizeof(float[]));
            for (j = 0; j < l.w * l.h * l.n; ++j) probs[j] = calloc(classes, sizeof(float[]));

            int m = plist.size;
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
                image orig = load_image_color(path, 0, 0);
                image sized = resize_image(orig, net.w, net.h);
                string id = basecfg(path);
                network_predict(net, sized.data);
                get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0);
                if (nms) do_nms(boxes, probs, l.w * l.h * l.n, 1, nms);

                char labelpath[4096];
                find_replace(path, "images", "labels", labelpath);
                find_replace(labelpath, "JPEGImages", "labels", labelpath);
                find_replace(labelpath, ".jpg", ".txt", labelpath);
                find_replace(labelpath, ".JPEG", ".txt", labelpath);

                int num_labels = 0;
                box_label* truth = read_boxes(labelpath, &num_labels);
                for (k = 0; k < l.w * l.h * l.n; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < num_labels; ++j)
                {
                    ++total;
                    box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                    float best_iou = 0;
                    for (k = 0; k < l.w * l.h * l.n; ++k)
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
            string name_list = option_find_str(options, "names", "data/names.list");
            string[] names = get_labels(name_list);

            image** alphabet = load_alphabet();
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image sized = resize_image(im, net.w, net.h);
                layer l = net.layers[net.n - 1];

                box* boxes = calloc(l.w * l.h * l.n, sizeof(box));
                float[]*probs = calloc(l.w * l.h * l.n, sizeof(float[]));
                for (j = 0; j < l.w * l.h * l.n; ++j) probs[j] = calloc(l.classes, sizeof(float[]));

                float[] X = sized.data;
                time = clock();
                network_predict(net, X);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
                if (nms) do_nms_sort(boxes, probs, l.w * l.h * l.n, l.classes, nms);
                draw_detections(im, l.w * l.h * l.n, thresh, boxes, probs, names, alphabet, l.classes);
                save_image(im, "predictions");
                show_image(im, "predictions");

                free_image(im);
                free_image(sized);
                free(boxes);
                free_ptrs((void**)probs, l.w * l.h * l.n);
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
                printf("%s\n", gpu_list);
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
                    gpus[i] = atoi(gpu_list);
                    gpu_list = strchr(gpu_list, ',') + 1;
                }
            }
            else
            {
                gpu = gpu_index;
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
                string name_list = option_find_str(options, "names", "data/names.list");
                string[] names = get_labels(name_list);
                demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix);
            }
        }

        #endregion

        #region CiFarFile

        void train_cifar(string cfgfile, string weightfile)
        {
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

            string backup_directory = "/home/pjreddie/backup/";
            int classes = 10;
            int N = 50000;

            string[] labels = get_labels("data/cifar/labels.txt");
            int epoch = (*net.seen) / N;
            data train = load_all_cifar10();
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                clock_t time = clock();

                float loss = train_network_sgd(net, train, 1);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95 + loss * .05;
                printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
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
            sprintf(buff, "%s/%s.weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free_ptrs((void**)labels, classes);
            free(basec);
            free_data(train);
        }

        void train_cifar_distill(string cfgfile, string weightfile)
        {
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

            string backup_directory = "/home/pjreddie/backup/";
            int classes = 10;
            int N = 50000;

            string[] labels = get_labels("data/cifar/labels.txt");
            int epoch = (*net.seen) / N;

            data train = load_all_cifar10();
            matrix soft = csv_to_matrix("results/ensemble.csv");

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
                printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
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
            sprintf(buff, "%s/%s.weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free_ptrs((void**)labels, classes);
            free(basec);
            free_data(train);
        }

        void test_cifar_multi(string filename, string weightfile)
        {
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            srand(time(0));

            float avg_acc = 0;
            data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

            int i;
            for (i = 0; i < test.X.rows; ++i)
            {
                image im = float_to_image(32, 32, 3, test.X.vals[i]);

                float pred[10] = { 0 };

                float[] p = network_predict(net, im.data);
                axpy_cpu(10, 1, p, 1, pred, 1);
                flip_image(im);
                p = network_predict(net, im.data);
                axpy_cpu(10, 1, p, 1, pred, 1);

                int index = max_index(pred, 10);
                int sclass = max_index(test.y.vals[i], 10);
                if (index == sclass) avg_acc += 1;
                free_image(im);
                printf("%4d: %.2f%%\n", i, 100.* avg_acc / (i + 1));
            }
        }

        void test_cifar(string filename, string weightfile)
        {
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            clock_t time;
            float avg_acc = 0;
            float avg_top5 = 0;
            data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

            time = clock();

            float[] acc = network_accuracies(net, test, 2);
            avg_acc += acc[0];
            avg_top5 += acc[1];
            printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock() - time), test.X.rows);
            free_data(test);
        }

        void extract_cifar()
        {
            string labels[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
            int i;
            data train = load_all_cifar10();
            data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");
            for (i = 0; i < train.X.rows; ++i)
            {
                image im = float_to_image(32, 32, 3, train.X.vals[i]);
                int sclass = max_index(train.y.vals[i], 10);
                char buff[256];
                sprintf(buff, "data/cifar/train/%d_%s", i, labels[sclass]);
                save_image_png(im, buff);
            }
            for (i = 0; i < test.X.rows; ++i)
            {
                image im = float_to_image(32, 32, 3, test.X.vals[i]);
                int sclass = max_index(test.y.vals[i], 10);
                char buff[256];
                sprintf(buff, "data/cifar/test/%d_%s", i, labels[sclass]);
                save_image_png(im, buff);
            }
        }

        void test_cifar_csv(string filename, string weightfile)
        {
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

            matrix pred = network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.rows; ++i)
            {
                image im = float_to_image(32, 32, 3, test.X.vals[i]);
                flip_image(im);
            }
            matrix pred2 = network_predict_data(net, test);
            scale_matrix(pred, .5);
            scale_matrix(pred2, .5);
            matrix_add_matrix(pred2, pred);

            matrix_to_csv(pred);
            Console.Error.Write($"Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
            free_data(test);
        }

        void test_cifar_csvtrain(string filename, string weightfile)
        {
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            data test = load_all_cifar10();

            matrix pred = network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.rows; ++i)
            {
                image im = float_to_image(32, 32, 3, test.X.vals[i]);
                flip_image(im);
            }
            matrix pred2 = network_predict_data(net, test);
            scale_matrix(pred, .5);
            scale_matrix(pred2, .5);
            matrix_add_matrix(pred2, pred);

            matrix_to_csv(pred);
            Console.Error.Write($"Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
            free_data(test);
        }

        void eval_cifar_csv()
        {
            data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

            matrix pred = csv_to_matrix("results/combined.csv");
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
            m.n = 128;
            m.data = calloc(128, sizeof(string));
            FILE* fp = fopen(filename, "rb");
            int count = 0;
            string line = 0;
            while ((line = fgetgo(fp)))
            {
                if (count >= m.n)
                {
                    m.n *= 2;
                    m.data = realloc(m.data, m.n * sizeof(string));
                }
                m.data[count] = line;
                ++count;
            }
            printf("%d\n", count);
            m.n = count;
            m.data = realloc(m.data, count * sizeof(string));
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
                string b = m.data[rand() % m.n];
                int row = b[0];
                int col = b[1];
                labels[col + 19 * (row + i * 19)] = 1;
                string_to_board(b + 2, boards + i * 19 * 19);
                boards[col + 19 * (row + i * 19)] = 0;

                int flip = rand() % 2;
                int rotate = rand() % 4;
                image ini = float_to_image(19, 19, 1, boards + i * 19 * 19);
                image outi = float_to_image(19, 19, 1, labels + i * 19 * 19);
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
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

            string backup_directory = "/home/pjreddie/backup/";

            char buff[256];
            float[] board = calloc(19 * 19 * net.batch, sizeof(float));
            float[] move = calloc(19 * 19 * net.batch, sizeof(float));
            moves m = load_go_moves("/home/pjreddie/backup/go.train");
            //moves m = load_go_moves("games.txt");

            int N = m.n;
            int epoch = (*net.seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                clock_t time = clock();

                random_go_moves(m, board, move, net.batch);
                float loss = train_network_datum(net, board, move) / net.batch;
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .95 + loss * .05;
                printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
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
            sprintf(buff, "%s/%s.weights", backup_directory, basec);
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

        void predict_move(network net, float[] board, float[] move, int multi)
        {
            float[] output = network_predict(net, board);
            copy_cpu(19 * 19, output, 1, move, 1);
            int i;
            if (multi)
            {
                image bim = float_to_image(19, 19, 1, board);
                for (i = 1; i < 8; ++i)
                {
                    rotate_image_cw(bim, i);
                    if (i >= 4) flip_image(bim);

                    float[] output = network_predict(net, board);
                    image oim = float_to_image(19, 19, 1, output);

                    if (i >= 4) flip_image(oim);
                    rotate_image_cw(oim, -i);

                    axpy_cpu(19 * 19, 1, output, 1, move, 1);

                    if (i >= 4) flip_image(bim);
                    rotate_image_cw(bim, -i);
                }
                scal_cpu(19 * 19, 1./ 8., move, 1);
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

        int generate_move(network net, int player, float[] board, int multi, float thresh, float temp, string ko, int print)
        {
            int i, j;
            for (i = 0; i < net.n; ++i) net.layers[i].temperature = temp;

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
            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

            float[] board = calloc(19 * 19, sizeof(float));
            float[] move = calloc(19 * 19, sizeof(float));
            moves m = load_go_moves("/home/pjreddie/backup/go.test");

            int N = m.n;
            int i;
            int correct = 0;
            for (i = 0; i < N; ++i)
            {
                string b = m.data[i];
                int row = b[0];
                int col = b[1];
                int truth = col + 19 * row;
                string_to_board(b + 2, board);
                predict_move(net, board, move, multi);
                int index = max_index(move, 19 * 19);
                if (index == truth) ++correct;
                printf("%d Accuracy %f\n", i, (float)correct / (i + 1));
            }
        }

        void engine_go(string filename, string weightfile, int multi)
        {
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));
            set_batch_network(&net, 1);
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
                    printf("=%s 2\n\n", ids);
                }
                else if (!strcmp(buff, "name"))
                {
                    printf("=%s DarkGo\n\n", ids);
                }
                else if (!strcmp(buff, "version"))
                {
                    printf("=%s 1.0\n\n", ids);
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
                    if (known) printf("=%s true\n\n", ids);
                    else printf("=%s false\n\n", ids);
                }
                else if (!strcmp(buff, "list_commands"))
                {
                    printf("=%s protocol_version\nname\nversion\nknown_command\nlist_commands\nquit\nboardsize\nclear_board\nkomi\nplay\ngenmove\nfinal_status_list\n\n", ids);
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
                        printf("?%s unacceptable size\n\n", ids);
                    }
                    else
                    {
                        printf("=%s \n\n", ids);
                    }
                }
                else if (!strcmp(buff, "clear_board"))
                {
                    passed = 0;
                    memset(board, 0, 19 * 19 * sizeof(float));
                    printf("=%s \n\n", ids);
                }
                else if (!strcmp(buff, "komi"))
                {
                    float komi = 0;
                    scanf("%f", &komi);
                    printf("=%s \n\n", ids);
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
                        printf("=%s \n\n", ids);
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

                    printf("=%s \n\n", ids);
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
                        printf("=%s pass\n\n", ids);
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
                        printf("=%s %c%d\n\n", ids, 'A' + col, row);
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
                            printf("%s\n", l);
                            free(l);
                        }
                    }
                    else
                    {
                        printf("?%s unknown command\n\n", ids);
                    }
                }
                else
                {
                    string line = fgetl(stdin);
                    free(line);
                    printf("?%s unknown command\n\n", ids);
                }
                fflush(stdout);
                fflush(stderr);
            }
        }

        void test_go(string cfg, string weights, int multi)
        {
            network net = parse_network_cfg(cfg);
            if (weights)
            {
                load_weights(&net, weights);
            }
            srand(time(0));
            set_batch_network(&net, 1);
            float[] board = calloc(19 * 19, sizeof(float));
            float[] move = calloc(19 * 19, sizeof(float));
            int color = 1;
            while (1)
            {
                float[] output = network_predict(net, board);
                copy_cpu(19 * 19, output, 1, move, 1);
                int i;
                if (multi)
                {
                    image bim = float_to_image(19, 19, 1, board);
                    for (i = 1; i < 8; ++i)
                    {
                        rotate_image_cw(bim, i);
                        if (i >= 4) flip_image(bim);

                        float[] output = network_predict(net, board);
                        image oim = float_to_image(19, 19, 1, output);

                        if (i >= 4) flip_image(oim);
                        rotate_image_cw(oim, -i);

                        axpy_cpu(19 * 19, 1, output, 1, move, 1);

                        if (i >= 4) flip_image(bim);
                        rotate_image_cw(bim, -i);
                    }
                    scal_cpu(19 * 19, 1./ 8., move, 1);
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
                    printf("%d: %c %d, %.2f%%\n", i + 1, col + 'A' + 1 * (col > 7 && noi), (inverted) ? 19 - row : row + 1, move[index] * 100);
                }
                //if(color == 1) printf("\u25EF Enter move: ");
                //else printf("\u25C9 Enter move: ");
                if (color == 1) printf("X Enter move: ");
                else printf("O Enter move: ");

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
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }

            network net2 = net;
            if (f2)
            {
                net2 = parse_network_cfg(f2);
                if (w2)
                {
                    load_weights(&net2, w2);
                }
            }
            srand(time(0));
            //char boards[300][93];
            char boards[300, 93];
            int count = 0;
            set_batch_network(&net, 1);
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
                            printf("%c", boards[i][j]);
                        }
                        printf("\n");
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
                network use = ((total % 2 == 0) == (player == 1)) ? net : net2;
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

        void reset_rnn_state(network net, int b)
        {
            int i;
            for (i = 0; i < net.n; ++i)
            {
                layer l = net.layers[i];
                if (l.state_gpu)
                {
                    fill_ongpu(l.outputs, 0, l.state_gpu + l.outputs * b, 1);
                }
            }
        }

        void train_char_rnn(string cfgfile, string weightfile, string filename, int clear, int tokenized)
        {
            srand(time(0));
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
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }

            int inputs = get_network_input_size(net);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int batch = net.batch;
            int steps = net.time_steps;
            if (clear) *net.seen = 0;
            int i = (*net.seen) / net.batch;

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
                    //printf("%d\n", j);
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
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
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
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void print_symbol(int n, string[] tokens)
        {
            if (tokens)
            {
                printf("%s ", tokens[n]);
            }
            else
            {
                printf("%c", n);
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

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.n; ++i) net.layers[i].temperature = temp;
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
                    //printf("%d %c %f\n",j, j, outf[j]);
                }
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                c = sample_array(outf, inputs);
                print_symbol(c, tokens);
            }
            printf("\n");
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

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            int inputs = get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.n; ++i) net.layers[i].temperature = temp;
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
            printf("\n");
        }

        void valid_tactic_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
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
                printf("%d %d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, words, pow(2, -sum / count), pow(2, -sum / words));
            }
        }

        void valid_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
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
                printf("%d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, pow(2, -sum / count), pow(2, -sum / words));
            }
        }

        void vec_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
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

                layer l = net.layers[0];
                cuda_pull_array(l.output_gpu, l.output, l.outputs);
                printf("%s", line);
                for (i = 0; i < l.outputs; ++i)
                {
                    printf(",%g", l.output[i]);
                }
                printf("\n");
            }
        }

        public static void run_char_rnn(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }
            string filename = find_char_arg(args.Count, args, "-file", "data/shakespeare.txt");
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


        float_pair get_rnn_vid_data(network net, string[] files, int n, int batch, int steps)
        {
            int b;
            assert(net.batch == steps + 1);
            image out_im = get_network_image(net);
            int output_size = out_im.w * out_im.h * out_im.c;
            printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
            float[] feats = calloc(net.batch * batch * output_size, sizeof(float));
            for (b = 0; b < batch; ++b)
            {
                int input_size = net.w * net.h * net.c;
                float[] input = calloc(input_size * net.batch, sizeof(float));
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

                printf("frames: %d, index: %d\n", frames, index);
                cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, index);

                int i;
                for (i = 0; i < net.batch; ++i)
                {
                    IplImage* src = cvQueryFrame(cap);
                    image im = ipl_to_image(src);
                    rgbgr_image(im);
                    image re = resize_image(im, net.w, net.h);
                    //show_image(re, "loaded");
                    //cvWaitKey(10);
                    memcpy(input + i * input_size, re.data, input_size * sizeof(float));
                    free_image(im);
                    free_image(re);
                }
                float[] output = network_predict(net, input);

                free(input);

                for (i = 0; i < net.batch; ++i)
                {
                    memcpy(feats + (b + i * batch) * output_size, output + i * output_size, output_size * sizeof(float));
                }

                cvReleaseCapture(&cap);
            }

            //printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
            float_pair p = { 0 };
            p.x = feats;
            p.y = feats + output_size * batch; //+ out_im.w*out_im.h*out_im.c;

            return p;
        }


        void train_vid_rnn(string cfgfile, string weightfile)
        {
            string train_videos = "data/vid/train.txt";
            string backup_directory = "/home/pjreddie/backup/";
            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            float avg_loss = -1;
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = net.batch * net.subdivisions;
            int i = *net.seen / imgs;

            list* plist = get_paths(train_videos);
            int N = plist.size;
            string[] paths = (string[])list_to_array(plist);
            clock_t time;
            int steps = net.time_steps;
            int batch = net.batch / net.time_steps;

            network extractor = parse_network_cfg("cfg/extractor.cfg");
            load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

            while (get_current_batch(net) < net.max_batches)
            {
                i += 1;
                time = clock();
                float_pair p = get_rnn_vid_data(extractor, paths, N, batch, steps);

                float loss = train_network_datum(net, p.x, p.y) / (net.batch);


                free(p.x);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time));
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
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
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }


        image save_reconstruction(network net, image* init, float[] feat, string name, int i)
        {
            image recon;
            if (init)
            {
                recon = copy_image(*init);
            }
            else
            {
                recon = make_random_image(net.w, net.h, 3);
            }

            image update = make_image(net.w, net.h, 3);
            reconstruct_picture(net, feat, recon, update, .01, .9, .1, 2, 50);
            char buff[256];
            sprintf(buff, "%s%d", name, i);
            save_image(recon, buff);
            free_image(update);
            return recon;
        }

        void generate_vid_rnn(string cfgfile, string weightfile)
        {
            network extractor = parse_network_cfg("cfg/extractor.recon.cfg");
            load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&extractor, 1);
            set_batch_network(&net, 1);

            int i;
            CvCapture* cap = cvCaptureFromFile("/extra/vid/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00007030.mp4");
            float[] feat;
            float[] next;
            next = NULL;
            image last;
            for (i = 0; i < 25; ++i)
            {
                image im = get_image_from_stream(cap);
                image re = resize_image(im, extractor.w, extractor.h);
                feat = network_predict(extractor, re.data);
                if (i > 0)
                {
                    printf("%f %f\n", mean_array(feat, 14 * 14 * 512), variance_array(feat, 14 * 14 * 512));
                    printf("%f %f\n", mean_array(next, 14 * 14 * 512), variance_array(next, 14 * 14 * 512));
                    printf("%f\n", mse_array(feat, 14 * 14 * 512));
                    axpy_cpu(14 * 14 * 512, -1, feat, 1, next, 1);
                    printf("%f\n", mse_array(next, 14 * 14 * 512));
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
                image newi = save_reconstruction(extractor, &last, next, "new", i);
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
            //char *train_images = "/home/pjreddie/data/voc/test/train.txt";
            //char *train_images = "/home/pjreddie/data/coco/train.txt";
            string train_images = "data/coco.trainval.txt";
            //char *train_images = "data/bags.train.list";
            string backup_directory = "/home/pjreddie/backup/";
            srand(time(0));
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            float avg_loss = -1;
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = net.batch * net.subdivisions;
            int i = *net.seen / imgs;
            data train, buffer;


            layer l = net.layers[net.n - 1];

            int side = l.side;
            int classes = l.classes;
            float jitter = l.jitter;

            list* plist = get_paths(train_images);
            //int N = plist.size;
            string[] paths = (string[])list_to_array(plist);

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.n = imgs;
            args.m = plist.size;
            args.classes = classes;
            args.jitter = jitter;
            args.num_boxes = side;
            args.d = &buffer;
            args.type = REGION_DATA;

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

                printf("Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss < 0) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;

                printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
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
            sprintf(buff, "%s/%s_final.weights", backup_directory, basec);
            save_weights(net, buff);
        }

        void print_cocos(FILE* fp, int image_id, box* boxes, float[]* probs, int num_boxes, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < num_boxes; ++i)
            {
                float xmin = boxes[i].x - boxes[i].w / 2.;
                float xmax = boxes[i].x + boxes[i].w / 2.;
                float ymin = boxes[i].y - boxes[i].h / 2.;
                float ymax = boxes[i].y + boxes[i].h / 2.;

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
            return atoi(p + 1);
        }

        void validate_coco(string cfgfile, string weightfile)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            srand(time(0));

            string basec = "results/";
            list* plist = get_paths("data/coco_val_5k.list");
            //list *plist = get_paths("/home/pjreddie/data/people-art/test.txt");
            //list *plist = get_paths("/home/pjreddie/data/voc/test/2007_test.txt");
            string[] paths = (string[])list_to_array(plist);

            layer l = net.layers[net.n - 1];
            int classes = l.classes;
            int side = l.side;

            int j;
            char buff[1024];
            snprintf(buff, 1024, "%s/coco_results.json", basec);
            FILE* fp = fopen(buff, "w");
            fprintf(fp, "[\n");

            box* boxes = (box*)calloc(side * side * l.n, sizeof(box));
            float[]*probs = (float[] *)calloc(side * side * l.n, sizeof(float[]));
            for (j = 0; j < side * side * l.n; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.size;
            int i = 0;
            int t;

            float thresh = .01;
            int nms = 1;
            float iou_thresh = .5;

            int nthreads = 8;
            image* val = (image*)calloc(nthreads, sizeof(image));
            image* val_resized = (image*)calloc(nthreads, sizeof(image));
            image* buf = (image*)calloc(nthreads, sizeof(image));
            image* buf_resized = (image*)calloc(nthreads, sizeof(image));
            pthread_t* thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.type = IMAGE_DATA;

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
                    float[] X = val_resized[t].data;
                    network_predict(net, X);
                    int w = val[t].w;
                    int h = val[t].h;
                    get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
                    if (nms) do_nms_sort(boxes, probs, side * side * l.n, classes, iou_thresh);
                    print_cocos(fp, image_id, boxes, probs, side * side * l.n, classes, w, h);
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
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            srand(time(0));

            string basec = "results/comp4_det_test_";
            list* plist = get_paths("/home/pjreddie/data/voc/test/2007_test.txt");
            string[] paths = (string[])list_to_array(plist);

            layer l = net.layers[net.n - 1];
            int classes = l.classes;
            int side = l.side;

            int j, k;
            FILE** fps = (FILE**)calloc(classes, sizeof(FILE*));
            for (j = 0; j < classes; ++j)
            {
                char buff[1024];
                snprintf(buff, 1024, "%s%s.txt", basec, coco_classes[j]);
                fps[j] = fopen(buff, "w");
            }
            box* boxes = (box*)calloc(side * side * l.n, sizeof(box));
            float[]*probs = (float[] *)calloc(side * side * l.n, sizeof(float[]));
            for (j = 0; j < side * side * l.n; ++j) probs[j] = (float[])calloc(classes, sizeof(float[]));

            int m = plist.size;
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
                image orig = load_image_color(path, 0, 0);
                image sized = resize_image(orig, net.w, net.h);
                string id = basecfg(path);
                network_predict(net, sized.data);
                get_detection_boxes(l, 1, 1, thresh, probs, boxes, 1);
                if (nms) do_nms(boxes, probs, side * side * l.n, 1, nms_thresh);

                char labelpath[4096];
                find_replace(path, "images", "labels", labelpath);
                find_replace(labelpath, "JPEGImages", "labels", labelpath);
                find_replace(labelpath, ".jpg", ".txt", labelpath);
                find_replace(labelpath, ".JPEG", ".txt", labelpath);

                int num_labels = 0;
                box_label* truth = read_boxes(labelpath, &num_labels);
                for (k = 0; k < side * side * l.n; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < num_labels; ++j)
                {
                    ++total;
                    box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                    float best_iou = 0;
                    for (k = 0; k < side * side * l.n; ++k)
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
            image** alphabet = load_alphabet();
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            detection_layer l = net.layers[net.n - 1];
            set_batch_network(&net, 1);
            srand(2222222);
            float nms = .4;
            clock_t time;
            char buff[256];
            string input = buff;
            int j;
            box* boxes = (box*)calloc(l.side * l.side * l.n, sizeof(box));
            float[]*probs = (float[] *)calloc(l.side * l.side * l.n, sizeof(float[]));
            for (j = 0; j < l.side * l.side * l.n; ++j) probs[j] = (float[])calloc(l.classes, sizeof(float[]));
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image sized = resize_image(im, net.w, net.h);
                float[] X = sized.data;
                time = clock();
                network_predict(net, X);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
                if (nms) do_nms_sort(boxes, probs, l.side * l.side * l.n, l.classes, nms);
                draw_detections(im, l.side * l.side * l.n, thresh, boxes, probs, coco_classes, alphabet, 80);
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
                v[i] = atof(p + 1);
            }
            return v;
        }

        void train_classifier(string datacfg, string cfgfile, string weightfile, int[] gpus, int ngpus, int clear)
        {
            int i;

            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            printf("%d\n", ngpus);
            network* nets = calloc(ngpus, sizeof(network));

            srand(time(0));
            int seed = rand();
            for (i = 0; i < ngpus; ++i)
            {
                srand(seed);
# ifdef GPU
                cuda_set_device(gpus[i]);
#endif
                nets[i] = parse_network_cfg(cfgfile);
                if (weightfile)
                {
                    load_weights(&nets[i], weightfile);
                }
                if (clear) *nets[i].seen = 0;
                nets[i].learning_rate *= ngpus;
            }
            srand(time(0));
            network net = nets[0];

            int imgs = net.batch * net.subdivisions * ngpus;

            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            list* options = read_data_cfg(datacfg);

            string backup_directory = option_find_str(options, "backup", "/backup/");
            string label_list = option_find_str(options, "labels", "data/labels.list");
            string train_list = option_find_str(options, "train", "data/train.list");
            int classes = option_find_int(options, "classes", 2);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(train_list);
            string[] paths = (string[])list_to_array(plist);
            printf("%d\n", plist.size);
            int N = plist.size;
            clock_t time;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.threads = 32;
            args.hierarchy = net.hierarchy;

            args.min = net.min_crop;
            args.max = net.max_crop;
            args.angle = net.angle;
            args.aspect = net.aspect;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;
            args.size = net.w;

            args.paths = paths;
            args.classes = classes;
            args.n = imgs;
            args.m = N;
            args.labels = labels;
            args.type = CLASSIFICATION_DATA;

            data train;
            data buffer;
            pthread_t load_thread;
            args.d = &buffer;
            load_thread = load_data(args);

            int epoch = (*net.seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                time = clock();

                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data(args);

                printf("Loaded: %lf seconds\n", sec(clock() - time));
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
                printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
                free_data(train);
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
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
            sprintf(buff, "%s/%s.weights", backup_directory, basec);
            save_weights(net, buff);

            free_network(net);
            free_ptrs((void**)labels, classes);
            free_ptrs((void**)paths, plist.size);
            free_list(plist);
            free(basec);
        }


        void validate_classifier_crop(string datacfg, string filename, string weightfile)
        {
            int i = 0;
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "data/labels.list");
            string valid_list = option_find_str(options, "valid", "data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
            free_list(plist);

            clock_t time;
            float avg_acc = 0;
            float avg_topk = 0;
            int splits = m / 1000;
            int num = (i + 1) * m / splits - i * m / splits;

            data val, buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;

            args.paths = paths;
            args.classes = classes;
            args.n = num;
            args.m = 0;
            args.labels = labels;
            args.d = &buffer;
            args.type = OLD_CLASSIFICATION_DATA;

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
                printf("Loaded: %d images ini %lf seconds\n", val.X.rows, sec(clock() - time));

                time = clock();
                float[] acc = network_accuracies(net, val, topk);
                avg_acc += acc[0];
                avg_topk += acc[1];
                printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc / i, topk, avg_topk / i, sec(clock() - time), val.X.rows);
                free_data(val);
            }
        }

        void validate_classifier_10(string datacfg, string filename, string weightfile)
        {
            int i, j;
            network net = parse_network_cfg(filename);
            set_batch_network(&net, 1);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "data/labels.list");
            string valid_list = option_find_str(options, "valid", "data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
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
                int w = net.w;
                int h = net.h;
                int shift = 32;
                image im = load_image_color(paths[i], w + shift, h + shift);
                image images[10];
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
                    float[] p = network_predict(net, images[j].data);
                    if (net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
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

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        void validate_classifier_full(string datacfg, string filename, string weightfile)
        {
            int i, j;
            network net = parse_network_cfg(filename);
            set_batch_network(&net, 1);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "data/labels.list");
            string valid_list = option_find_str(options, "valid", "data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
            free_list(plist);

            float avg_acc = 0;
            float avg_topk = 0;
            int[] indexes = calloc(topk, sizeof(int));

            int size = net.w;
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
                image im = load_image_color(paths[i], 0, 0);
                image resized = resize_min(im, size);
                resize_network(&net, resized.w, resized.h);
                float[] pred = network_predict(net, resized.data);
                if (net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);

                free_image(im);
                free_image(resized);
                top_k(pred, classes, topk, indexes);

                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }


        void validate_classifier_single(string datacfg, string filename, string weightfile)
        {
            int i, j;
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "data/labels.list");
            string leaf_list = option_find_str(options, "leaves", 0);
            if (leaf_list) change_leaves(net.hierarchy, leaf_list);
            string valid_list = option_find_str(options, "valid", "data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
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
                image im = load_image_color(paths[i], 0, 0);
                image resized = resize_min(im, net.w);
                image crop = crop_image(resized, (resized.w - net.w) / 2, (resized.h - net.h) / 2, net.w, net.h);
                //show_image(im, "orig");
                //show_image(crop, "cropped");
                //cvWaitKey(0);
                float[] pred = network_predict(net, crop.data);
                if (net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);

                if (resized.data != im.data) free_image(resized);
                free_image(im);
                free_image(crop);
                top_k(pred, classes, topk, indexes);

                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        void validate_classifier_multi(string datacfg, string filename, string weightfile)
        {
            int i, j;
            network net = parse_network_cfg(filename);
            set_batch_network(&net, 1);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "labels", "data/labels.list");
            string valid_list = option_find_str(options, "valid", "data/train.list");
            int classes = option_find_int(options, "classes", 2);
            int topk = option_find_int(options, "top", 1);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(valid_list);
            int scales[] = { 224, 288, 320, 352, 384 };
            int nscales = sizeof(scales) / sizeof(scales[0]);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
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
                image im = load_image_color(paths[i], 0, 0);
                for (j = 0; j < nscales; ++j)
                {
                    image r = resize_min(im, scales[j]);
                    resize_network(&net, r.w, r.h);
                    float[] p = network_predict(net, r.data);
                    if (net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
                    axpy_cpu(classes, 1, p, 1, pred, 1);
                    flip_image(r);
                    p = network_predict(net, r.data);
                    axpy_cpu(classes, 1, p, 1, pred, 1);
                    if (r.data != im.data) free_image(r);
                }
                free_image(im);
                top_k(pred, classes, topk, indexes);
                free(pred);
                if (indexes[0] == class2) avg_acc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avg_topk += 1;
                }

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
            }
        }

        void try_classifier(string datacfg, string cfgfile, string weightfile, string filename, int layer_num)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            srand(2222222);

            list* options = read_data_cfg(datacfg);

            string name_list = option_find_str(options, "names", 0);
            if (!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
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
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image orig = load_image_color(input, 0, 0);
                image r = resize_min(orig, 256);
                image im = crop_image(r, (r.w - 224 - 1) / 2 + 1, (r.h - 224 - 1) / 2 + 1, 224, 224);
                float mean[] = { 0.48263312050943, 0.45230225481413, 0.40099074308742 };
                float std[] = { 0.22590347483426, 0.22120921437787, 0.22103996251583 };
                float var[3];
                var[0] = std[0] * std[0];
                var[1] = std[1] * std[1];
                var[2] = std[2] * std[2];

                normalize_cpu(im.data, mean, var, 1, 3, im.w * im.h);

                float[] X = im.data;
                time = clock();
                float[] predictions = network_predict(net, X);

                layer l = net.layers[layer_num];
                for (i = 0; i < l.c; ++i)
                {
                    if (l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
                }
                cuda_pull_array(l.output_gpu, l.output, l.outputs);
                for (i = 0; i < l.outputs; ++i)
                {
                    printf("%f\n", l.output[i]);
                }

                top_predictions(net, top, indexes);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    printf("%s: %f\n", names[index], predictions[index]);
                }
                free_image(im);
                if (filename) break;
            }
        }

        public static void predict_classifier(string datacfg, string cfgfile, string weightfile, string filename, int top)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            srand(2222222);

            list* options = read_data_cfg(datacfg);

            string name_list = option_find_str(options, "names", 0);
            if (!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
            if (top == 0) top = option_find_int(options, "top", 1);

            int i = 0;
            string[] names = get_labels(name_list);
            clock_t time;
            int[] indexes = calloc(top, sizeof(int));
            char buff[256];
            string input = buff;
            int size = net.w;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image r = resize_min(im, size);
                resize_network(&net, r.w, r.h);
                printf("%d %d\n", r.w, r.h);

                float[] X = r.data;
                time = clock();
                float[] predictions = network_predict(net, X);
                if (net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
                top_k(predictions, net.outputs, top, indexes);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    if (net.hierarchy) printf("%d, %s: %f, parent: %s \n", index, names[index], predictions[index], (net.hierarchy.parent[index] >= 0) ? names[net.hierarchy.parent[index]] : "Root");
                    else printf("%s: %f\n", names[index], predictions[index]);
                }
                if (r.data != im.data) free_image(r);
                free_image(im);
                if (filename) break;
            }
        }


        void label_classifier(string datacfg, string filename, string weightfile)
        {
            int i;
            network net = parse_network_cfg(filename);
            set_batch_network(&net, 1);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string label_list = option_find_str(options, "names", "data/labels.list");
            string test_list = option_find_str(options, "test", "data/train.list");
            int classes = option_find_int(options, "classes", 2);

            string[] labels = get_labels(label_list);
            list* plist = get_paths(test_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
            free_list(plist);

            for (i = 0; i < m; ++i)
            {
                image im = load_image_color(paths[i], 0, 0);
                image resized = resize_min(im, net.w);
                image crop = crop_image(resized, (resized.w - net.w) / 2, (resized.h - net.h) / 2, net.w, net.h);
                float[] pred = network_predict(net, crop.data);

                if (resized.data != im.data) free_image(resized);
                free_image(im);
                free_image(crop);
                int ind = max_index(pred, classes);

                printf("%s\n", labels[ind]);
            }
        }


        void test_classifier(string datacfg, string cfgfile, string weightfile, int target_layer)
        {
            int curr = 0;
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* options = read_data_cfg(datacfg);

            string test_list = option_find_str(options, "test", "data/test.list");
            int classes = option_find_int(options, "classes", 2);

            list* plist = get_paths(test_list);

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
            free_list(plist);

            clock_t time;

            data val, buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.classes = classes;
            args.n = net.batch;
            args.m = 0;
            args.labels = 0;
            args.d = &buffer;
            args.type = OLD_CLASSIFICATION_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            for (curr = net.batch; curr < m; curr += net.batch)
            {
                time = clock();

                pthread_join(load_thread, 0);
                val = buffer;

                if (curr < m)
                {
                    args.paths = paths + curr;
                    if (curr + net.batch > m) args.n = m - curr;
                    load_thread = load_data_in_thread(args);
                }
                Console.Error.Write($"Loaded: %d images ini %lf seconds\n", val.X.rows, sec(clock() - time));

                time = clock();
                matrix pred = network_predict_data(net, val);

                int i, j;

                for (i = 0; i < pred.rows; ++i)
                {
                    printf("%s", paths[curr - net.batch + i]);
                    for (j = 0; j < pred.cols; ++j)
                    {
                        printf("\t%g", pred.vals[i][j]);
                    }
                    printf("\n");
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

            printf("Classifier Demo\n");
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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

                image ini = get_image_from_stream(cap);
                if (!ini.data) break;
                image in_s = resize_image(ini, net.w, net.h);

                image outo = ini;
                int x1 = outo.w / 20;
                int y1 = outo.h / 20;
                int x2 = 2 * x1;
                int y2 = outo.h - outo.h / 20;

                int border = .01 * outo.h;
                int h = y2 - y1 - 2 * border;
                int w = x2 - x1 - 2 * border;

                float[] predictions = network_predict(net, in_s.data);
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

                printf("\033[2J");
                printf("\033[1;1H");
                printf("\nFPS:%.0f\n", fps);

                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    printf("%.1f%%: %s\n", predictions[index] * 100, names[index]);
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

            printf("Classifier Demo\n");
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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

                image ini = get_image_from_stream(cap);
                image in_s = resize_image(ini, net.w, net.h);
                show_image(ini, "Threat Detection");

                float[] predictions = network_predict(net, in_s.data);
                top_predictions(net, top, indexes);

                printf("\033[2J");
                printf("\033[1;1H");

                int threat = 0;
                for (i = 0; i < sizeof(bad_cats) / sizeof(bad_cats[0]); ++i)
                {
                    int index = bad_cats[i];
                    if (predictions[index] > .01)
                    {
                        printf("Threat Detected!\n");
                        threat = 1;
                        break;
                    }
                }
                if (!threat) printf("Scanning...\n");
                for (i = 0; i < sizeof(bad_cats) / sizeof(bad_cats[0]); ++i)
                {
                    int index = bad_cats[i];
                    if (predictions[index] > .01)
                    {
                        printf("%s\n", names[index]);
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
            printf("Classifier Demo\n");
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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

                image ini = get_image_from_stream(cap);
                image in_s = resize_image(ini, net.w, net.h);
                show_image(ini, "Classifier");

                float[] predictions = network_predict(net, in_s.data);
                if (net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 1);
                top_predictions(net, top, indexes);

                printf("\033[2J");
                printf("\033[1;1H");
                printf("\nFPS:%.0f\n", fps);

                for (i = 0; i < top; ++i)
                {
                    int index = indexes[i];
                    printf("%.1f%%: %s\n", predictions[index] * 100, names[index]);
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
                printf("%s\n", gpu_list);
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
                    gpus[i] = atoi(gpu_list);
                    gpu_list = strchr(gpu_list, ',') + 1;
                }
            }
            else
            {
                gpu = gpu_index;
                gpus = &gpu;
                ngpus = 1;
            }

            int cam_index = Utils.find_int_arg(args.Count, args, "-c", 0);
            int top = Utils.find_int_arg(args.Count, args, "-t", 0);
            int clear = find_arg(args.Count, args, "-clear");
            string data = args[3];
            string cfg = args[4];
            string weights = (args.Count > 5) ? args[5] : 0;
            string filename = (args.Count > 6) ? args[6] : 0;
            string layer_s = (args.Count > 7) ? args[7] : 0;
            int layer = layer_s ? atoi(layer_s) : -1;
            if (0 == strcmp(args[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
            else if (0 == strcmp(args[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
            else if (0 == strcmp(args[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear);
            else if (0 == strcmp(args[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
            else if (0 == strcmp(args[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
            else if (0 == strcmp(args[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
            else if (0 == strcmp(args[2], "test")) test_classifier(data, cfg, weights, layer);
            else if (0 == strcmp(args[2], "label")) label_classifier(data, cfg, weights);
            else if (0 == strcmp(args[2], "valid")) validate_classifier_single(data, cfg, weights);
            else if (0 == strcmp(args[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
            else if (0 == strcmp(args[2], "valid10")) validate_classifier_10(data, cfg, weights);
            else if (0 == strcmp(args[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
            else if (0 == strcmp(args[2], "validfull")) validate_classifier_full(data, cfg, weights);
        }


        #endregion

        #region ArtFile


        void demo_art(string cfgfile, string weightfile, int cam_index)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);

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
                image ini = get_image_from_stream(cap);
                image in_s = resize_image(ini, net.w, net.h);
                show_image(ini, window);

                float[] p = network_predict(net, in_s.data);

                printf("\033[2J");
                printf("\033[1;1H");

                float score = 0;
                for (i = 0; i < n; ++i)
                {
                    float s = p[idx[i]];
                    if (s > score) score = s;
                }
                score = score;
                printf("I APPRECIATE THIS ARTWORK: %10.7f%%\n", score * 100);
                printf("[");
                int upper = 30;
                for (i = 0; i < upper; ++i)
                {
                    printf("%c", ((i + .5) < score * upper) ? 219 : ' ');
                }
                printf("]\n");

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
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            if (clear) *net.seen = 0;
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = 1024;
            list* plist = get_paths("/home/pjreddie/tag/train.list");
            string[] paths = (string[])list_to_array(plist);
            printf("%d\n", plist.size);
            int N = plist.size;
            clock_t time;
            pthread_t load_thread;
            data train;
            data buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;

            args.min = net.w;
            args.max = net.max_crop;
            args.size = net.w;

            args.paths = paths;
            args.classes = net.outputs;
            args.n = imgs;
            args.m = N;
            args.d = &buffer;
            args.type = TAG_DATA;

            args.angle = net.angle;
            args.exposure = net.exposure;
            args.saturation = net.saturation;
            args.hue = net.hue;

            Console.Error.Write($"%d classes\n", net.outputs);

            load_thread = load_data_in_thread(args);
            int epoch = (*net.seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;

                load_thread = load_data_in_thread(args);
                printf("Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
                free_data(train);
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
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
            sprintf(buff, "%s/%s.weights", backup_directory, basec);
            save_weights(net, buff);

            pthread_join(load_thread, 0);
            free_data(buffer);
            free_network(net);
            free_ptrs((void**)paths, plist.size);
            free_list(plist);
            free(basec);
        }

        void test_tag(string cfgfile, string weightfile, string filename)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            srand(2222222);
            int i = 0;
            string[] names = get_labels("data/tags.txt");
            clock_t time;
            int indexes[10];
            char buff[256];
            string input = buff;
            int size = net.w;
            while (1)
            {
                if (filename)
                {
                    strncpy(input, filename, 256);
                }
                else
                {
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image r = resize_min(im, size);
                resize_network(&net, r.w, r.h);
                printf("%d %d\n", r.w, r.h);

                float[] X = r.data;
                time = clock();
                float[] predictions = network_predict(net, X);
                top_predictions(net, 10, indexes);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                for (i = 0; i < 10; ++i)
                {
                    int index = indexes[i];
                    printf("%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }
                if (r.data != im.data) free_image(r);
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
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = 1024;
            list* plist = get_paths("data/compare.train.list");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.size;
            printf("%d\n", N);
            clock_t time;
            pthread_t load_thread;
            data train;
            data buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.classes = 20;
            args.n = imgs;
            args.m = N;
            args.d = &buffer;
            args.type = COMPARE_DATA;

            load_thread = load_data_in_thread(args);
            int epoch = *net.seen / N;
            int i = 0;
            while (1)
            {
                ++i;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;

                load_thread = load_data_in_thread(args);
                printf("Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                printf("%.3f: %f, %f avg, %lf seconds, %d images\n", (float)*net.seen / N, loss, avg_loss, sec(clock() - time), *net.seen);
                free_data(train);
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d_minor_%d.weights", backup_directory, basec, epoch, i);
                    save_weights(net, buff);
                }
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    i = 0;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                    if (epoch % 22 == 0) net.learning_rate *= .1;
                }
            }
            pthread_join(load_thread, 0);
            free_data(buffer);
            free_network(net);
            free_ptrs((void**)paths, plist.size);
            free_list(plist);
            free(basec);
        }

        void validate_compare(string filename, string weightfile)
        {
            int i = 0;
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            list* plist = get_paths("data/compare.val.list");
            //list *plist = get_paths("data/compare.val.old");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.size / 2;
            free_list(plist);

            clock_t time;
            int correct = 0;
            int total = 0;
            int splits = 10;
            int num = (i + 1) * N / splits - i * N / splits;

            data val, buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.classes = 20;
            args.n = num;
            args.m = 0;
            args.d = &buffer;
            args.type = COMPARE_DATA;

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
                printf("Loaded: %d images ini %lf seconds\n", val.X.rows, sec(clock() - time));

                time = clock();
                matrix pred = network_predict_data(net, val);
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
                printf("%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct / total, sec(clock() - time), val.X.rows);
                free_data(val);
            }
        }


        int total_compares = 0;
        int current_class = 0;

        int elo_comparator(void* a, void* b)
        {
            sortable_bbox box1 = *(sortable_bbox*)a;
            sortable_bbox box2 = *(sortable_bbox*)b;
            if (box1.elos[current_class] == box2.elos[current_class]) return 0;
            if (box1.elos[current_class] > box2.elos[current_class]) return -1;
            return 1;
        }

        int bbox_comparator(void* a, void* b)
        {
            ++total_compares;
            sortable_bbox box1 = *(sortable_bbox*)a;
            sortable_bbox box2 = *(sortable_bbox*)b;
            network net = box1.net;
            int sclass = box1.sclass;

            image im1 = load_image_color(box1.filename, net.w, net.h);
            image im2 = load_image_color(box2.filename, net.w, net.h);
            float[] X = (float[])calloc(net.w * net.h * net.c, sizeof(float));
            memcpy(X, im1.data, im1.w * im1.h * im1.c * sizeof(float));
            memcpy(X + im1.w * im1.h * im1.c, im2.data, im2.w * im2.h * im2.c * sizeof(float));
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

        void bbox_update(sortable_bbox* a, sortable_bbox* b, int sclass, int result)
        {
            int k = 32;
            float EA = 1./ (1 + pow(10, (b.elos[sclass] - a.elos[sclass]) / 400.));
            float EB = 1./ (1 + pow(10, (a.elos[sclass] - b.elos[sclass]) / 400.));
            float SA = result ? 1 : 0;
            float SB = result ? 0 : 1;
            a.elos[sclass] += k * (SA - EA);
            b.elos[sclass] += k * (SB - EB);
        }

        void bbox_fight(network net, sortable_bbox* a, sortable_bbox* b, int classes, int sclass)
        {
            image im1 = load_image_color(a.filename, net.w, net.h);
            image im2 = load_image_color(b.filename, net.w, net.h);
            float[] X = (float[])calloc(net.w * net.h * net.c, sizeof(float));
            memcpy(X, im1.data, im1.w * im1.h * im1.c * sizeof(float));
            memcpy(X + im1.w * im1.h * im1.c, im2.data, im2.w * im2.h * im2.c * sizeof(float));
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
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));
            set_batch_network(&net, 1);

            list* plist = get_paths("data/compare.sort.list");
            //list *plist = get_paths("data/compare.val.old");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.size;
            free_list(plist);
            sortable_bbox* boxes = (sortable_bbox*)calloc(N, sizeof(sortable_bbox));
            printf("Sorting %d boxes...\n", N);
            for (i = 0; i < N; ++i)
            {
                boxes[i].filename = paths[i];
                boxes[i].net = net;
                boxes[i].sclass = 7;
                boxes[i].elo = 1500;
            }
            clock_t time = clock();
            qsort(boxes, N, sizeof(sortable_bbox), bbox_comparator);
            for (i = 0; i < N; ++i)
            {
                printf("%s\n", boxes[i].filename);
            }
            printf("Sorted ini %d compares, %f secs\n", total_compares, sec(clock() - time));
        }

        void BattleRoyaleWithCheese(string filename, string weightfile)
        {
            int classes = 20;
            int i, j;
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));
            set_batch_network(&net, 1);

            list* plist = get_paths("data/compare.sort.list");
            //list *plist = get_paths("data/compare.small.list");
            //list *plist = get_paths("data/compare.cat.list");
            //list *plist = get_paths("data/compare.val.old");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.size;
            int total = N;
            free_list(plist);
            sortable_bbox* boxes = (sortable_bbox*)calloc(N, sizeof(sortable_bbox));
            printf("Battling %d boxes...\n", N);
            for (i = 0; i < N; ++i)
            {
                boxes[i].filename = paths[i];
                boxes[i].net = net;
                boxes[i].classes = classes;
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
                printf("Round: %d\n", round);
                shuffle(boxes, N, sizeof(sortable_bbox));
                for (i = 0; i < N / 2; ++i)
                {
                    bbox_fight(net, boxes + i * 2, boxes + i * 2 + 1, classes, -1);
                }
                printf("Round: %f secs, %d remaining\n", sec(clock() - round_time), N);
            }

            int sclass;

            for (sclass = 0; sclass < classes; ++sclass)
            {

                N = total;
                current_class = sclass;
                qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
                N /= 2;

                for (round = 1; round <= 100; ++round)
                {
                    clock_t round_time = clock();
                    printf("Round: %d\n", round);

                    sorta_shuffle(boxes, N, sizeof(sortable_bbox), 10);
                    for (i = 0; i < N / 2; ++i)
                    {
                        bbox_fight(net, boxes + i * 2, boxes + i * 2 + 1, classes, sclass);
                    }
                    qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
                    if (round <= 20) N = (N * 9 / 10) / 2 * 2;

                    printf("Round: %f secs, %d remaining\n", sec(clock() - round_time), N);
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
            printf("Tournament ini %d compares, %f secs\n", total_compares, sec(clock() - time));
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
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            string backup_directory = "/home/pjreddie/backup/";
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = 1024;
            int i = *net.seen / imgs;
            string[] labels = dice_labels;
            list* plist = get_paths("data/dice/dice.train.list");
            string[] paths = (string[])list_to_array(plist);
            printf("%d\n", plist.size);
            clock_t time;
            while (1)
            {
                ++i;
                time = clock();
                data train = load_data_old(paths, imgs, plist.size, labels, 6, net.w, net.h);
                printf("Loaded: %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock() - time), *net.seen);
                free_data(train);
                if ((i % 100) == 0) net.learning_rate *= .1;
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, i);
                    save_weights(net, buff);
                }
            }
        }

        void validate_dice(string filename, string weightfile)
        {
            network net = parse_network_cfg(filename);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            srand(time(0));

            string[] labels = dice_labels;
            list* plist = get_paths("data/dice/dice.val.list");

            string[] paths = (string[])list_to_array(plist);
            int m = plist.size;
            free_list(plist);

            data val = load_data_old(paths, m, 0, labels, 6, net.w, net.h);
            float[] acc = network_accuracies(net, val, 2);
            printf("Validation Accuracy: %f, %d images\n", acc[0], m);
            free_data(val);
        }

        void test_dice(string cfgfile, string weightfile, string filename)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, net.w, net.h);
                float[] X = im.data;
                float[] predictions = network_predict(net, X);
                top_predictions(net, 6, indexes);
                for (i = 0; i < 6; ++i)
                {
                    int index = indexes[i];
                    printf("%s: %f\n", names[index], predictions[index]);
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
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = net.batch * net.subdivisions;
            list* plist = get_paths("figures.list");
            string[] paths = (string[])list_to_array(plist);
            clock_t time;
            int N = plist.size;
            printf("N: %d\n", N);
            image outf = get_network_image(net);

            data train, buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.out_w = outf.w;
            args.out_h = outf.h;
            args.paths = paths;
            args.n = imgs;
            args.m = N;
            args.d = &buffer;
            args.type = WRITING_DATA;

            pthread_t load_thread = load_data_in_thread(args);
            int epoch = (*net.seen) / N;
            while (get_current_batch(net) < net.max_batches || net.max_batches == 0)
            {
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data_in_thread(args);
                printf("Loaded %lf seconds\n", sec(clock() - time));

                time = clock();
                float loss = train_network(net, train);


                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
                free_data(train);
                if (get_current_batch(net) % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "%s/%s_batch_%d.weights", backup_directory, basec, get_current_batch(net));
                    save_weights(net, buff);
                }
                if (*net.seen / N > epoch)
                {
                    epoch = *net.seen / N;
                    char buff[256];
                    sprintf(buff, "%s/%s_%d.weights", backup_directory, basec, epoch);
                    save_weights(net, buff);
                }
            }
        }

        void test_writing(string cfgfile, string weightfile, string filename)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
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
                    printf("Enter Image Path: ");
                    fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }

                image im = load_image_color(input, 0, 0);
                resize_network(&net, im.w, im.h);
                printf("%d %d %d\n", im.h, im.w, im.c);
                float[] X = im.data;
                time = clock();
                network_predict(net, X);
                printf("%s: Predicted ini %f seconds.\n", input, sec(clock() - time));
                image pred = get_network_image(net);

                image upsampled = resize_image(pred, im.w, im.h);
                image thresh = threshold_image(upsampled, .5);
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


        void fix_data_captcha(data d, int mask)
        {
            matrix labels = d.y;
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
            srand(time(0));
            float avg_loss = -1;
            string basec = basecfg(cfgfile);
            printf("%s\n", basec);
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
            int imgs = 1024;
            int i = *net.seen / imgs;
            int solved = 1;
            list* plist;
            string[] labels = get_labels("/data/captcha/reimgs.labels.list");
            if (solved)
            {
                plist = get_paths("/data/captcha/reimgs.solved.list");
            }
            else
            {
                plist = get_paths("/data/captcha/reimgs.raw.list");
            }
            string[] paths = (string[])list_to_array(plist);
            printf("%d\n", plist.size);
            clock_t time;
            pthread_t load_thread;
            data train;
            data buffer;

            load_args args = { 0 };
            args.w = net.w;
            args.h = net.h;
            args.paths = paths;
            args.classes = 26;
            args.n = imgs;
            args.m = plist.size;
            args.labels = labels;
            args.d = &buffer;
            args.type = CLASSIFICATION_DATA;

            load_thread = load_data_in_thread(args);
            while (1)
            {
                ++i;
                time = clock();
                pthread_join(load_thread, 0);
                train = buffer;
                fix_data_captcha(train, solved);

                load_thread = load_data_in_thread(args);
                printf("Loaded: %lf seconds\n", sec(clock() - time));
                time = clock();
                float loss = train_network(net, train);
                if (avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss * .9 + loss * .1;
                printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock() - time), *net.seen);
                free_data(train);
                if (i % 100 == 0)
                {
                    char buff[256];
                    sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights", basec, i);
                    save_weights(net, buff);
                }
            }
        }

        void test_captcha(string cfgfile, string weightfile, string filename)
        {
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            set_batch_network(&net, 1);
            srand(2222222);
            int i = 0;
            string[] names = get_labels("/data/captcha/reimgs.labels.list");
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
                    //printf("Enter Image Path: ");
                    //fflush(stdout);
                    input = fgets(input, 256, stdin);
                    if (!input) return;
                    strtok(input, "\n");
                }
                image im = load_image_color(input, net.w, net.h);
                float[] X = im.data;
                float[] predictions = network_predict(net, X);
                top_predictions(net, 26, indexes);
                //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
                for (i = 0; i < 26; ++i)
                {
                    int index = indexes[i];
                    if (i != 0) printf(", ");
                    printf("%s %f", names[index], predictions[index]);
                }
                printf("\n");
                fflush(stdout);
                free_image(im);
                if (filename) break;
            }
        }

        void valid_captcha(string cfgfile, string weightfile, string filename)
        {
            string[] labels = get_labels("/data/captcha/reimgs.labels.list");
            network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            list* plist = get_paths("/data/captcha/reimgs.fg.list");
            string[] paths = (string[])list_to_array(plist);
            int N = plist.size;
            int outputs = net.outputs;

            set_batch_network(&net, 1);
            srand(2222222);
            int i, j;
            for (i = 0; i < N; ++i)
            {
                if (i % 100 == 0) Console.Error.Write($"%d\n", i);
                image im = load_image_color(paths[i], net.w, net.h);
                float[] X = im.data;
                float[] predictions = network_predict(net, X);
                //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
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
                printf("%d, ", truth);
                for (j = 0; j < outputs; ++j)
                {
                    if (j != 0) printf(", ");
                    printf("%f", predictions[j]);
                }
                printf("\n");
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

        void optimize_picture(network* net, image orig, int max_layer, float scale, float rate, float thresh, int norm)
        {
            //scale_image(orig, 2);
            //translate_image(orig, -1);
            net.n = max_layer + 1;

            int dx = rand() % 16 - 8;
            int dy = rand() % 16 - 8;
            int flip = rand() % 2;

            image crop = crop_image(orig, dx, dy, orig.w, orig.h);
            image im = resize_image(crop, (int)(orig.w * scale), (int)(orig.h * scale));
            if (flip) flip_image(im);

            resize_network(net, im.w, im.h);
            layer last = net.layers[net.n - 1];
            //net.layers[net.n - 1].activation = LINEAR;

            image delta = make_image(im.w, im.h, im.c);

            network_state state = { 0 };
            i
            state.input = cuda_make_array(im.data, im.w * im.h * im.c);
            state.delta = cuda_make_array(im.data, im.w * im.h * im.c);

            forward_network_gpu(*net, state);
            copy_ongpu(last.outputs, last.output_gpu, 1, last.delta_gpu, 1);

            cuda_pull_array(last.delta_gpu, last.delta, last.outputs);
            calculate_loss(last.delta, last.delta, last.outputs, thresh);
            cuda_push_array(last.delta_gpu, last.delta, last.outputs);

            backward_network_gpu(*net, state);

            cuda_pull_array(state.delta, delta.data, im.w * im.h * im.c); i

            if (flip) flip_image(delta);
            //normalize_array(delta.data, delta.w*delta.h*delta.c);
            image resized = resize_image(delta, orig.w, orig.h);
            image outi = crop_image(resized, -dx, -dy, orig.w, orig.h);

            if (norm) normalize_array(outi.data, outi.w * outi.h * outi.c);
            axpy_cpu(orig.w * orig.h * orig.c, rate, outi.data, 1, orig.data, 1);


            constrain_image(orig);

            free_image(crop);
            free_image(im);
            free_image(delta);
            free_image(resized);
            free_image(outi);

        }

        void smooth(image recon, image update, float lambda, int num)
        {
            int i, j, k;
            int ii, jj;
            for (k = 0; k < recon.c; ++k)
            {
                for (j = 0; j < recon.h; ++j)
                {
                    for (i = 0; i < recon.w; ++i)
                    {
                        int out_index = i + recon.w * (j + recon.h * k);
                        for (jj = j - num; jj <= j + num && jj < recon.h; ++jj)
                        {
                            if (jj < 0) continue;
                            for (ii = i - num; ii <= i + num && ii < recon.w; ++ii)
                            {
                                if (ii < 0) continue;
                                int in_index = ii + recon.w * (jj + recon.h * k);
                                update.data[out_index] += lambda * (recon.data[in_index] - recon.data[out_index]);
                            }
                        }
                    }
                }
            }
        }

        void reconstruct_picture(network net, float[] features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters)
        {
            int iter = 0;
            for (iter = 0; iter < iters; ++iter)
            {
                image delta = make_image(recon.w, recon.h, recon.c);

                network_state state = { 0 };
                state.input = cuda_make_array(recon.data, recon.w * recon.h * recon.c);
                state.delta = cuda_make_array(delta.data, delta.w * delta.h * delta.c);
                state.truth = cuda_make_array(features, get_network_output_size(net));

                forward_network_gpu(net, state);
                backward_network_gpu(net, state);

                cuda_pull_array(state.delta, delta.data, delta.w * delta.h * delta.c);

                axpy_cpu(recon.w * recon.h * recon.c, 1, delta.data, 1, update.data, 1);
                smooth(recon, update, lambda, smooth_size);

                axpy_cpu(recon.w * recon.h * recon.c, rate, update.data, 1, recon.data, 1);
                scal_cpu(recon.w * recon.h * recon.c, momentum, update.data, 1);

                //float mag = mag_array(recon.data, recon.w*recon.h*recon.c);
                //scal_cpu(recon.w*recon.h*recon.c, 600/mag, recon.data, 1);

                constrain_image(recon);
                free_image(delta);
            }
        }


        public static void run_nightmare(List<string> args)
        {
            srand(0);
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [cfg] [weights] [image] [layer] [options! (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[2];
            string weights = args[3];
            string input = args[4];
            int max_layer = atoi(args[5]);

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

            network net = parse_network_cfg(cfg);
            load_weights(&net, weights);
            string cfgbase = basecfg(cfg);
            string imbase = basecfg(input);

            set_batch_network(&net, 1);
            image im = load_image_color(input, 0, 0);
            if (0)
            {
                float scale = 1;
                if (im.w > 512 || im.h > 512)
                {
                    if (im.w > im.h) scale = 512.0 / im.w;
                    else scale = 512.0 / im.h;
                }
                image resized = resize_image(im, scale * im.w, scale * im.h);
                free_image(im);
                im = resized;
            }

            float[] features = 0;
            image update;
            if (reconstruct)
            {
                resize_network(&net, im.w, im.h);

                int zz = 0;
                network_predict(net, im.data);
                image out_im = get_network_image(net);
                image crop = crop_image(out_im, zz, zz, out_im.w - 2 * zz, out_im.h - 2 * zz);
                //flip_image(crop);
                image f_im = resize_image(crop, out_im.w, out_im.h);
                free_image(crop);
                printf("%d features\n", out_im.w * out_im.h * out_im.c);


                im = resize_image(im, im.w, im.h);
                f_im = resize_image(f_im, f_im.w, f_im.h);
                features = f_im.data;

                int i;
                for (i = 0; i < 14 * 14 * 512; ++i)
                {
                    features[i] += rand_uniform(-.19, .19);
                }

                free_image(im);
                im = make_random_image(im.w, im.h, im.c);
                update = make_image(im.w, im.h, im.c);

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
                        int layer = max_layer + rand() % range - range / 2;
                        int octave = rand() % octaves;
                        optimize_picture(&net, im, layer, 1 / pow(1.33333333, octave), rate, thresh, norm);
                    }
                }
                Console.Error.Write($"done\n");
                if (0)
                {
                    image g = grayscale_image(im);
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
                printf("%d %s\n", e, buff);
                save_image(im, buff);
                //show_image(im, buff);
                //cvWaitKey(0);

                if (rotate)
                {
                    image rot = rotate_image(im, rotate);
                    free_image(im);
                    im = rot;
                }
                image crop = crop_image(im, im.w * (1. - zoom) / 2., im.h * (1.- zoom) / 2., im.w * zoom, im.h * zoom);
                image resized = resize_image(crop, im.w, im.h);
                free_image(im);
                free_image(crop);
                im = resized;
            }
        }


        #endregion
    }
}
