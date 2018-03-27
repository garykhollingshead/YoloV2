using System;
using System.Collections.Generic;
using System.Linq;
using Yolo_V2.Data;

namespace Yolo_V2
{
    class YoloV2
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
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
                    float thresh = find_float_arg(args, "-thresh", .24);
                    string filename = (argc > 4) ? args[4] : 0;
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
                    composite_3d(args[2], args[3], args[4], (argc > 5) ? atof(args[5]) : 0);
                    break;
                case "test":
                    test_resize(args[2]);
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
                    speed(args[2], (argc > 3 && args[3]) ? atoi(args[3]) : 0);
                    break;
                case "oneoff":
                    oneoff(args[2], args[3], args[4]);
                    break;
                case "partial":
                    partial(args[2], args[3], args[4], atoi(args[5]));
                    break;
                case "visualize":
                    visualize(args[2], (argc > 3) ? args[3] : 0);
                    break;
                case "imtest":
                    test_resize(args[2]);
                    break;
                default:
                    Console.Error.WriteLine($"Not an option: {args[1]}");
                    break;
            }
        }

        private void average(List<string> args)
        {
            string cfgfile = args[2];
            string outfile = args[3];

            Network net = Network parse_network_cfg(cfgfile);
            Network sum = parse_network_cfg(cfgfile);

            string weightfile = args[4];
            load_weights(&sum, weightfile);

            int i, j;
            int n = argc - 5;
            for (i = 0; i < n; ++i)
            {
                weightfile = args[i + 5];
                load_weights(&net, weightfile);
                for (j = 0; j < net.n; ++j)
                {
                    Layer l = net.layers[j];
                    Layer out = sum.layers[j];
                    if (l.type == CONVOLUTIONAL)
                    {
                        int num = l.n * l.c * l.size * l.size;
                        axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                        axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                        if (l.batch_normalize)
                        {
                            axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                            axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                            axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                        }
                    }
                    if (l.type == CONNECTED)
                    {
                        axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                        axpy_cpu(l.outputs * l.inputs, 1, l.weights, 1, out.weights, 1);
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
                    scal_cpu(l.n, 1./ n, l.biases, 1);
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
                    scal_cpu(l.outputs, 1./ n, l.biases, 1);
                    scal_cpu(l.outputs * l.inputs, 1./ n, l.weights, 1);
                }
            }
            save_weights(sum, outfile);
        }

        private void speed(string cfgfile, int tics)
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

        private void operations(string cfgfile)
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

        private void oneoff(string cfgfile, string weightfile, string outfile)
        {
            gpu_index = -1;
            Network net = parse_network_cfg(cfgfile);
            int oldn = net.layers[net.n - 2].n;
            int c = net.layers[net.n - 2].c;
            net.layers[net.n - 2].n = 9372;
            net.layers[net.n - 2].biases += 5;
            net.layers[net.n - 2].weights += 5 * c;
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            net.layers[net.n - 2].biases -= 5;
            net.layers[net.n - 2].weights -= 5 * c;
            net.layers[net.n - 2].n = oldn;
            printf("%d\n", oldn);
            Layer l = net.layers[net.n - 2];
            copy_cpu(l.n / 3, l.biases, 1, l.biases + l.n / 3, 1);
            copy_cpu(l.n / 3, l.biases, 1, l.biases + 2 * l.n / 3, 1);
            copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + l.n / 3 * l.c, 1);
            copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + 2 * l.n / 3 * l.c, 1);
            *net.seen = 0;
            save_weights(net, outfile);
        }

        private void partial(string cfgfile, string weightfile, string outfile, int max)
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

        private void rescale_net(string cfgfile, string weightfile, string outfile)
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

        private void rgbgr_net(string cfgfile, string weightfile, string outfile)
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

        private void reset_normalize_net(string cfgfile, string weightfile, string outfile)
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

        private void normalize_net(string cfgfile, string weightfile, string outfile)
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
                    *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
                    *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
                    *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
                    *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
                    *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
                    *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
                    net.layers[i].batch_normalize = 1;
                }
            }
            save_weights(net, outfile);
        }

        private void statistics_net(string cfgfile, string weightfile)
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

        private void denormalize_net(string cfgfile, string weightfile, string outfile)
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
                    l.input_z_layer->batch_normalize = 0;
                    l.input_r_layer->batch_normalize = 0;
                    l.input_h_layer->batch_normalize = 0;
                    l.state_z_layer->batch_normalize = 0;
                    l.state_r_layer->batch_normalize = 0;
                    l.state_h_layer->batch_normalize = 0;
                    net.layers[i].batch_normalize = 0;
                }
            }
            save_weights(net, outfile);
        }

        private void visualize(string cfgfile, string weightfile)
        {
            Network net = parse_network_cfg(cfgfile);
            if (weightfile)
            {
                load_weights(&net, weightfile);
            }
            visualize_network(net);
            cvWaitKey(0);
        }

    }
}
