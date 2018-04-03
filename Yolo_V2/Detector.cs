using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using Emgu.CV;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    class Detector
    {
        public static void train_detector(string datacfg, string cfgfile, string weightfile, bool clear)
        {
            var options = OptionList.read_data_cfg(datacfg);
            string trainImages = OptionList.option_find_str(options, "train", "Data.Data/train.list");
            string backupDirectory = OptionList.option_find_str(options, "backup", "/backup/");


            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avgLoss = -1;
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

            string[] paths = Data.Data.GetPaths(trainImages);

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

            Thread loadThread = Data.Data.load_data(args);
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

                    loadThread.Join();
                    loadThread = Data.Data.load_data(args);

                    for (i = 0; i < 1; ++i)
                    {
                        Network.resize_network(nets[i], dim, dim);
                    }
                    net = nets[0];
                }
                sw.Start();
                loadThread.Join();
                train = buffer;
                loadThread = Data.Data.load_data(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss;
                loss = Network.train_network(net, train);
                if (avgLoss < 0) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;

                i = Network.get_current_batch(net);
                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {

                    string buffa = $"{backupDirectory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buffa);
                }
            }

            string buff = $"{backupDirectory}/{basec}_final.Weights";
            Parser.save_weights(net, buff);
        }

        private static void print_cocos(FileStream fp, string imagePath, Box[] boxes, float[][] probs, int numBoxes, int classes, int w, int h)
        {
            int i, j;
            int imageId = Coco.get_coco_image_id(imagePath);
            for (i = 0; i < numBoxes; ++i)
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
                        var temp = Encoding.UTF8.GetBytes($"{{\"image_id\":{imageId}, \"category_id\":{Coco.CocoIds[j]}, \"bbox\":[{bx}, {@by}, {bw}, {bh}], \"score\":{probs[i][j]}}},\n");
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
            string validImages = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            string nameList = OptionList.option_find_str(options, "names", "Data.Data/names.list");
            string prefix = OptionList.option_find_str(options, "results", "results");
            string[] names = Data.Data.get_labels(nameList);
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
            string[] paths = Data.Data.GetPaths(validImages);

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
                Image[] valResized = new Image[nthreads];
                Image[] buf = new Image[nthreads];
                Image[] bufResized = new Image[nthreads];
                Thread[] thr = new Thread[nthreads];

                LoadArgs args = new LoadArgs();
                args.W = net.W;
                args.H = net.H;
                args.Type = DataType.ImageData;

                for (t = 0; t < nthreads; ++t)
                {
                    args.Path = paths[i + t];
                    args.Im = buf[t];
                    args.Resized = bufResized[t];
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
                        valResized[t] = bufResized[t];
                    }
                    for (t = 0; t < nthreads && i + t < m; ++t)
                    {
                        args.Path = paths[i + t];
                        args.Im = buf[t];
                        args.Resized = bufResized[t];
                        thr[t] = Data.Data.load_data_in_thread(args);
                    }
                    for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
                    {
                        string path = paths[i + t - nthreads];
                        string id = Utils.Basecfg(path);
                        float[] x = valResized[t].Data;
                        Network.network_predict(net, x);
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
            float iouThresh = .5f;
            float nms = .4f;

            int total = 0;
            int correct = 0;
            int proposals = 0;
            float avgIou = 0;

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

                int numLabels = 0;
                BoxLabel[] truth = Data.Data.read_boxes(labelpath, ref numLabels);
                for (k = 0; k < l.W * l.H * l.N; ++k)
                {
                    if (probs[k][0] > thresh)
                    {
                        ++proposals;
                    }
                }
                for (j = 0; j < numLabels; ++j)
                {
                    ++total;
                    Box t = new Box(truth[j].X, truth[j].Y, truth[j].W, truth[j].H);
                    float bestIou = 0;
                    for (k = 0; k < l.W * l.H * l.N; ++k)
                    {
                        float iou = Box.box_iou(boxes[k], t);
                        if (probs[k][0] > thresh && iou > bestIou)
                        {
                            bestIou = iou;
                        }
                    }
                    avgIou += bestIou;
                    if (bestIou > iouThresh)
                    {
                        ++correct;
                    }
                }

                Console.Error.Write($"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avgIou * 100 / total, 100f * correct / total);
            }
        }

        public static void test_detector(string datacfg, string cfgfile, string weightfile, string filename, float thresh)
        {
            var options = OptionList.read_data_cfg(datacfg);
            string nameList = OptionList.option_find_str(options, "names", "Data.Data/names.list");
            string[] names = Data.Data.get_labels(nameList);

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

                float[] x = sized.Data;
                sw.Start();
                Network.network_predict(net, x);
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
            int camIndex = Utils.find_int_arg(args, "-c", 0);
            int frameSkip = Utils.find_int_arg(args, "-s", 0);
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
                string nameList = OptionList.option_find_str(options, "names", "Data.Data/names.list");
                string[] names = Data.Data.get_labels(nameList);
                Demo.DemoRun(cfg, weights, thresh, camIndex, filename, names, classes, frameSkip, prefix);
            }
        }
    }
}