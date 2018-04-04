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
    public static class Yolo
    {
        private static readonly string[] VocNames = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

        private static void train_yolo(string cfgfile, string weightfile)
        {
            string trainImages = "/Data.Data/voc/train.txt";
            string backupDirectory = "/home/pjreddie/backup/";

            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"%s\n", basec);
            float avgLoss = -1;
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

            string[] paths = Data.Data.GetPaths(trainImages);

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

            Thread loadThread = Data.Data.load_data_in_thread(args);
            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Start();
                loadThread.Join();
                var train = buffer;
                loadThread = Data.Data.load_data_in_thread(args);

                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avgLoss < 0) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;

                sw.Stop();
                Console.Write($"%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds, i * imgs);
                if (i % 1000 == 0 || (i < 1000 && i % 100 == 0))
                {
                    string buff = $"{backupDirectory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backupDirectory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }

        private static void print_yolo_detections(FileStream fps, string id, Box[] boxes, float[][] probs, int probsJ, int total, int classes, int w, int h)
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

        private static void validate_yolo(string cfgfile, string weightfile)
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
            float iouThresh = .5f;

            int nthreads = 8;
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

            var sw = new Stopwatch();
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
                    l.get_detection_boxes(w, h, thresh, probs, boxes, false);
                    if (nms)
                    {
                        Box.do_nms_sort(boxes, probs, l.Side * l.Side * l.N, classes, iouThresh);
                    }

                    for (j = 0; j < classes; ++j)
                    {
                        string buff = $"{basec}{VocNames[j]}.txt";
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

        private static void validate_yolo_recall(string cfgfile, string weightfile)
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
            int side = l.Side;

            int j, k;
            Box[] boxes = new Box[side * side * l.N];
            float[][] probs = new float[side * side * l.N][];
            for (j = 0; j < side * side * l.N; ++j) probs[j] = new float[classes];

            int m = paths.Length;
            int i = 0;

            float thresh = .001f;
            float iouThresh = .5f;

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
                l.get_detection_boxes(orig.W, orig.H, thresh, probs, boxes, true);

                string labelpath;
                Utils.find_replace(path, "images", "labels", out labelpath);
                Utils.find_replace(labelpath, "JPEGImages", "labels", out labelpath);
                Utils.find_replace(labelpath, ".jpg", ".txt", out labelpath);
                Utils.find_replace(labelpath, ".JPEG", ".txt", out labelpath);

                int numLabels = 0;
                BoxLabel[] truth = Data.Data.read_boxes(labelpath, ref numLabels);
                for (k = 0; k < side * side * l.N; ++k)
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
                    for (k = 0; k < side * side * l.N; ++k)
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

        private static void test_yolo(string cfgfile, string weightfile, string filename, float thresh)
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
                float[] x = sized.Data;
                sw.Start();
                Network.network_predict(net, x);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                l.get_detection_boxes(1, 1, thresh, probs, boxes, false);
                Box.do_nms_sort(boxes, probs, l.Side * l.Side * l.N, l.Classes, nms);

                LoadArgs.draw_detections(im, l.Side * l.Side * l.N, thresh, boxes, probs, VocNames, alphabet, 20);
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
            int camIndex = Utils.find_int_arg(args, "-c", 0);
            int frameSkip = Utils.find_int_arg(args, "-s", 0);
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
            else if (args[2] == "demo") Demo.DemoRun(cfg, weights, thresh, camIndex, filename, VocNames, 20, frameSkip, prefix);
        }
    }
}