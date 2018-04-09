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
    public static class Coco
    {
        private static readonly string[] CocoClasses = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
        public static readonly int[] CocoIds = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };
       
        private static void train_coco(string cfgfile, string weightfile)
        {
            //char *train_images = "/home/pjreddie/Data.Data/voc/test/train.txt";
            //char *train_images = "/home/pjreddie/Data.Data/coco/train.txt";
            string trainImages = "Data.Data/coco.trainval.txt";
            //char *train_images = "Data.Data/bags.train.list";
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
                sw.Reset();
                sw.Start();
                loadThread.Join();
                var train = buffer;
                loadThread = Data.Data.load_data_in_thread(args);

                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Stop();
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
                if (i % 100 == 0)
                {

                    string buff = $"{backupDirectory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backupDirectory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }

        private static void print_cocos(FileStream fp, int imageId, Box[] boxes, float[][] probs, int numBoxes, int classes, int w, int h)
        {
            int i, j;
            for (i = 0; i < numBoxes; ++i)
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
                            $"{{\"image_id\":{imageId}, \"category_id\":{CocoIds[j]}, \"bbox\":[{bx}, {@by}, {bw}, {bh}], \"score\":{probs[i][j]}}},\n");
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

        private static void validate_coco(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            Network.set_batch_network(ref net, 1);
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
                float iouThresh = .5f;

                int nthreads = 8;
                var val = new Image[nthreads];
                var valResized = new Image[nthreads];
                var buf = new Image[nthreads];
                var bufResized = new Image[nthreads];
                var thr = new Thread[nthreads];

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
                        int imageId = get_coco_image_id(path);
                        float[] x = valResized[t].Data;
                        Network.network_predict(ref net, ref x);
                        int w = val[t].Width;
                        int h = val[t].Height;
                        l.get_detection_boxes( w, h, thresh, ref probs, ref boxes, false);
                        if (nms) Box.do_nms_sort(boxes, probs, side * side * l.N, classes, iouThresh);
                        print_cocos(fp, imageId, boxes, probs, side * side * l.N, classes, w, h);
                    }
                }

                fp.Seek(-1, SeekOrigin.Current);
                temp = Encoding.UTF8.GetBytes("[\n]\n");
                fp.Write(temp, 0, temp.Length);
                sw.Stop();
                Console.Error.Write($"Total Detection Time: %f Seconds\n", sw.Elapsed.Seconds);
            }
        }

        private static void validate_coco_recall(string cfgfile, string weightfile)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(ref net, 1);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);


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
                Network.network_predict(ref net, ref sized.Data);
                l.get_detection_boxes( 1, 1, thresh, ref probs, ref boxes, true);

                Utils.find_replace(path, "images", "labels", out var labelpath);
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
                    Box t = new Box(truth[j].X, truth[j].Y, truth[j].W, truth[j].H );
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

                Console.Error.Write(
                    $"{i} {correct} {total}\tRPs/Img: {proposals/(i+1):F2}\tIOU: {avgIou:P}\tRecall:{correct / total:P}\n");
                
            }
        }

        private static void test_coco(string cfgfile, string weightfile, string filename, float thresh)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Layer l = net.Layers[net.N - 1];
            Network.set_batch_network(ref net, 1);
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
                float[] x = sized.Data;
                sw.Reset();
                sw.Start();
                Network.network_predict(ref net, ref x);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                l.get_detection_boxes( 1, 1, thresh, ref probs, ref boxes, false);
                if (nms != 0) Box.do_nms_sort(boxes, probs, l.Side * l.Side * l.N, l.Classes, nms);
                LoadArgs.draw_detections(ref im, l.Side * l.Side * l.N, thresh, boxes, probs, CocoClasses, 80);
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
            int camIndex = Utils.find_int_arg( args, "-c", 0);
            int frameSkip = Utils.find_int_arg( args, "-s", 0);

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
            else if (args[2] == "demo") Demo.DemoRun(cfg, weights, thresh, camIndex, filename, CocoClasses, 80, frameSkip, prefix);
        }
    }
}