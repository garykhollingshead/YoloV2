using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Emgu.CV;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Classifier
    {
        private static void train_classifier(string datacfg, string cfgfile, string weightfile, int[] gpus, int ngpus, bool clear)
        {
            int i;

            float avgLoss = -1;
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

            string backupDirectory = OptionList.option_find_str(options, "backup", "/backup/");
            string labelList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string trainList = OptionList.option_find_str(options, "train", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);

            string[] labels = Data.Data.get_labels(labelList);

            string[] paths = Data.Data.GetPaths(trainList);
            Console.Write($"%d\n", paths.Length);
            int n = paths.Length;
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
            args.M = n;
            args.Labels = labels;
            args.Type = DataType.ClassificationData;

            Data.Data train;
            Data.Data buffer = new Data.Data();
            Thread loadThread;
            args.D = buffer;
            loadThread = Data.Data.load_data(args);

            int epoch = (net.Seen) / n;
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();

                loadThread.Join();
                train = buffer;
                loadThread = Data.Data.load_data(args);

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
                if (avgLoss == -1) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), (float)(net.Seen) / n, loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds, net.Seen);
                if (net.Seen / n > epoch)
                {
                    epoch = net.Seen / n;

                    string buff = $"{backupDirectory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backupDirectory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backupDirectory}/{basec}.Weights";
            Parser.save_weights(net, buff2);
        }

        private static void validate_classifier_crop(string datacfg, string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string labelList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string validList = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(labelList);

            string[] paths = Data.Data.GetPaths(validList);
            int m = paths.Length;

            var sw = new Stopwatch();
            float avgAcc = 0;
            float avgTopk = 0;
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

            Thread loadThread = Data.Data.load_data_in_thread(args);
            for (i = 1; i <= splits; ++i)
            {
                sw.Reset();
                sw.Start();

                loadThread.Join();
                var val = buffer;

                num = (i + 1) * m / splits - i * m / splits;
                string[] part = new string[paths.Length - (i * m / splits)];
                Array.Copy(paths, i * m / splits, part, 0, part.Length);
                if (i != splits)
                {
                    args.Paths = part;
                    loadThread = Data.Data.load_data_in_thread(args);
                }
                sw.Stop();
                Console.Write($"Loaded: %d images ini %lf seconds\n", val.X.Rows, sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float[] acc = Network.network_accuracies(net, val, topk);
                avgAcc += acc[0];
                avgTopk += acc[1];
                sw.Stop();
                Console.Write($"%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avgAcc / i, topk, avgTopk / i, sw.Elapsed.Seconds, val.X.Rows);
            }
        }

        private static void validate_classifier_10(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string labelList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string validList = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(labelList);

            string[] paths = Data.Data.GetPaths(validList);
            int m = paths.Length;

            float avgAcc = 0;
            float avgTopk = 0;
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
                if (indexes[0] == class2) avgAcc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avgTopk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avgAcc / (i + 1), topk, avgTopk / (i + 1));
            }
        }

        private static void validate_classifier_full(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string labelList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string validList = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(labelList);

            string[] paths = Data.Data.GetPaths(validList);
            int m = paths.Length;

            float avgAcc = 0;
            float avgTopk = 0;
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

                if (indexes[0] == class2) avgAcc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avgTopk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avgAcc / (i + 1), topk, avgTopk / (i + 1));
            }
        }

        private static void validate_classifier_single(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);


            var options = OptionList.read_data_cfg(datacfg);

            string labelList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string leafList = OptionList.option_find_str(options, "leaves", "");
            if (!string.IsNullOrEmpty(leafList)) net.Hierarchy.Change_leaves( leafList);
            string validList = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(labelList);

            string[] paths = Data.Data.GetPaths(validList);
            int m = paths.Length;

            float avgAcc = 0;
            float avgTopk = 0;
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

                if (indexes[0] == class2) avgAcc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avgTopk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avgAcc / (i + 1), topk, avgTopk / (i + 1));
            }
        }

        private static void validate_classifier_multi(string datacfg, string filename, string weightfile)
        {
            int i, j;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string labelList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            string validList = OptionList.option_find_str(options, "valid", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);
            int topk = OptionList.option_find_int(options, "top", 1);

            string[] labels = Data.Data.get_labels(labelList);
            int[] scales = { 224, 288, 320, 352, 384 };
            int nscales = scales.Length;

            string[] paths = Data.Data.GetPaths(validList);
            int m = paths.Length;

            float avgAcc = 0;
            float avgTopk = 0;
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
                if (indexes[0] == class2) avgAcc += 1;
                for (j = 0; j < topk; ++j)
                {
                    if (indexes[j] == class2) avgTopk += 1;
                }

                Console.Write($"%d: top 1: %f, top %d: %f\n", i, avgAcc / (i + 1), topk, avgTopk / (i + 1));
            }
        }

        private static void try_classifier(string datacfg, string cfgfile, string weightfile, string filename, int layerNum)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
            Utils.Rand = new Random(2222222);

            var options = OptionList.read_data_cfg(datacfg);

            string nameList = OptionList.option_find_str(options, "names", "");
            if (string.IsNullOrEmpty(nameList)) nameList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            int top = OptionList.option_find_int(options, "top", 1);

            int i = 0;
            string[] names = Data.Data.get_labels(nameList);
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

                float[] x = im.Data;
                sw.Reset();
                sw.Start();
                float[] predictions = Network.network_predict(net, x);

                Layer l = net.Layers[layerNum];
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

            string nameList = OptionList.option_find_str(options, "names", "");
            if (string.IsNullOrEmpty(nameList)) nameList = OptionList.option_find_str(options, "labels", "Data.Data/labels.list");
            if (top == 0) top = OptionList.option_find_int(options, "top", 1);

            int i = 0;
            string[] names = Data.Data.get_labels(nameList);
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

                float[] x = r.Data;
                sw.Reset();
                sw.Start();
                float[] predictions = Network.network_predict(net, x);
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

        private static void label_classifier(string datacfg, string filename, string weightfile)
        {
            int i;
            Network net = Parser.parse_network_cfg(filename);
            Network.set_batch_network(net, 1);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string labelList = OptionList.option_find_str(options, "names", "Data.Data/labels.list");
            string testList = OptionList.option_find_str(options, "test", "Data.Data/train.list");
            int classes = OptionList.option_find_int(options, "classes", 2);

            string[] labels = Data.Data.get_labels(labelList);

            string[] paths = Data.Data.GetPaths(testList);
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

        private static void test_classifier(string datacfg, string cfgfile, string weightfile, int targetLayer)
        {
            int curr = 0;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var options = OptionList.read_data_cfg(datacfg);

            string testList = OptionList.option_find_str(options, "test", "Data.Data/test.list");
            int classes = OptionList.option_find_int(options, "classes", 2);

            string[] paths = Data.Data.GetPaths(testList);
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

            Thread loadThread = Data.Data.load_data_in_thread(args);
            for (curr = net.Batch; curr < m; curr += net.Batch)
            {
                sw.Reset();
                sw.Start();

                loadThread.Join();
                var val = buffer;

                if (curr < m)
                {
                    args.Paths = new string[paths.Length - curr];
                    Array.Copy(paths, curr, args.Paths, 0, args.Paths.Length);
                    if (curr + net.Batch > m) args.N = m - curr;
                    loadThread = Data.Data.load_data_in_thread(args);
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

        private static void threat_classifier(string datacfg, string cfgfile, string weightfile, int camIndex, string filename)
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
                : new VideoCapture(camIndex))
            {

                int top = OptionList.option_find_int(options, "top", 1);

                string nameList = OptionList.option_find_str(options, "names", "");
                string[] names = Data.Data.get_labels(nameList);

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
                    Image inS = LoadArgs.resize_image(ini, net.W, net.H);

                    Image outo = ini;
                    int x1 = outo.W / 20;
                    int y1 = outo.H / 20;
                    int x2 = 2 * x1;
                    int y2 = outo.H - outo.H / 20;

                    int border = (int) (.01f * outo.H);
                    int h = y2 - y1 - 2 * border;
                    int w = x2 - x1 - 2 * border;

                    float[] predictions = Network.network_predict(net, inS.Data);
                    float currThreat = 0;
                    currThreat = predictions[0] * 0f +
                                 predictions[1] * .6f +
                                 predictions[2];
                    threat = roll * currThreat + (1 - roll) * threat;

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
                            (int)(y1 + .42 * h + 3 * border), 3 * border, 1, 1, 0);
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

                    string buff = $"/home/pjreddie/tmp/threat_{count:D6}";

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

        private static void gun_classifier(string datacfg, string cfgfile, string weightfile, int camIndex, string filename)
        {
            int[] badCats = { 218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697 };

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
                : new VideoCapture(camIndex))
            {

                int top = OptionList.option_find_int(options, "top", 1);

                string nameList = OptionList.option_find_str(options, "names", "");
                string[] names = Data.Data.get_labels(nameList);

                int[] indexes = new int[top];

                if (cap.IsOpened) Utils.Error("Couldn't connect to webcam.\n");

                float fps = 0;
                int i;

                while (true)
                {
                    var sw = new Stopwatch();
                    sw.Start();

                    Image ini = LoadArgs.get_image_from_stream(cap);
                    Image inS = LoadArgs.resize_image(ini, net.W, net.H);
                    LoadArgs.show_image(ini, "Threat Detection");

                    float[] predictions = Network.network_predict(net, inS.Data);
                    Network.top_predictions(net, top, indexes);

                    Console.Write($"\033[2J");
                    Console.Write($"\033[1;1H");

                    bool threat = false;
                    for (i = 0; i < badCats.Length; ++i)
                    {
                        int index = badCats[i];
                        if (predictions[index] > .01)
                        {
                            Console.Write($"Threat Detected!\n");
                            threat = true;
                            break;
                        }
                    }

                    if (threat) Console.Write($"Scanning...\n");
                    for (i = 0; i < badCats.Length; ++i)
                    {
                        int index = badCats[i];
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

        private static void demo_classifier(string datacfg, string cfgfile, string weightfile, int camIndex, string filename)
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
                : new VideoCapture(camIndex))
            {
                int top = OptionList.option_find_int(options, "top", 1);

                string nameList = OptionList.option_find_str(options, "names", "");
                string[] names = Data.Data.get_labels(nameList);

                int[] indexes = new int[top];

                if (cap != null) Utils.Error("Couldn't connect to webcam.\n");

                float fps = 0;
                int i;

                while (true)
                {
                    var sw = new Stopwatch();
                    sw.Start();

                    Image ini = LoadArgs.get_image_from_stream(cap);
                    Image inS = LoadArgs.resize_image(ini, net.W, net.H);
                    LoadArgs.show_image(ini, "Classifier");

                    float[] predictions = Network.network_predict(net, inS.Data);
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

            int camIndex = Utils.find_int_arg(args, "-c", 0);
            int top = Utils.find_int_arg(args, "-t", 0);
            bool clear = Utils.find_arg(args, "-clear");
            string data = args[3];
            string cfg = args[4];
            string weights = (args.Count > 5) ? args[5] : "";
            string filename = (args.Count > 6) ? args[6] : "";
            string layerS = (args.Count > 7) ? args[7] : "";
            int layer = !string.IsNullOrEmpty(layerS) ? int.Parse(layerS) : -1;
            if (args[2] == "predict") predict_classifier(data, cfg, weights, filename, top);
            else if (args[2] == "try") try_classifier(data, cfg, weights, filename, int.Parse(layerS));
            else if (args[2] == "train") train_classifier(data, cfg, weights, new int[0], 1, clear);
            else if (args[2] == "demo") demo_classifier(data, cfg, weights, camIndex, filename);
            else if (args[2] == "gun") gun_classifier(data, cfg, weights, camIndex, filename);
            else if (args[2] == "threat") threat_classifier(data, cfg, weights, camIndex, filename);
            else if (args[2] == "test") test_classifier(data, cfg, weights, layer);
            else if (args[2] == "label") label_classifier(data, cfg, weights);
            else if (args[2] == "valid") validate_classifier_single(data, cfg, weights);
            else if (args[2] == "validmulti") validate_classifier_multi(data, cfg, weights);
            else if (args[2] == "valid10") validate_classifier_10(data, cfg, weights);
            else if (args[2] == "validcrop") validate_classifier_crop(data, cfg, weights);
            else if (args[2] == "validfull") validate_classifier_full(data, cfg, weights);
        }
    }
}