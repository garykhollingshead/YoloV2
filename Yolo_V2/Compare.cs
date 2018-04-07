using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Compare
    {
        private static int TotalCompares;
        private static int CurrentClass;

        private static void train_compare(string cfgfile, string weightfile)
        {

            float avgLoss = -1;
            string basec = Utils.Basecfg(cfgfile);
            string backupDirectory = "/home/pjreddie/backup/";
            Console.Write($"%s\n", basec);
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            string[] paths = Data.Data.GetPaths("Data.Data/compare.train.list");
            int n = paths.Length;
            Console.Write($"%d\n", n);
            var sw = new Stopwatch();
            Thread loadThread;
            Data.Data train;
            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Paths = paths;
            args.Classes = 20;
            args.N = imgs;
            args.M = n;
            args.D = buffer;
            args.Type = DataType.CompareData;

            loadThread = Data.Data.load_data_in_thread(args);
            int epoch = net.Seen / n;
            int i = 0;
            while (true)
            {
                ++i;
                sw.Reset();
                sw.Start();
                loadThread.Join();
                train = buffer;

                loadThread = Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded: %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);
                if (avgLoss == -1) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;
                sw.Stop();
                Console.Write($"%.3f: %f, %f avg, %lf seconds, %d images\n", (float)net.Seen / n, loss, avgLoss, sw.Elapsed.Seconds, net.Seen);
                if (i % 100 == 0)
                {

                    string buff = $"{backupDirectory}/{basec}_{epoch}_minor_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (net.Seen / n > epoch)
                {
                    epoch = net.Seen / n;
                    i = 0;

                    string buff = $"{backupDirectory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                    if (epoch % 22 == 0) net.LearningRate *= .1f;
                }
            }
            loadThread.Join();
        }

        private static void validate_compare(string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            string[] paths = Data.Data.GetPaths("Data.Data/compare.val.list");
            int n = paths.Length / 2;

            var sw = new Stopwatch();
            int correct = 0;
            int total = 0;
            int splits = 10;
            int num = (i + 1) * n / splits - i * n / splits;

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

            Thread loadThread = Data.Data.load_data_in_thread(args);
            for (i = 1; i <= splits; ++i)
            {
                sw.Reset();
                sw.Start();

                loadThread.Join();
                var val = buffer;

                num = (i + 1) * n / splits - i * n / splits;
                string[] part = new string[paths.Length - (i * n / splits)];
                Array.Copy(paths, i * n / splits, part, 0, part.Length);
                if (i != splits)
                {
                    args.Paths = part;
                    loadThread = Data.Data.load_data_in_thread(args);
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

        private static int elo_comparator(SortableBbox a, SortableBbox b)
        {
            if (a.Elos[CurrentClass] == b.Elos[CurrentClass]) return 0;
            if (a.Elos[CurrentClass] > b.Elos[CurrentClass]) return -1;
            return 1;
        }

        private static int bbox_comparator(SortableBbox a, SortableBbox b)
        {
            ++TotalCompares;
            Network net = a.Net;
            int sclass = a.Sclass;

            Mat im1 = LoadArgs.load_image_color(a.Filename, net.W, net.H);
            Mat im2 = LoadArgs.load_image_color(b.Filename, net.W, net.H);
            float[] x = new float[net.W * net.H * net.C];
            Array.Copy(im1.Data, 0, x, 0, im1.Data.Length);
            Array.Copy(im2.Data, 0, x, im1.Data.Length, im2.Data.Length);
            float[] predictions = Network.network_predict(net, x);

            if (predictions[sclass * 2] > predictions[sclass * 2 + 1])
            {
                return 1;
            }
            return -1;
        }

        private static void bbox_update(SortableBbox a, SortableBbox b, int sclass, bool result)
        {
            int k = 32;
            float ea = 1.0f / (1 + (float)Math.Pow(10, (b.Elos[sclass] - a.Elos[sclass]) / 400f));
            float eb = 1.0f / (1 + (float)Math.Pow(10, (a.Elos[sclass] - b.Elos[sclass]) / 400f));
            float sa = result ? 1 : 0;
            float sb = result ? 0 : 1;
            a.Elos[sclass] += k * (sa - ea);
            b.Elos[sclass] += k * (sb - eb);
        }

        private static void bbox_fight(Network net, SortableBbox a, SortableBbox b, int classes, int sclass)
        {
            Mat im1 = LoadArgs.load_image_color(a.Filename, net.W, net.H);
            Mat im2 = LoadArgs.load_image_color(b.Filename, net.W, net.H);
            float[] x = new float[net.W * net.H * net.C];
            Array.Copy(im1.Data, 0, x, 0, im1.Data.Length);
            Array.Copy(im2.Data, 0, x, im1.Data.Length, im2.Data.Length);
            float[] predictions = Network.network_predict(net, x);
            ++TotalCompares;

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

        private static void SortMaster3000(string filename, string weightfile)
        {
            int i = 0;
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            Network.set_batch_network(net, 1);

            string[] paths = Data.Data.GetPaths("Data.Data/compare.sort.list");
            int n = paths.Length;
            SortableBbox[] boxes = new SortableBbox[n];
            Console.Write($"Sorting %d boxes...\n", n);
            for (i = 0; i < n; ++i)
            {
                boxes[i].Filename = paths[i];
                boxes[i].Net = net;
                boxes[i].Sclass = 7;
                boxes[i].Elo = 1500;
            }
            var sw = new Stopwatch();
            sw.Start();
            Array.Sort(boxes, bbox_comparator);
            for (i = 0; i < n; ++i)
            {
                Console.Write($"%s\n", boxes[i].Filename);
            }
            sw.Stop();
            Console.Write($"Sorted ini %d compares, %f secs\n", TotalCompares, sw.Elapsed.Seconds);
        }

        private static void BattleRoyaleWithCheese(string filename, string weightfile)
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
            int n = paths.Length;
            int total = n;
            SortableBbox[] boxes = new SortableBbox[n];
            Console.Write($"Battling %d boxes...\n", n);
            for (i = 0; i < n; ++i)
            {
                boxes[i].Filename = paths[i];
                boxes[i].Net = net;
                boxes[i].Classes = classes;
                boxes[i].Elos = new float[classes];
                for (j = 0; j < classes; ++j)
                {
                    boxes[i].Elos[j] = 1500;
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
                for (i = 0; i < n / 2; ++i)
                {
                    bbox_fight(net, boxes[i * 2], boxes[i * 2 + 1], classes, -1);
                }
                sw.Stop();
                Console.Write($"Round: %f secs, %d remaining\n", sw.Elapsed.Seconds, n);
            }

            for (var iClass = 0; iClass < classes; ++iClass)
            {

                n = total;
                CurrentClass = iClass;
                Array.Sort(boxes, elo_comparator);
                n /= 2;

                for (round = 1; round <= 100; ++round)
                {
                    var sw = new Stopwatch();
                    sw.Start();
                    Console.Write($"Round: %d\n", round);

                    Utils.sorta_shuffle(boxes, 10);
                    for (i = 0; i < n / 2; ++i)
                    {
                        bbox_fight(net, boxes[ i * 2], boxes[ i * 2 + 1], classes, iClass);
                    }
                    Array.Sort(boxes, elo_comparator);
                    if (round <= 20) n = (n * 9 / 10) / 2 * 2;

                    sw.Stop();
                    Console.Write($"Round: %f secs, %d remaining\n", sw.Elapsed.Seconds, n);
                }

                string buff = $"results/battle_{iClass}.(float)Math.Log";
                var lines = new List<string>();
                for (i = 0; i < n; ++i)
                {
                    lines.Add($"{boxes[i].Filename} {boxes[i].Elos[iClass]}");
                }
                File.WriteAllLines(buff, lines);
            }
            swTotal.Stop();
            Console.Write($"Tournament ini %d compares, %f secs\n", TotalCompares, swTotal.Elapsed.Seconds);
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
    }
}