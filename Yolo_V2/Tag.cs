using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Tag
    {
        private static void train_tag(string cfgfile, string weightfile, bool clear)
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
            if (clear) net.Seen = 0;
            Console.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int imgs = 1024;
            string[] paths = Data.Data.GetPaths("/home/pjreddie/tag/train.list");
            Console.Write($"%d\n", paths.Length);
            int n = paths.Length;
            var sw = new Stopwatch();
            Thread loadThread;
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
            args.M = n;
            args.D = buffer;
            args.Type = DataType.TagData;

            args.Angle = net.Angle;
            args.Exposure = net.Exposure;
            args.Saturation = net.Saturation;
            args.Hue = net.Hue;

            Console.Error.Write($"%d classes\n", net.Outputs);

            loadThread = Data.Data.load_data_in_thread(args);
            int epoch = (net.Seen) / n;
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
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
                Console.Write(
                    $"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), (float)(net.Seen) / n, loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds, net.Seen);

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

            loadThread.Join();
        }

        private static void test_tag(string cfgfile, string weightfile, string filename)
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

                float[] x = r.Data;

                sw.Reset();
                sw.Start();
                float[] predictions = Network.network_predict(net, x);
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
    }
}