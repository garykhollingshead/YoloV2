using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Emgu.CV;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Writing
    {
        private static void train_writing(string cfgfile, string weightfile)
        {
            string backupDirectory = "/home/pjreddie/backup/";

            float avgLoss = -1;
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
            int n = paths.Length;
            Console.Write($"N: %d\n", n);
            Image outf = Network.get_network_image(net);

            Data.Data buffer = new Data.Data();

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.OutW = outf.Width;
            args.OutH = outf.Height;
            args.Paths = paths;
            args.N = imgs;
            args.M = n;
            args.D = buffer;
            args.Type = DataType.WritingData;

            Data.Data.load_data_in_thread(args);
            int epoch = (net.Seen) / n;
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();
                var train = buffer;
                Data.Data.load_data_in_thread(args);
                sw.Stop();
                Console.Write($"Loaded %lf seconds\n", sw.Elapsed.Seconds);

                sw.Reset();
                sw.Start();
                float loss = Network.train_network(net, train);


                if (avgLoss == -1) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;
                Console.Write($"%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", Network.get_current_batch(net), (float)(net.Seen) / n, loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds, net.Seen);
                if (Network.get_current_batch(net) % 100 == 0)
                {

                    string buff = $"{backupDirectory}/{basec}_batch_{Network.get_current_batch(net)}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (net.Seen / n > epoch)
                {
                    epoch = net.Seen / n;

                    string buff = $"{backupDirectory}/{basec}_{epoch}.Weights";
                    Parser.save_weights(net, buff);
                }
            }
        }

        private static void test_writing(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(ref net, 1);
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
                Network.resize_network(ref net, im.Width, im.Height);
                Console.Write($"%d %d %d\n", im.Height, im.Width, im.NumberOfChannels);
                float[] x = im.Data;
                sw.Reset();
                sw.Start();
                Network.network_predict(ref net, ref x);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                Image pred = Network.get_network_image(net);

                Image upsampled = LoadArgs.resize_image(pred, im.Width, im.Height);
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
    }
}