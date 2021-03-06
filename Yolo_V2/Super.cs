﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Super
    {
        private static void train_super(string cfgfile, string weightfile)
        {
            string trainImages = "/Data.Data/imagenet/imagenet1k.train.list";
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

            string[] paths = Data.Data.GetPaths(trainImages);

            LoadArgs args = new LoadArgs();
            args.W = net.W;
            args.H = net.H;
            args.Scale = 4;
            args.Paths = paths;
            args.N = imgs;
            args.M = paths.Length;
            args.D = buffer;
            args.Type = DataType.SuperData;

            Thread loadThread = Data.Data.load_data_in_thread(args);
            var sw = new Stopwatch();
            //while(i*imgs < N*120){
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
                if (i % 1000 == 0)
                {
                    string buffa = $"{backupDirectory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buffa);
                }
                if (i % 100 == 0)
                {
                    string buffb = $"{backupDirectory}/{basec}.backup";
                    Parser.save_weights(net, buffb);
                }
            }
            string buff = $"{backupDirectory}/{basec}_final.Weights";
            Parser.save_weights(net, buff);
        }

        private static void test_super(string cfgfile, string weightfile, string filename)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);
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
                Network.resize_network(net, im.W, im.H);
                Console.Write($"%d %d\n", im.W, im.H);

                float[] x = im.Data;
                sw.Start();
                Network.network_predict(net, x);
                Image outi = Network.get_network_image(net);
                sw.Stop();
                Console.Write($"%s: Predicted ini %f seconds.\n", input, sw.Elapsed.Seconds);
                LoadArgs.save_image(outi, "outf");

                if (!string.IsNullOrEmpty(filename)) break;
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
            string weights = (args.Count > 4) ? args[4] : "";
            string filename = (args.Count > 5) ? args[5] : "";
            if (args[2] == "train") train_super(cfg, weights);
            else if (args[2] == "test") test_super(cfg, weights, filename);
        }
    }
}