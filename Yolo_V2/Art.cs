using System;
using System.Collections.Generic;
using Emgu.CV;
using Yolo_V2.Data;

namespace Yolo_V2
{
    public static class Art
    {
        private static void demo_art(string cfgfile, string weightfile, int camIndex)
        {
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);

            Utils.Rand = new Random(2222222);
            using (VideoCapture cap = new VideoCapture(camIndex))
            {
                string window = "ArtJudgementBot9000!!!";
                if (cap != null) Utils.Error("Couldn't connect to webcam.\n");
                int i;
                int[] idx = {37, 401, 434};
                int n = idx.Length;

                while (true)
                {
                    Image ini = LoadArgs.get_image_from_stream(cap);
                    Image inS = LoadArgs.resize_image(ini, net.W, net.H);
                    LoadArgs.show_image(ini, window);

                    float[] p = Network.network_predict(net, inS.Data);

                    Console.Write($"\033[2J");
                    Console.Write($"\033[1;1H");

                    float score = 0;
                    for (i = 0; i < n; ++i)
                    {
                        float s = p[idx[i]];
                        if (s > score) score = s;
                    }

                    Console.Write($"I APPRECIATE THIS ARTWORK: %10.7f%%\n", score * 100);
                    Console.Write($"[");
                    int upper = 30;
                    for (i = 0; i < upper; ++i)
                    {
                        Console.Write($"%c", ((i + .5) < score * upper) ? 219 : ' ');
                    }

                    Console.Write($"]\n");

                    CvInvoke.WaitKey(1);
                }
            }
        }

        public static void run_art(List<string> args)
        {
            int camIndex = Utils.find_int_arg(args, "-c", 0);
            string cfg = args[2];
            string weights = args[3];
            demo_art(cfg, weights, camIndex);
        }
    }
}