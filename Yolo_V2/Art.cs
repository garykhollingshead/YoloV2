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
                    using (Mat ini = LoadArgs.get_image_from_stream(cap))
                    {
                        LoadArgs.show_image(ini, window);
                        CvInvoke.Resize(ini, ini, new System.Drawing.Size(net.W, net.H));

                        float[] p = Network.network_predict(net, ini.GetData());

                        Console.Clear();
                        Console.SetCursorPosition(1, 1);

                        float score = 0;
                        for (i = 0; i < n; ++i)
                        {
                            float s = p[idx[i]];
                            if (s > score) score = s;
                        }

                        Console.Write($"I APPRECIATE THIS ARTWORK: {score:P}\n");
                        Console.Write($"[");
                        int upper = 30;
                        for (i = 0; i < upper; ++i)
                        {
                            char c = ((i + .5) < (score * upper)) ? (char) 219 : ' ';
                            Console.Write($"{c}");
                        }

                        Console.Write($"]\n");

                        CvInvoke.WaitKey(1);
                    }
                }
            }
        }

        public static void run_art(List<string> args)
        {
            int camIndex = Utils.find_value_arg(args, "-c", 0);
            string cfg = args[2];
            string weights = args[3];
            demo_art(cfg, weights, camIndex);
        }
    }
}