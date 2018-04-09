using System;
using System.Collections.Generic;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Yolo_V2.Data;

namespace Yolo_V2
{
    public static class RnnVid
    {
        private static FloatPair get_rnn_vid_data(Network net, string[] files, int n, int batch, int steps)
        {
            int b;
            Image outIm = Network.get_network_image(net);
            int outputSize = outIm.Width * outIm.Height * outIm.NumberOfChannels;
            Console.Write($"%d %d %d\n", outIm.Width, outIm.Height, outIm.NumberOfChannels);
            float[] feats = new float[net.Batch * batch * outputSize];
            for (b = 0; b < batch; ++b)
            {
                int inputSize = net.W * net.H * net.C;
                float[] input = new float[inputSize * net.Batch];
                string filename = files[Utils.Rand.Next() % n];
                using (VideoCapture cap = new VideoCapture(filename))
                {
                    int frames = (int)cap.GetCaptureProperty(CapProp.FrameCount);
                    int index = Utils.Rand.Next() % (frames - steps - 2);
                    if (frames < (steps + 4))
                    {
                        --b;
                        continue;
                    }

                    Console.Write($"frames: %d, index: %d\n", frames, index);
                    cap.SetCaptureProperty(CapProp.PosFrames, index);

                    int i;
                    for (i = 0; i < net.Batch; ++i)
                    {
                        using (Mat src = cap.QueryFrame())
                        {
                            Image im = new Image(src);

                            //LoadArgs.rgbgr_image(im);
                            Image re = LoadArgs.resize_image(im, net.W, net.H);
                            Array.Copy(re.Data, 0, input, i * inputSize, inputSize);
                        }
                    }

                    float[] output = Network.network_predict(ref net, ref input);

                    for (i = 0; i < net.Batch; ++i)
                    {
                        Array.Copy(output, i * outputSize, feats, (b + i * batch) * outputSize, outputSize);
                    }
                }
            }

            FloatPair p = new FloatPair();
            p.X = feats;
            p.Y = new float[feats.Length - outputSize * batch];
            Array.Copy(feats, outputSize * batch, p.Y, 0, p.Y.Length);

            return p;
        }

        private static void train_vid_rnn(string cfgfile, string weightfile)
        {
            string trainVideos = "Data.Data/vid/train.txt";
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

            string[] paths = Data.Data.GetPaths(trainVideos);
            int n = paths.Length;
            var sw = new Stopwatch();
            int steps = net.TimeSteps;
            int batch = net.Batch / net.TimeSteps;

            Network extractor = Parser.parse_network_cfg("cfg/extractor.cfg");
            Parser.load_weights(extractor, "/home/pjreddie/trained/yolo-coco.conv");

            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Reset();
                sw.Start();
                FloatPair p = get_rnn_vid_data(extractor, paths, n, batch, steps);

                float loss = Network.train_network_datum(net, p.X, p.Y) / (net.Batch);

                if (avgLoss < 0) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;

                sw.Stop();
                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds);
                if (i % 100 == 0)
                {

                    string buff = $"{backupDirectory}/{basec}_{i}.Weights";
                    Parser.save_weights(net, buff);
                }
                if (i % 10 == 0)
                {

                    string buff = $"{backupDirectory}/{basec}.backup";
                    Parser.save_weights(net, buff);
                }
            }

            string buff2 = $"{backupDirectory}/{basec}_final.Weights";
            Parser.save_weights(net, buff2);
        }

        private static Image save_reconstruction(Network net, Image init, float[] feat, string name, int i)
        {
            Image recon;
            if (init != null)
            {
                recon = new Image(init);
            }
            else
            {
                recon = LoadArgs.make_random_image(net.W, net.H, 3);
            }

            Image update = new Image(net.W, net.H, 3);
            Nightmare.reconstruct_picture(net, feat, recon, update, .01f, .9f, .1f, 2, 50);

            string buff = $"{name}{i}";
            LoadArgs.save_image(recon, buff);
            return recon;
        }

        private static void generate_vid_rnn(string cfgfile, string weightfile)
        {
            Network extractor = Parser.parse_network_cfg("cfg/extractor.recon.cfg");
            Parser.load_weights(extractor, "/home/pjreddie/trained/yolo-coco.conv");

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(ref extractor, 1);
            Network.set_batch_network(ref net, 1);

            int i;
            VideoCapture cap = new VideoCapture("/extra/vid/ILSVRC2015/Data.Data/VID/snippets/val/ILSVRC2015_val_00007030.mp4");
            float[] feat;
            float[] next;
            next = null;
            Image last = null;
            for (i = 0; i < 25; ++i)
            {
                Image im = LoadArgs.get_image_from_stream(cap);
                Image re = LoadArgs.resize_image(im, extractor.W, extractor.H);
                feat = Network.network_predict(ref extractor, ref re.Data);
                if (i > 0)
                {
                    Console.Write($"%f %f\n", Utils.mean_array(feat, 14 * 14 * 512), Utils.variance_array(feat, 14 * 14 * 512));
                    Console.Write($"%f %f\n", Utils.mean_array(next, 14 * 14 * 512), Utils.variance_array(next, 14 * 14 * 512));
                    Console.Write($"%f\n", Utils.mse_array(feat, 14 * 14 * 512));
                    Blas.Axpy_cpu(14 * 14 * 512, -1, feat, next);
                    Console.Write($"%f\n", Utils.mse_array(next, 14 * 14 * 512));
                }
                next = Network.network_predict(ref net, ref feat);
                if (i == 24) last = new Image(re);
            }
            for (i = 0; i < 30; ++i)
            {
                next = Network.network_predict(ref net, ref next);
                Image newi = save_reconstruction(extractor, last, next, "new", i);
                last = newi;
            }
        }

        public static void run_vid_rnn(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_vid_rnn(cfg, weights);
            else if (args[2] == "generate") generate_vid_rnn(cfg, weights);
        }
    }
}