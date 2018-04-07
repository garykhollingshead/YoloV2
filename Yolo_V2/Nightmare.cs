using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Yolo_V2.Data;

namespace Yolo_V2
{
    public static class Nightmare
    {
        private static void calculate_loss(float[] output, float[] delta, int n, float thresh)
        {
            int i;
            float mean = Utils.mean_array(output, n);
            float var = Utils.variance_array(output, n);
            for (i = 0; i < n; ++i)
            {
                if (delta[i] > mean + thresh * (float)Math.Sqrt(var)) delta[i] = output[i];
                else delta[i] = 0;
            }
        }

        private static void optimize_picture(Network net, Mat orig, int maxLayer, float scale, float rate, float thresh, bool norm)
        {
            net.N = maxLayer + 1;

            int dx = Utils.Rand.Next() % 16 - 8;
            int dy = Utils.Rand.Next() % 16 - 8;
            bool flip = Utils.Rand.Next() % 2 != 0;

            Mat im = new Mat(orig, new Rectangle(dx, dy, orig.Width - dx, orig.Height - dy));
            CvInvoke.Resize(im, im, new Size((int)(orig.Width * scale), (int)(orig.Height * scale)));
            if (flip)
                CvInvoke.Flip(im, im, FlipType.Horizontal);

            Network.resize_network(net, im.Width, im.Height);
            Layer last = net.Layers[net.N - 1];

            Mat delta = new Mat(im.Width, im.Height, im.NumberOfChannels);

            NetworkState state = new NetworkState();

            var copy = new Mat(im);
            state.Input = (float[])copy.Data.Clone();
            state.Delta = (float[])copy.Data.Clone();

            Network.forward_network_gpu(net, state);
            Blas.copy_ongpu(last.Outputs, last.OutputGpu, last.DeltaGpu);

            Array.Copy(last.DeltaGpu, last.Delta, last.Outputs);
            calculate_loss(last.Delta, last.Delta, last.Outputs, thresh);
            Array.Copy(last.Delta, last.DeltaGpu, last.Outputs);

            Network.backward_network_gpu(net, state);

            Array.Copy(state.Delta, delta.Data, im.Width * im.Height * im.NumberOfChannels);

            using (var outi = delta.ToMat())
            {
                if (flip) CvInvoke.Flip(outi, outi, FlipType.Horizontal);

                CvInvoke.Resize(outi, outi, new Size(orig.Width, orig.Height));
                using (var outCrop = new Mat(outi, new Rectangle(-dx, -dy, orig.Width + dx, orig.Height + dy)))
                {
                    outi.SetTo(outCrop);
                }
                if (norm) CvInvoke.Normalize(outi, outi);
                var bytes = orig.GetData();
                Blas.Axpy_cpu(orig.Width * orig.Height * orig.NumberOfChannels, rate, outi.GetData(), bytes);
                delta = new Mat(orig.Width, orig.Height, orig.NumberOfChannels, bytes);
                Array.Copy(bytes, 0, state.Delta, 0, im.Width * im.Height * im.NumberOfChannels);
            }

            LoadArgs.constrain_image(orig);
        }

        private static void Smooth(Mat recon, Mat update, float lambda, int num)
        {
            int i, j, k;
            int ii, jj;
            var updateData = update.GetData();
            var reconData = recon.GetData();
            for (k = 0; k < recon.NumberOfChannels; ++k)
            {
                for (j = 0; j < recon.Height; ++j)
                {
                    for (i = 0; i < recon.Width; ++i)
                    {
                        int outIndex = i + recon.Width * (j + recon.Height * k);
                        for (jj = j - num; jj <= j + num && jj < recon.Height; ++jj)
                        {
                            if (jj < 0) continue;
                            for (ii = i - num; ii <= i + num && ii < recon.Width; ++ii)
                            {
                                if (ii < 0) continue;
                                int inIndex = ii + recon.Width * (jj + recon.Height * k);
                                updateData[outIndex] += (byte)(lambda * (reconData[inIndex] - reconData[outIndex]));
                            }
                        }
                    }
                }
            }
            update.SetTo(updateData);
        }

        public static void reconstruct_picture(Network net, float[] features, Mat recon, Mat update, float rate, float momentum, float lambda, int smoothSize, int iters)
        {
            for (var iter = 0; iter < iters; ++iter)
            {
                Mat delta = new Mat(recon.Width, recon.Height, recon.NumberOfChannels);

                NetworkState state = new NetworkState();
                state.Input = (float[])recon.Data.Clone();
                state.Delta = new float[delta.Data.Length];
                state.Truth = new float[Network.get_network_output_size(net)];
                Array.Copy(features, 0, state.Truth, 0, state.Truth.Length);

                Network.forward_network_gpu(net, state);
                Network.backward_network_gpu(net, state);

                Array.Copy(state.Delta, delta.Data, delta.W * delta.H * delta.C);
                var updateFloats = Mat.GetFloats(update.GetData());
                Blas.Axpy_cpu(recon.Width * recon.Height * recon.NumberOfChannels, 1, delta.Data, updateFloats);
                update.SetTo(Mat.GetBytes(updateFloats));
                Smooth(recon, update, lambda, smoothSize);
                var ubytes = update.GetData();
                var rbytes = recon.GetData();
                Blas.Axpy_cpu(recon.Width * recon.Height * recon.NumberOfChannels, rate, ubytes, rbytes);
                Blas.Scal_cpu(recon.Width * recon.Height * recon.NumberOfChannels, momentum, ubytes, 1);
                recon.SetTo(rbytes);
                update.SetTo(ubytes);
                LoadArgs.constrain_image(recon);
            }
        }

        public static void run_nightmare(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [cfg] [weights] [Mat] [Layer] [options! (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[2];
            string weights = args[3];
            string input = args[4];
            int maxLayer = int.Parse(args[5]);

            int range = Utils.find_value_arg(args, "-range", 1);
            bool norm = Utils.find_value_arg(args, "-norm", 1) != 0;
            int rounds = Utils.find_value_arg(args, "-rounds", 1);
            int iters = Utils.find_value_arg(args, "-iters", 10);
            int octaves = Utils.find_value_arg(args, "-octaves", 4);
            float zoom = Utils.find_value_arg(args, "-zoom", 1);
            float rate = Utils.find_value_arg(args, "-rate", .04f);
            float thresh = Utils.find_value_arg(args, "-thresh", 1);
            float rotate = Utils.find_value_arg(args, "-rotate", 0);
            float momentum = Utils.find_value_arg(args, "-momentum", .9f);
            float lambda = Utils.find_value_arg(args, "-lambda", .01f);
            string prefix = Utils.find_value_arg(args, "-prefix", "");
            bool reconstruct = Utils.find_arg(args, "-reconstruct");
            int smoothSize = Utils.find_value_arg(args, "-smooth", 1);

            Network net = Parser.parse_network_cfg(cfg);
            Parser.load_weights(net, weights);
            string cfgbase = Utils.Basecfg(cfg);
            string imbase = Utils.Basecfg(input);

            Network.set_batch_network(net, 1);
            Mat im = LoadArgs.load_image_color(input, 0, 0).ToMat();

            float[] features = new float[0];
            Mat update = null;
            if (reconstruct)
            {
                Network.resize_network(net, im.Width, im.Height);

                int zz = 0;
                var imData = Mat.GetFloats(im.GetData());
                Network.network_predict(net, imData);
                Mat outIm = Network.get_network_image(net);
                Mat crop = LoadArgs.crop_image(outIm, zz, zz, outIm.W - 2 * zz, outIm.H - 2 * zz);
                Mat fIm = LoadArgs.resize_image(crop, outIm.W, outIm.H);
                Console.Write($"%d features\n", outIm.W * outIm.H * outIm.C);

                CvInvoke.Resize(im, im, new Size(im.Width, im.Height));
                fIm = LoadArgs.resize_image(fIm, fIm.W, fIm.H);
                features = fIm.Data;

                int i;
                for (i = 0; i < 14 * 14 * 512; ++i)
                {
                    features[i] += Utils.rand_uniform(-.19f, .19f);
                }

                im = LoadArgs.make_random_image(im.W, im.H, im.C);
                update = new Mat(im.W, im.H, im.C).ToMat();

            }

            int e;
            int n;
            for (e = 0; e < rounds; ++e)
            {
                Console.Error.Write($"Iteration: ");
                for (n = 0; n < iters; ++n)
                {
                    Console.Error.Write($"%d, ", n);
                    if (reconstruct)
                    {
                        reconstruct_picture(net, features, im, update, rate, momentum, lambda, smoothSize, 1);
                        //if ((n+1)%30 == 0) rate *= .5;
                        LoadArgs.show_image(im, "reconstruction");
                        CvInvoke.WaitKey(10);
                    }
                    else
                    {
                        int layer = maxLayer + Utils.Rand.Next() % range - range / 2;
                        int octave = Utils.Rand.Next() % octaves;
                        optimize_picture(net, im, layer, 1 / (float)Math.Pow(1.33333333, octave), rate, thresh, norm);
                    }
                }
                Console.Error.Write($"done\n");
                string buff;
                if (!string.IsNullOrEmpty(prefix))
                {
                    buff = $"{prefix}_{imbase}_{cfgbase}_{maxLayer}_{e:D6}";
                }
                else
                {
                    buff = $"{imbase}_{cfgbase}_{maxLayer}_{e:D6}";
                }
                Console.Write($"%d %s\n", e, buff);
                LoadArgs.save_image(im, buff);
                
                CvInvoke.WaitKey();

                if (rotate != 0)
                {
                    Mat rot = LoadArgs.rotate_image(im, rotate);
                    im = rot;
                }
                Mat crop = LoadArgs.crop_image(im, (int)(im.W * (1f - zoom) / 2f), (int)(im.H * (1f - zoom) / 2f), (int)(im.W * zoom), (int)(im.H * zoom));
                Mat resized = LoadArgs.resize_image(crop, im.W, im.H);
                im = resized;
            }
        }
    }
}