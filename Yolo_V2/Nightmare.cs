using System;
using System.Collections.Generic;
using Emgu.CV;
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

        private static void optimize_picture(Network net, Image orig, int maxLayer, float scale, float rate, float thresh, bool norm)
        {
            net.N = maxLayer + 1;

            int dx = Utils.Rand.Next() % 16 - 8;
            int dy = Utils.Rand.Next() % 16 - 8;
            bool flip = Utils.Rand.Next() % 2 != 0;

            Image crop = LoadArgs.crop_image(orig, dx, dy, orig.Width, orig.Height);
            Image im = LoadArgs.resize_image(crop, (int)(orig.Width * scale), (int)(orig.Height * scale));
            if (flip) LoadArgs.flip_image(ref im);

            Network.resize_network(ref net, im.Width, im.Height);
            Layer last = net.Layers[net.N - 1];

            Image delta = new Image(im.Width, im.Height, im.NumberOfChannels);

            NetworkState state = new NetworkState();
            state.Input = (float[])im.Data.Clone();
            state.Delta = (float[])im.Data.Clone();

            Network.forward_network_gpu(ref net, ref state);
            Blas.copy_ongpu(last.Outputs, last.OutputGpu, ref last.DeltaGpu);

            Array.Copy(last.DeltaGpu, last.Delta, last.Outputs);
            calculate_loss(last.Delta, last.Delta, last.Outputs, thresh);
            Array.Copy(last.Delta, last.DeltaGpu, last.Outputs);

            Network.backward_network_gpu(ref net, ref state);

            Array.Copy(state.Delta, delta.Data, im.Width * im.Height * im.NumberOfChannels);


            if (flip) LoadArgs.flip_image(ref delta);

            Image resized = LoadArgs.resize_image(delta, orig.Width, orig.Height);
            Image outi = LoadArgs.crop_image(resized, -dx, -dy, orig.Width, orig.Height);

            if (norm) Utils.normalize_array(ref outi.Data, outi.Width * outi.Height * outi.NumberOfChannels);
            Blas.Axpy_cpu(orig.Width * orig.Height * orig.NumberOfChannels, rate, outi.Data, orig.Data);
            
            LoadArgs.constrain_image(orig);
        }

        private static void Smooth(Image recon, Image update, float lambda, int num)
        {
            int i, j, k;
            int ii, jj;
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
                                update.Data[outIndex] += lambda * (recon.Data[inIndex] - recon.Data[outIndex]);
                            }
                        }
                    }
                }
            }
        }

        public static void reconstruct_picture(Network net, float[] features, Image recon, Image update, float rate, float momentum, float lambda, int smoothSize, int iters)
        {
            int iter = 0;
            for (iter = 0; iter < iters; ++iter)
            {
                Image delta = new Image(recon.Width, recon.Height, recon.NumberOfChannels);

                NetworkState state = new NetworkState();
                state.Input = (float[])recon.Data.Clone();
                state.Delta = (float[])delta.Data.Clone();
                state.Truth = new float[Network.get_network_output_size(net)];
                Array.Copy(features, 0, state.Truth, 0, state.Truth.Length);

                Network.forward_network_gpu(ref net, ref state);
                Network.backward_network_gpu(ref net, ref state);

                Array.Copy(state.Delta, delta.Data, delta.Width * delta.Height * delta.NumberOfChannels);

                Blas.Axpy_cpu(recon.Width * recon.Height * recon.NumberOfChannels, 1, delta.Data, update.Data);
                Smooth(recon, update, lambda, smoothSize);

                Blas.Axpy_cpu(recon.Width * recon.Height * recon.NumberOfChannels, rate, update.Data, recon.Data);
                Blas.Scal_cpu(recon.Width * recon.Height * recon.NumberOfChannels, momentum, update.Data, 1);

                LoadArgs.constrain_image(recon);
            }
        }

        public static void run_nightmare(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [cfg] [weights] [Image] [Layer] [options! (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[2];
            string weights = args[3];
            string input = args[4];
            int maxLayer = int.Parse(args[5]);

            int range = Utils.find_int_arg(args, "-range", 1);
            bool norm = Utils.find_int_arg(args, "-norm", 1) != 0;
            int rounds = Utils.find_int_arg(args, "-rounds", 1);
            int iters = Utils.find_int_arg(args, "-iters", 10);
            int octaves = Utils.find_int_arg(args, "-octaves", 4);
            float zoom = Utils.find_int_arg(args, "-zoom", 1);
            float rate = Utils.find_int_arg(args, "-rate", .04f);
            float thresh = Utils.find_int_arg(args, "-thresh", 1);
            float rotate = Utils.find_int_arg(args, "-rotate", 0);
            float momentum = Utils.find_int_arg(args, "-momentum", .9f);
            float lambda = Utils.find_int_arg(args, "-lambda", .01f);
            string prefix = Utils.find_int_arg(args, "-prefix", "");
            bool reconstruct = Utils.find_arg(args, "-reconstruct");
            int smoothSize = Utils.find_int_arg(args, "-smooth", 1);

            Network net = Parser.parse_network_cfg(cfg);
            Parser.load_weights(net, weights);
            string cfgbase = Utils.Basecfg(cfg);
            string imbase = Utils.Basecfg(input);

            Network.set_batch_network(ref net, 1);
            Image im = LoadArgs.load_image_color(input, 0, 0);
            
            float[] features = new float[0];
            Image update = null;
            if (reconstruct)
            {
                Network.resize_network(ref net, im.Width, im.Height);

                int zz = 0;
                Network.network_predict(ref net, ref im.Data);
                Image outIm = Network.get_network_image(net);
                Image crop = LoadArgs.crop_image(outIm, zz, zz, outIm.Width - 2 * zz, outIm.Height - 2 * zz);
                Image fIm = LoadArgs.resize_image(crop, outIm.Width, outIm.Height);
                Console.Write($"%d features\n", outIm.Width * outIm.Height * outIm.NumberOfChannels);


                im = LoadArgs.resize_image(im, im.Width, im.Height);
                fIm = LoadArgs.resize_image(fIm, fIm.Width, fIm.Height);
                features = fIm.Data;

                int i;
                for (i = 0; i < 14 * 14 * 512; ++i)
                {
                    features[i] += Utils.rand_uniform(-.19f, .19f);
                }

                im = LoadArgs.make_random_image(im.Width, im.Height, im.NumberOfChannels);
                update = new Image(im.Width, im.Height, im.NumberOfChannels);

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
                //LoadArgs.show_image(im, buff);
                //CvInvoke.WaitKey();

                if (rotate != 0)
                {
                    Image rot = LoadArgs.rotate_image(im, rotate);
                    im = rot;
                }
                Image crop = LoadArgs.crop_image(im, (int)(im.Width * (1f - zoom) / 2f), (int)(im.Height * (1f - zoom) / 2f), (int)(im.Width * zoom), (int)(im.Height * zoom));
                Image resized = LoadArgs.resize_image(crop, im.Width, im.Height);
                im = resized;
            }
        }
    }
}