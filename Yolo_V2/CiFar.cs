using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Yolo_V2.Data;

namespace Yolo_V2
{
    public static class CiFar
    {
        private static void train_cifar(string cfgfile, string weightfile)
        {

            float avgLoss = -1;
            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"{basec}\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: {net.LearningRate}, Momentum: {net.Momentum}, Decay: {net.Decay}\n");

            string backupDirectory = "/home/pjreddie/backup/";
            int n = 50000;

            Data.Data.get_labels("Data.Data/cifar/labels.txt");
            int epoch = (net.Seen) / n;
            Data.Data train = Data.Data.load_all_cifar10();
            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();

                float loss = Network.train_network_sgd(net, train, 1);
                if (avgLoss == -1) avgLoss = loss;
                avgLoss = avgLoss * .95f + loss * .05f;
                sw.Stop();
                Console.Write(
                    $"{Network.get_current_batch(net)}, {net.Seen / n:F3}: {loss}, {avgLoss} avg, {Network.get_current_rate(net)} rate, {sw.Elapsed.Seconds} seconds, {net.Seen} images\n");
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

        private static void train_cifar_distill(string cfgfile, string weightfile)
        {

            float avgLoss = -1;
            string basec = Utils.Basecfg(cfgfile);
            Console.Write($"{basec}\n");
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Console.Write($"Learning Rate: {net.LearningRate}, Momentum: {net.Momentum}, Decay: {net.Decay}\n");

            string backupDirectory = "/home/pjreddie/backup/";
            int n = 50000;

            Data.Data.get_labels("Data.Data/cifar/labels.txt");
            int epoch = (net.Seen) / n;

            Data.Data train = Data.Data.load_all_cifar10();
            Matrix soft = new Matrix("results/ensemble.csv");

            float weight = .9f;
            soft.scale_matrix(weight);
            train.Y.scale_matrix(1.0f - weight);
            Matrix.matrix_add_matrix(soft, train.Y);

            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches || net.MaxBatches == 0)
            {
                sw.Reset();
                sw.Start();

                float loss = Network.train_network_sgd(net, train, 1);
                if (avgLoss == -1) avgLoss = loss;
                avgLoss = avgLoss * .95f + loss * .05f;
                sw.Stop();
                Console.Write(
                    $"{Network.get_current_batch(net)}, {net.Seen / n:F3}: {loss}, {avgLoss} avg, {Network.get_current_rate(net)} rate, {sw.Elapsed.Seconds} seconds, {net.Seen} images\n");
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

        private static void test_cifar_multi(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);


            float avgAcc = 0;
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            int i;
            for (i = 0; i < test.X.Rows; ++i)
            {
                Mat im = new Mat(new Size(32, 32), DepthType.Cv8U, 3);
                im.SetTo(test.X.Vals[i]);

                byte[] pred = new byte[10];

                byte[] p = Network.network_predict(net, im.GetData());
                Blas.Axpy_cpu(10, 1, p, pred);
                im.SetTo(p);
                LoadArgs.flip_image(im);
                p = Network.network_predict(net, im.GetData());
                Blas.Axpy_cpu(10, 1, p, pred);

                int index = Utils.max_index(pred, 10);
                int sclass = Utils.max_index(test.Y.Vals[i], 10);
                if (index == sclass) avgAcc += 1;
                Console.Write($"{i}: {avgAcc / (i + 1):P}\n");
            }
        }

        private static void test_cifar(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            var sw = new Stopwatch();
            float avgAcc = 0;
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            sw.Start();

            float[] acc = Network.network_accuracies(net, test, 2);
            avgAcc += acc[0];
            sw.Stop();
            Console.Write($"top1: {avgAcc}, {sw.Elapsed.Seconds} seconds, {test.X.Rows} images\n");
        }

        private static void extract_cifar()
        {
            string[] labels = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
            int i;
            Data.Data train = Data.Data.load_all_cifar10();
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");
            for (i = 0; i < train.X.Rows; ++i)
            {
                using (Mat im = new Mat(new Size(32, 32), DepthType.Cv8U, 3))
                {
                    im.SetTo(train.X.Vals[i]);
                    int sclass = Utils.max_index(train.Y.Vals[i], 10);
                    string buff = $"Data.Data/cifar/train/{i}_{labels[sclass]}";
                    LoadArgs.save_image_png(im, buff);
                }
            }
            for (i = 0; i < test.X.Rows; ++i)
            {
                using (Mat im = new Mat(new Size(32, 32), DepthType.Cv8U, 3))
                {
                    im.SetTo(train.X.Vals[i]);
                    int sclass = Utils.max_index(test.Y.Vals[i], 10);
                    string buff = $"Data.Data/cifar/test/{i}_{labels[sclass]}";
                    LoadArgs.save_image_png(im, buff);
                }
            }
        }

        private static void test_cifar_csv(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            Matrix pred = Network.network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.Rows; ++i)
            {
                using (Mat im = new Mat(new Size(32, 32), DepthType.Cv8U, 3))
                {
                    im.SetTo(test.X.Vals[i]);
                    LoadArgs.flip_image(im);
                }
            }
            Matrix pred2 = Network.network_predict_data(net, test);
            pred.scale_matrix(.5f);
            pred2.scale_matrix(.5f);
            Matrix.matrix_add_matrix(pred2, pred);

            pred.to_csv();
            Console.Error.Write($"Accuracy: {Matrix.matrix_topk_accuracy(test.Y, pred, 1)}\n");
        }

        private static void test_cifar_csvtrain(string filename, string weightfile)
        {
            Network net = Parser.parse_network_cfg(filename);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }


            Data.Data test = Data.Data.load_all_cifar10();

            Matrix pred = Network.network_predict_data(net, test);

            int i;
            for (i = 0; i < test.X.Rows; ++i)
            {
                using (Mat im = new Mat(new Size(32, 32), DepthType.Cv8U, 3))
                {
                    im.SetTo(test.X.Vals[i]);
                    LoadArgs.flip_image(im);
                }
            }
            Matrix pred2 = Network.network_predict_data(net, test);
            pred.scale_matrix(.5f);
            pred2.scale_matrix(.5f);
            Matrix.matrix_add_matrix(pred2, pred);

            pred.to_csv();
            Console.Error.Write($"Accuracy: %f\n", Matrix.matrix_topk_accuracy(test.Y, pred, 1));
        }

        private static void eval_cifar_csv()
        {
            Data.Data test = Data.Data.load_cifar10_data("Data.Data/cifar/cifar-10-batches-bin/test_batch.bin");

            Matrix pred = new Matrix("results/combined.csv");
            Console.Error.Write($"%d %d\n", pred.Rows, pred.Cols);

            Console.Error.Write($"Accuracy: %f\n", Matrix.matrix_topk_accuracy(test.Y, pred, 1));
        }

        public static void run_cifar(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_cifar(cfg, weights);
            else if (args[2] == "extract") extract_cifar();
            else if (args[2] == "distill") train_cifar_distill(cfg, weights);
            else if (args[2] == "test") test_cifar(cfg, weights);
            else if (args[2] == "multi") test_cifar_multi(cfg, weights);
            else if (args[2] == "csv") test_cifar_csv(cfg, weights);
            else if (args[2] == "csvtrain") test_cifar_csvtrain(cfg, weights);
            else if (args[2] == "eval") eval_cifar_csv();
        }
    }
}