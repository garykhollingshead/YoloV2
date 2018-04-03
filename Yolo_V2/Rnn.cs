using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Yolo_V2.Data;

namespace Yolo_V2
{
    class Rnn
    {
        public static int[] read_tokenized_data(string filename, out int read)
        {
            var lines = File.ReadAllLines(filename);
            int n;
            List<int> ns = new List<int>();
            foreach (var line in lines)
            {
                var tokens = line.Split(' ');
                foreach (var token in tokens)
                {
                    if (int.TryParse(token, out n))
                    {
                        ns.Add(n);
                    }
                    else
                    {
                        read = ns.Count;
                        return ns.ToArray();
                    }
                }
            }
            read = ns.Count;
            return ns.ToArray();
        }

        public static string[] read_tokens(string filename, out int read)
        {
            var lines = File.ReadAllLines(filename);
            read = lines.Length;
            return lines;
        }

        public static FloatPair get_rnn_token_data(int[] tokens, int[] offsets, int characters, int len, int batch, int steps)
        {
            float[] x = new float[batch * steps * characters];
            float[] y = new float[batch * steps * characters];
            int i, j;
            for (i = 0; i < batch; ++i)
            {
                for (j = 0; j < steps; ++j)
                {
                    int curr = tokens[(offsets[i]) % len];
                    int next = tokens[(offsets[i] + 1) % len];

                    x[(j * batch + i) * characters + curr] = 1;
                    y[(j * batch + i) * characters + next] = 1;

                    offsets[i] = (offsets[i] + 1) % len;

                    if (curr >= characters || curr < 0 || next >= characters || next < 0)
                    {
                        Utils.Error("Bad char");
                    }
                }
            }
            FloatPair p = new FloatPair();
            p.X = x;
            p.Y = y;
            return p;
        }

        public static FloatPair get_rnn_data(byte[] text, int[] offsets, int characters, int len, int batch, int steps)
        {
            float[] x = new float[batch * steps * characters];
            float[] y = new float[batch * steps * characters];
            int i, j;
            for (i = 0; i < batch; ++i)
            {
                for (j = 0; j < steps; ++j)
                {
                    byte curr = text[(offsets[i]) % len];
                    byte next = text[(offsets[i] + 1) % len];

                    x[(j * batch + i) * characters + curr] = 1;
                    y[(j * batch + i) * characters + next] = 1;

                    offsets[i] = (offsets[i] + 1) % len;

                    if (curr > 255 || curr <= 0 || next > 255 || next <= 0)
                    {
                        Utils.Error("Bad char");
                    }
                }
            }
            FloatPair p = new FloatPair();
            p.X = x;
            p.Y = y;
            return p;
        }

        public static void reset_rnn_state(Network net, int b)
        {
            int i;
            for (i = 0; i < net.N; ++i)
            {
                Layer l = net.Layers[i];
                if (l.StateGpu.Length != 0)
                {
                    Blas.fill_ongpu(l.Outputs, 0, l.StateGpu, 1, l.Outputs * b);
                }
            }
        }

        public static void train_char_rnn(string cfgfile, string weightfile, string filename, bool clear, bool tokenized)
        {

            byte[] text = new byte[0];
            int[] tokens = new int[0];
            int size;
            if (tokenized)
            {
                tokens = read_tokenized_data(filename, out size);
            }
            else
            {
                using (var fstream = File.OpenRead(filename))
                {
                    size = (int)fstream.Length;
                    text = new byte[size];
                    fstream.Read(text, 0, text.Length);
                }
            }

            string backupDirectory = "/home/pjreddie/backup/";
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);
            float avgLoss = -1;
            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }

            int inputs = Network.get_network_input_size(net);
            Console.Error.Write($"Learning Rate: %g, Momentum: %g, Decay: %g\n", net.LearningRate, net.Momentum, net.Decay);
            int batch = net.Batch;
            int steps = net.TimeSteps;
            if (clear) net.Seen = 0;
            int i = (net.Seen) / net.Batch;

            int streams = batch / steps;
            int[] offsets = new int[streams];
            int j;
            for (j = 0; j < streams; ++j)
            {
                offsets[j] = Utils.Rand.Next() % size;
            }

            var sw = new Stopwatch();
            while (Network.get_current_batch(net) < net.MaxBatches)
            {
                i += 1;
                sw.Reset();
                sw.Start();
                FloatPair p;
                if (tokenized)
                {
                    p = get_rnn_token_data(tokens, offsets, inputs, size, streams, steps);
                }
                else
                {
                    p = get_rnn_data(text, offsets, inputs, size, streams, steps);
                }

                float loss = Network.train_network_datum(net, p.X, p.Y) / (batch);
                if (avgLoss < 0) avgLoss = loss;
                avgLoss = avgLoss * .9f + loss * .1f;

                int chars = Network.get_current_batch(net) * batch;
                sw.Stop();
                Console.Error.Write($"%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avgLoss, Network.get_current_rate(net), sw.Elapsed.Seconds, (float)chars / size);

                for (j = 0; j < streams; ++j)
                {
                    //Console.Write($"%d\n", j);
                    if (Utils.Rand.Next() % 10 == 0)
                    {
                        //Console.Error.Write($"Reset\n");
                        offsets[j] = Utils.Rand.Next() % size;
                        reset_rnn_state(net, j);
                    }
                }

                if (i % 1000 == 0)
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

        public static void print_symbol(int n, string[] tokens)
        {
            if (!string.IsNullOrEmpty(tokens[n]))
            {
                Console.Write($"%s ", tokens[n]);
            }
            else
            {
                Console.Write($"%c", n);
            }
        }

        public static void test_char_rnn(string cfgfile, string weightfile, int num, string seed, float temp, int rseed, string tokenFile)
        {
            string[] tokens = new string[0];
            if (!string.IsNullOrEmpty(tokenFile))
            {
                int n;
                tokens = read_tokens(tokenFile, out n);
            }

            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].Temperature = temp;
            int c = 0;
            int len = seed.Length;
            float[] input = new float[inputs];


            for (i = 0; i < len - 1; ++i)
            {
                c = seed[i];
                input[c] = 1;
                Network.network_predict(net, input);
                input[c] = 0;
                print_symbol(c, tokens);
            }
            if (len != 0) c = seed[len - 1];
            print_symbol(c, tokens);
            for (i = 0; i < num; ++i)
            {
                input[c] = 1;
                float[] outf = Network.network_predict(net, input);
                input[c] = 0;
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                c = Utils.sample_array(outf, inputs);
                print_symbol(c, tokens);
            }
            Console.Write($"\n");
        }

        public static void test_tactic_rnn(string cfgfile, string weightfile, int num, float temp, int rseed, string tokenFile)
        {
            string[] tokens = new string[0];
            if (!string.IsNullOrEmpty(tokenFile))
            {
                int n;
                tokens = read_tokens(tokenFile, out n);
            }

            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int i, j;
            for (i = 0; i < net.N; ++i) net.Layers[i].Temperature = temp;
            float[] input = new float[inputs];
            float[] outf = new float[0];
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);
            foreach (var c in readLine)
            {
                input[c] = 1;
                outf = Network.network_predict(net, input);
                input[c] = 0;
            }
            for (i = 0; i < num; ++i)
            {
                var c = readLine.Last();
                for (j = 0; j < inputs; ++j)
                {
                    if (outf[j] < .0001) outf[j] = 0;
                }
                int next = Utils.sample_array(outf, inputs);
                if (c == '.' && next == '\n') break;
                c = (char)next;
                print_symbol(c, tokens);

                input[c] = 1;
                outf = Network.network_predict(net, input);
                input[c] = 0;
            }
            Console.Write($"\n");
        }

        public static void valid_tactic_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int count = 0;
            int words = 1;
            int c;
            int len = seed.Length;
            float[] input = new float[inputs];

            for (var i = 0; i < len; ++i)
            {
                c = seed[i];
                input[c] = 1;
                Network.network_predict(net, input);
                input[c] = 0;
            }
            float sum = 0;
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);
            float log2 = (float)Math.Log(2);
            bool iIn = false;
            for (int i = 0; i < readLine.Length - 1; i++)
            {
                c = readLine[i];
                var next = readLine[i + 1];
                if (next < 0 || next >= 255) Utils.Error("Out of range character");
                input[c] = 1;
                float[] outf = Network.network_predict(net, input);
                input[c] = 0;

                if (c == '.' && next == '\n') iIn = false;
                if (!iIn)
                {
                    if (c == '>' && next == '>')
                    {
                        iIn = true;
                        ++words;
                    }
                    continue;
                }
                ++count;
                sum += (float)Math.Log(outf[next]) / log2;
                Console.Write($"%d %d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, words, (float)Math.Pow(2, -sum / count), (float)Math.Pow(2, -sum / words));
            }
        }

        public static void valid_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int count = 0;
            int words = 1;
            int c;
            int len = seed.Length;
            float[] input = new float[inputs];
            for (var i = 0; i < len; ++i)
            {
                c = seed[i];
                input[c] = 1;
                Network.network_predict(net, input);
                input[c] = 0;
            }
            float sum = 0;
            float log2 = (float)Math.Log(2);
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);

            for (var i = 0; i < readLine.Length - 1; i++)
            {
                c = readLine[i];
                var next = readLine[i + 1];
                if (next < 0 || next >= 255) Utils.Error("Out of range character");
                ++count;
                if (next == ' ' || next == '\n' || next == '\t') ++words;
                input[c] = 1;
                float[] outf = Network.network_predict(net, input);
                input[c] = 0;
                sum += (float)Math.Log(outf[next]) / log2;
                Console.Write($"%d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, (float)Math.Pow(2, -sum / count), (float)Math.Pow(2, -sum / words));
            }
        }

        public static void vec_char_rnn(string cfgfile, string weightfile, string seed)
        {
            string basec = Utils.Basecfg(cfgfile);
            Console.Error.Write($"%s\n", basec);

            Network net = Parser.parse_network_cfg(cfgfile);
            if (string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            int inputs = Network.get_network_input_size(net);

            int c;
            int seedLen = seed.Length;
            float[] input = new float[inputs];
            int i;
            var inStream = Console.OpenStandardInput();
            var bytes = new byte[inStream.Length];
            inStream.Read(bytes, 0, bytes.Length);
            var readLine = Encoding.UTF8.GetString(bytes);
            foreach (var line in readLine.Split('\n'))
            {
                reset_rnn_state(net, 0);
                for (i = 0; i < seedLen; ++i)
                {
                    c = seed[i];
                    input[c] = 1;
                    Network.network_predict(net, input);
                    input[c] = 0;
                }
                Utils.Strip(line);
                int strLen = line.Length;
                for (i = 0; i < strLen; ++i)
                {
                    c = line[i];
                    input[c] = 1;
                    Network.network_predict(net, input);
                    input[c] = 0;
                }
                c = ' ';
                input[c] = 1;
                Network.network_predict(net, input);
                input[c] = 0;

                Layer l = net.Layers[0];
                Array.Copy(l.OutputGpu, l.Output, l.Outputs);
                Console.Write($"%s", line);
                for (i = 0; i < l.Outputs; ++i)
                {
                    Console.Write($",%g", l.Output[i]);
                }
                Console.Write($"\n");
            }
        }

        public static void run_char_rnn(List<string> args)
        {
            if (args.Count < 4)
            {
                Console.Error.Write($"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", args[0], args[1]);
                return;
            }
            string filename = Utils.find_int_arg(args, "-file", "Data.Data/shakespeare.txt");
            string seed = Utils.find_int_arg(args, "-seed", "\n\n");
            int len = Utils.find_int_arg(args, "-len", 1000);
            float temp = Utils.find_int_arg(args, "-temp", .7f);
            int rseed = Utils.find_int_arg(args, "-srand", (int)DateTime.Now.Ticks);
            bool clear = Utils.find_arg(args, "-clear");
            bool tokenized = Utils.find_arg(args, "-tokenized");
            string tokens = Utils.find_int_arg(args, "-tokens", "");

            string cfg = args[3];
            string weights = (args.Count > 4) ? args[4] : "";
            if (args[2] == "train") train_char_rnn(cfg, weights, filename, clear, tokenized);
            else if (args[2] == "valid") valid_char_rnn(cfg, weights, seed);
            else if (args[2] == "validtactic") valid_tactic_rnn(cfg, weights, seed);
            else if (args[2] == "vec") vec_char_rnn(cfg, weights, seed);
            else if (args[2] == "generate") test_char_rnn(cfg, weights, len, seed, temp, rseed, tokens);
            else if (args[2] == "generatetactic") test_tactic_rnn(cfg, weights, len, temp, rseed, tokens);
        }
    }
}