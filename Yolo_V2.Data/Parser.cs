using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public static class Parser
    {
        private static LayerType string_to_layer_type(string type)
        {
            switch (type)
            {
                case "[shortcut]":
                    return LayerType.Shortcut;
                case "[crop]":
                    return LayerType.Crop;
                case "[cost]":
                    return LayerType.Cost;
                case "[detection]":
                    return LayerType.Detection;
                case "[region]":
                    return LayerType.Region;
                case "[local]":
                    return LayerType.Local;
                case "[conv]":
                case "[convolutional]":
                    return LayerType.Convolutional;
                case "[activation]":
                    return LayerType.Active;
                case "[net]":
                case "[Network]":
                    return LayerType.Network;
                case "[crnn]":
                    return LayerType.Crnn;
                case "[gru]":
                    return LayerType.Gru;
                case "[rnn]":
                    return LayerType.Rnn;
                case "[conn]":
                case "[connected]":
                    return LayerType.Connected;
                case "[max]":
                case "[maxpool]":
                    return LayerType.Maxpool;
                case "[reorg]":
                    return LayerType.Reorg;
                case "[avg]":
                case "[avgpool]":
                    return LayerType.Avgpool;
                case "[dropout]":
                    return LayerType.Dropout;
                case "[lrn]":
                case "[normalization]":
                    return LayerType.Normalization;
                case "[batchnorm]":
                    return LayerType.Batchnorm;
                case "[soft]":
                case "[softmax]":
                    return LayerType.Softmax;
                case "[route]":
                    return LayerType.Route;
            }

            return LayerType.Blank;
        }

        private static Layer parse_local(KeyValuePair[] options, SizeParams parameters)
        {
            int n = OptionList.option_find_int(options, "filters", 1);
            int size = OptionList.option_find_int(options, "size", 1);
            int stride = OptionList.option_find_int(options, "stride", 1);
            int pad = OptionList.option_find_int(options, "pad", 0);
            string activationS = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activationS);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before local Layer must output image.");

            Layer layer = new Layer(batch, h, w, c, n, size, stride, pad, activation);

            return layer;
        }

        private static Layer parse_convolutional(KeyValuePair[] options, SizeParams parameters)
        {
            int n = OptionList.option_find_int(options, "filters", 1);
            int size = OptionList.option_find_int(options, "size", 1);
            int stride = OptionList.option_find_int(options, "stride", 1);
            int pad = OptionList.option_find_int_quiet(options, "pad", 0);
            int padding = OptionList.option_find_int_quiet(options, "padding", 0);
            if (pad != 0) padding = size / 2;

            string activationS = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activationS);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before convolutional Layer must output image.");
            bool batchNormalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0) != 0;
            bool binary = OptionList.option_find_int_quiet(options, "binary", 0) != 0;
            bool xnor = OptionList.option_find_int_quiet(options, "xnor", 0) != 0;

            Layer layer = Layer.make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batchNormalize, binary, xnor, parameters.Net.Adam);
            layer.Flipped = OptionList.option_find_int_quiet(options, "flipped", 0);
            layer.Dot = OptionList.option_find_float_quiet(options, "dot", 0);
            if (parameters.Net.Adam)
            {
                layer.B1 = parameters.Net.B1;
                layer.B2 = parameters.Net.B2;
                layer.Eps = parameters.Net.Eps;
            }

            return layer;
        }

        private static Layer parse_crnn(KeyValuePair[] options, SizeParams parameters)
        {
            int outputFilters = OptionList.option_find_int(options, "output_filters", 1);
            int hiddenFilters = OptionList.option_find_int(options, "hidden_filters", 1);
            string activationS = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activationS);
            bool batchNormalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0) != 0;

            Layer l = Layer.make_crnn_layer(parameters.Batch, parameters.W, parameters.H, parameters.C, hiddenFilters, outputFilters, parameters.TimeSteps, activation, batchNormalize);

            l.Shortcut = OptionList.option_find_int_quiet(options, "shortcut", 0) != 0;

            return l;
        }

        private static Layer parse_rnn(KeyValuePair[] options, SizeParams parameters)
        {
            int output = OptionList.option_find_int(options, "output", 1);
            int hidden = OptionList.option_find_int(options, "hidden", 1);
            string activationS = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activationS);
            bool batchNormalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0) != 0;
            int logistic = OptionList.option_find_int_quiet(options, "logistic", 0);

            Layer l = Layer.make_rnn_layer(parameters.Batch, parameters.Inputs, hidden, output, parameters.TimeSteps, activation, batchNormalize, logistic);

            l.Shortcut = OptionList.option_find_int_quiet(options, "shortcut", 0) != 0;

            return l;
        }

        private static Layer parse_gru(KeyValuePair[] options, SizeParams parameters)
        {
            int output = OptionList.option_find_int(options, "output", 1);
            bool batchNormalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0) != 0;

            Layer l = Layer.make_gru_layer(parameters.Batch, parameters.Inputs, output, parameters.TimeSteps, batchNormalize);

            return l;
        }

        private static Layer parse_connected(KeyValuePair[] options, SizeParams parameters)
        {
            int output = OptionList.option_find_int(options, "output", 1);
            string activationS = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activationS);
            bool batchNormalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0) != 0;

            return Layer.make_connected_layer(parameters.Batch, parameters.Inputs, output, activation, batchNormalize);
        }

        private static Layer parse_softmax(KeyValuePair[] options, SizeParams parameters)
        {
            int groups = OptionList.option_find_int_quiet(options, "groups", 1);
            Layer layer = Layer.make_softmax_layer(parameters.Batch, parameters.Inputs, groups);
            layer.Temperature = OptionList.option_find_float_quiet(options, "temperature", 1);
            string treeFile = OptionList.option_find_str(options, "tree", "");
            if (!string.IsNullOrEmpty(treeFile)) layer.SoftmaxTree = new Tree(treeFile);
            return layer;
        }

        private static Layer parse_region(KeyValuePair[] options, SizeParams parameters)
        {
            int coords = OptionList.option_find_int(options, "coords", 4);
            int classes = OptionList.option_find_int(options, "classes", 20);
            int num = OptionList.option_find_int(options, "num", 1);

            Layer l = Layer.make_region_layer(parameters.Batch, parameters.W, parameters.H, num, classes, coords);

            l.Log = OptionList.option_find_int_quiet(options, "log", 0);
            l.Sqrt = OptionList.option_find_int_quiet(options, "sqrt", 0) != 0;

            l.Softmax = OptionList.option_find_int(options, "softmax", 0) != 0;
            l.MaxBoxes = OptionList.option_find_int_quiet(options, "max", 30);
            l.Jitter = OptionList.option_find_float(options, "jitter", .2f);
            l.Rescore = OptionList.option_find_int_quiet(options, "rescore", 0) != 0;

            l.Thresh = OptionList.option_find_float(options, "thresh", .5f);
            l.Classfix = OptionList.option_find_int_quiet(options, "classfix", 0);
            l.Absolute = OptionList.option_find_int_quiet(options, "absolute", 0);
            l.Random = OptionList.option_find_int_quiet(options, "random", 0) != 0;

            l.CoordScale = OptionList.option_find_float(options, "coord_scale", 1);
            l.ObjectScale = OptionList.option_find_float(options, "object_scale", 1);
            l.NoobjectScale = OptionList.option_find_float(options, "noobject_scale", 1);
            l.ClassScale = OptionList.option_find_float(options, "class_scale", 1);
            l.BiasMatch = OptionList.option_find_int_quiet(options, "bias_match", 0) != 0;

            string treeFile = OptionList.option_find_str(options, "tree", "");
            if (!string.IsNullOrEmpty(treeFile)) l.SoftmaxTree = new Tree(treeFile);
            string mapFile = OptionList.option_find_str(options, "map", "");
            if (!string.IsNullOrEmpty(mapFile)) l.Map = Utils.read_map(mapFile);

            string a = OptionList.option_find_str(options, "anchors", null);
            if (!string.IsNullOrEmpty(a))
            {
                var lines = a.Split(',');
                for (int i = 0; i < lines.Length; ++i)
                {
                    l.BiasesComplete[l.BiasesIndex + i] = float.Parse(lines[i]);
                }
            }
            return l;
        }

        private static Layer parse_detection(KeyValuePair[] options, SizeParams parameters)
        {
            int coords = OptionList.option_find_int(options, "coords", 1);
            int classes = OptionList.option_find_int(options, "classes", 1);
            bool rescore = OptionList.option_find_int(options, "rescore", 0) != 0;
            int num = OptionList.option_find_int(options, "num", 1);
            int side = OptionList.option_find_int(options, "side", 7);
            Layer layer = Layer.make_detection_layer(parameters.Batch, parameters.Inputs, num, side, classes, coords, rescore);

            layer.Softmax = OptionList.option_find_int(options, "softmax", 0) != 0;
            layer.Sqrt = OptionList.option_find_int(options, "sqrt", 0) != 0;

            layer.MaxBoxes = OptionList.option_find_int_quiet(options, "max", 30);
            layer.CoordScale = OptionList.option_find_float(options, "coord_scale", 1);
            layer.Forced = OptionList.option_find_int(options, "forced", 0);
            layer.ObjectScale = OptionList.option_find_float(options, "object_scale", 1);
            layer.NoobjectScale = OptionList.option_find_float(options, "noobject_scale", 1);
            layer.ClassScale = OptionList.option_find_float(options, "class_scale", 1);
            layer.Jitter = OptionList.option_find_float(options, "jitter", .2f);
            layer.Random = OptionList.option_find_int_quiet(options, "random", 0) != 0;
            layer.Reorg = OptionList.option_find_int_quiet(options, "reorg", 0);
            return layer;
        }

        private static Layer parse_cost(KeyValuePair[] options, SizeParams parameters)
        {
            string typeS = OptionList.option_find_str(options, "type", "sse");
            CostType type = (CostType)Enum.Parse(typeof(CostType), typeS);
            float scale = OptionList.option_find_float_quiet(options, "scale", 1);
            Layer layer = Layer.make_cost_layer(parameters.Batch, parameters.Inputs, type, scale);
            layer.Ratio = OptionList.option_find_float_quiet(options, "ratio", 0);
            return layer;
        }

        private static Layer parse_crop(KeyValuePair[] options, SizeParams parameters)
        {
            int cropHeight = OptionList.option_find_int(options, "crop_height", 1);
            int cropWidth = OptionList.option_find_int(options, "crop_width", 1);
            bool flip = OptionList.option_find_int(options, "flip", 0) != 0;
            float angle = OptionList.option_find_float(options, "angle", 0);
            float saturation = OptionList.option_find_float(options, "saturation", 1);
            float exposure = OptionList.option_find_float(options, "exposure", 1);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before crop Layer must output image.");

            bool noadjust = OptionList.option_find_int_quiet(options, "noadjust", 0) != 0;

            Layer l = Layer.make_crop_layer(batch, h, w, c, cropHeight, cropWidth, flip, angle, saturation, exposure);
            l.Shift = OptionList.option_find_float(options, "shift", 0);
            l.Noadjust = noadjust;
            return l;
        }

        private static Layer parse_reorg(KeyValuePair[] options, SizeParams parameters)
        {
            int stride = OptionList.option_find_int(options, "stride", 1);
            bool reverse = OptionList.option_find_int_quiet(options, "reverse", 0) != 0;

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before reorg Layer must output image.");

            return Layer.make_reorg_layer(batch, w, h, c, stride, reverse);
        }

        private static Layer parse_maxpool(KeyValuePair[] options, SizeParams parameters)
        {
            int stride = OptionList.option_find_int(options, "stride", 1);
            int size = OptionList.option_find_int(options, "size", stride);
            int padding = OptionList.option_find_int_quiet(options, "padding", (size - 1) / 2);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before maxpool Layer must output image.");

            return Layer.make_maxpool_layer(batch, h, w, c, size, stride, padding);
        }

        private static Layer parse_avgpool(KeyValuePair[] options, SizeParams parameters)
        {
            int batch, w, h, c;
            w = parameters.W;
            h = parameters.H;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before avgpool Layer must output image.");

            return Layer.make_avgpool_layer(batch, w, h, c);
        }

        private static Layer parse_dropout(KeyValuePair[] options, SizeParams parameters)
        {
            float probability = OptionList.option_find_float(options, "probability", .5f);
            Layer layer = Layer.make_dropout_layer(parameters.Batch, parameters.Inputs, probability);
            layer.OutW = parameters.W;
            layer.OutH = parameters.H;
            layer.OutC = parameters.C;
            return layer;
        }

        private static Layer parse_normalization(KeyValuePair[] options, SizeParams parameters)
        {
            float alpha = OptionList.option_find_float(options, "alpha", .0001f);
            float beta = OptionList.option_find_float(options, "beta", .75f);
            float kappa = OptionList.option_find_float(options, "kappa", 1);
            int size = OptionList.option_find_int(options, "size", 5);
            return Layer.make_normalization_layer(parameters.Batch, parameters.W, parameters.H, parameters.C, size, alpha, beta, kappa);
        }

        private static Layer parse_batchnorm(KeyValuePair[] options, SizeParams parameters)
        {
            return Layer.make_batchnorm_layer(parameters.Batch, parameters.W, parameters.H, parameters.C);
        }

        private static Layer parse_shortcut(KeyValuePair[] options, SizeParams parameters, Network net)
        {
            string l = OptionList.option_find(options, "from");
            int index = int.Parse(l);
            if (index < 0) index = parameters.Index + index;

            int batch = parameters.Batch;
            Layer from = net.Layers[index];

            Layer s = Layer.make_shortcut_layer(batch, index, parameters.W, parameters.H, parameters.C, from.OutW, from.OutH, from.OutC);

            string activationS = OptionList.option_find_str(options, "activation", "linear");
            Activation activation = ActivationsHelper.Get_activation(activationS);
            s.Activation = activation;
            return s;
        }

        private static Layer parse_activation(KeyValuePair[] options, SizeParams parameters)
        {
            string activationS = OptionList.option_find_str(options, "activation", "linear");
            Activation activation = ActivationsHelper.Get_activation(activationS);

            Layer l = Layer.make_activation_layer(parameters.Batch, parameters.Inputs, activation);

            l.OutH = parameters.H;
            l.OutW = parameters.W;
            l.OutC = parameters.C;
            l.H = parameters.H;
            l.W = parameters.W;
            l.C = parameters.C;

            return l;
        }

        private static Layer parse_route(KeyValuePair[] options, SizeParams parameters, Network net)
        {
            string l = OptionList.option_find(options, "layers");
            if (string.IsNullOrEmpty(l)) Utils.Error("Route Layer must specify input layers");
            var lines = l.Split(',');
            int n = lines.Length - 1;


            int[] layers = new int[n];
            int[] sizes = new int[n];

            for (var i = 0; i < lines.Length; ++i)
            {
                int index = int.Parse(lines[i]);
                if (index < 0) index = parameters.Index + index;
                layers[i] = index;
                sizes[i] = net.Layers[index].Outputs;
            }

            int batch = parameters.Batch;

            Layer layer = Layer.make_route_layer(batch, n, layers, sizes);

            var first = net.Layers[layers[0]];
            layer.OutW = first.OutW;
            layer.OutH = first.OutH;
            layer.OutC = first.OutC;
            for (var i = 1; i < n; ++i)
            {
                int index = layers[i];
                var next = net.Layers[index];
                if (next.OutW == first.OutW && next.OutH == first.OutH)
                {
                    layer.OutC += next.OutC;
                }
                else
                {
                    layer.OutH = layer.OutW = layer.OutC = 0;
                }
            }

            return layer;
        }

        private static LearningRatePolicy get_policy(string s)
        {
            return (LearningRatePolicy)Enum.Parse(typeof(LearningRatePolicy), s);
            //if (s, "random") == 0) return RANDOM;
            //if (s, "poly") == 0) return POLY;
            //if (s, "constant") == 0) return CONSTANT;
            //if (s, "step") == 0) return STEP;
            //if (s, "exp") == 0) return EXP;
            //if (s, "sigmoid") == 0) return SIG;
            //if (s, "steps") == 0) return STEPS;
            //Console.Error.Write("Couldn't find policy %s, going with constant\n", s);
            //return CONSTANT;
        }

        private static void parse_net_options(KeyValuePair[] options, Network net)
        {
            net.Batch = OptionList.option_find_int(options, "batch", 1);
            net.LearningRate = OptionList.option_find_float(options, "learning_rate", .001f);
            net.Momentum = OptionList.option_find_float(options, "momentum", .9f);
            net.Decay = OptionList.option_find_float(options, "decay", .0001f);
            int subdivs = OptionList.option_find_int(options, "subdivisions", 1);
            net.TimeSteps = OptionList.option_find_int_quiet(options, "time_steps", 1);
            net.Batch /= subdivs;
            net.Batch *= net.TimeSteps;
            net.Subdivisions = subdivs;

            net.Adam = OptionList.option_find_int_quiet(options, "adam", 0) != 0;
            if (net.Adam )
            {
                net.B1 = OptionList.option_find_float(options, "B1", .9f);
                net.B2 = OptionList.option_find_float(options, "B2", .999f);
                net.Eps = OptionList.option_find_float(options, "eps", .000001f);
            }

            net.H = OptionList.option_find_int_quiet(options, "height", 0);
            net.W = OptionList.option_find_int_quiet(options, "width", 0);
            net.C = OptionList.option_find_int_quiet(options, "channels", 0);
            net.Inputs = OptionList.option_find_int_quiet(options, "inputs", net.H * net.W * net.C);
            net.MaxCrop = OptionList.option_find_int_quiet(options, "max_crop", net.W * 2);
            net.MinCrop = OptionList.option_find_int_quiet(options, "min_crop", net.W);

            net.Angle = OptionList.option_find_float_quiet(options, "angle", 0);
            net.Aspect = OptionList.option_find_float_quiet(options, "aspect", 1);
            net.Saturation = OptionList.option_find_float_quiet(options, "saturation", 1);
            net.Exposure = OptionList.option_find_float_quiet(options, "exposure", 1);
            net.Hue = OptionList.option_find_float_quiet(options, "hue", 0);

            if (net.Inputs == 0 && !(net.H != 0 && net.W != 0 && net.C != 0)) Utils.Error("No input parameters supplied");

            string policyS = OptionList.option_find_str(options, "policy", "constant");
            net.Policy = get_policy(policyS);
            net.BurnIn = OptionList.option_find_int_quiet(options, "burn_in", 0);
            if (net.Policy == LearningRatePolicy.Step)
            {
                net.Step = OptionList.option_find_int(options, "step", 1);
                net.Scale = OptionList.option_find_float(options, "scale", 1);
            }
            else if (net.Policy == LearningRatePolicy.Steps)
            {
                string l = OptionList.option_find(options, "steps");
                string p = OptionList.option_find(options, "scales");
                if (string.IsNullOrEmpty(l) || string.IsNullOrEmpty(p)) Utils.Error("STEPS policy must have steps and scales in cfg file");

                var lines = l.Split(',');
                int[] steps = new int[lines.Length];
                float[] scales = new float[lines.Length];
                for (var i = 0; i < lines.Length; ++i)
                {
                    steps[i] = int.Parse(lines[i]);
                    scales[i] = float.Parse(lines[i]);
                }

                net.Scales = scales;
                net.Steps = steps;
                net.NumSteps = lines.Length;
            }
            else if (net.Policy == LearningRatePolicy.Exp)
            {
                net.Gamma = OptionList.option_find_float(options, "gamma", 1);
            }
            else if (net.Policy == LearningRatePolicy.Sig)
            {
                net.Gamma = OptionList.option_find_float(options, "gamma", 1);
                net.Step = OptionList.option_find_int(options, "step", 1);
            }
            else if (net.Policy == LearningRatePolicy.Poly || net.Policy == LearningRatePolicy.Random)
            {
                net.Power = OptionList.option_find_float(options, "power", 1);
            }
            net.MaxBatches = OptionList.option_find_int(options, "max_batches", 0);
        }

        private static bool is_network(Section s)
        {
            return s.Type == "[net]" || s.Type == "[Network]";
        }

        public static Network parse_network_cfg(string filename)
        {
            Section[] sections = read_cfg(filename);
            if (sections.Length < 1) Utils.Error("Config file has no Sections");
            var n = sections[0];
            Network net = new Network(sections.Length - 1);
            SizeParams parameters = new SizeParams();

            var s = new Section(n);
            var options = s.Options;
            if (is_network(s)) Utils.Error("First Section must be [net] or [Network]");
            parse_net_options(options, net);

            parameters.H = net.H;
            parameters.W = net.W;
            parameters.C = net.C;
            parameters.Inputs = net.Inputs;
            parameters.Batch = net.Batch;
            parameters.TimeSteps = net.TimeSteps;
            parameters.Net = net;

            ulong workspaceSize = 0;
            var index = 1;
            int count = 0;
            Console.Error.Write("Layer     filters    size              input                output\n");
            while (index < sections.Length)
            {
                n = sections[index];
                index++;
                parameters.Index = count;
                Console.Error.Write($"{count:5} ");
                s = new Section(n);
                options = s.Options;
                Layer l = new Layer();
                LayerType lt = string_to_layer_type(s.Type);
                if (lt == LayerType.Convolutional)
                {
                    l = parse_convolutional(options, parameters);
                }
                else if (lt == LayerType.Local)
                {
                    l = parse_local(options, parameters);
                }
                else if (lt == LayerType.Avgpool)
                {
                    l = parse_activation(options, parameters);
                }
                else if (lt == LayerType.Rnn)
                {
                    l = parse_rnn(options, parameters);
                }
                else if (lt == LayerType.Gru)
                {
                    l = parse_gru(options, parameters);
                }
                else if (lt == LayerType.Crnn)
                {
                    l = parse_crnn(options, parameters);
                }
                else if (lt == LayerType.Connected)
                {
                    l = parse_connected(options, parameters);
                }
                else if (lt == LayerType.Crop)
                {
                    l = parse_crop(options, parameters);
                }
                else if (lt == LayerType.Cost)
                {
                    l = parse_cost(options, parameters);
                }
                else if (lt == LayerType.Region)
                {
                    l = parse_region(options, parameters);
                }
                else if (lt == LayerType.Detection)
                {
                    l = parse_detection(options, parameters);
                }
                else if (lt == LayerType.Softmax)
                {
                    l = parse_softmax(options, parameters);
                    net.Hierarchy = l.SoftmaxTree;
                }
                else if (lt == LayerType.Normalization)
                {
                    l = parse_normalization(options, parameters);
                }
                else if (lt == LayerType.Batchnorm)
                {
                    l = parse_batchnorm(options, parameters);
                }
                else if (lt == LayerType.Maxpool)
                {
                    l = parse_maxpool(options, parameters);
                }
                else if (lt == LayerType.Reorg)
                {
                    l = parse_reorg(options, parameters);
                }
                else if (lt == LayerType.Avgpool)
                {
                    l = parse_avgpool(options, parameters);
                }
                else if (lt == LayerType.Route)
                {
                    l = parse_route(options, parameters, net);
                }
                else if (lt == LayerType.Shortcut)
                {
                    l = parse_shortcut(options, parameters, net);
                }
                else if (lt == LayerType.Dropout)
                {
                    l = parse_dropout(options, parameters);
                    l.Output = net.Layers[count - 1].Output;
                    l.Delta = net.Layers[count - 1].Delta;
                    l.OutputGpu = net.Layers[count - 1].OutputGpu;
                    l.DeltaGpu = net.Layers[count - 1].DeltaGpu;
                }
                else
                {
                    Console.Error.Write($"LayerType not recognized: {s.Type}\n");
                }
                l.Dontload = OptionList.option_find_int_quiet(options, "dontload", 0) != 0;
                l.Dontloadscales = OptionList.option_find_int_quiet(options, "dontloadscales", 0) != 0;
                OptionList.option_unused(options);
                net.Layers[count] = l;
                if (l.WorkspaceSize > workspaceSize) workspaceSize = l.WorkspaceSize;
                ++count;
                if (index + 1 < sections.Length)
                {
                    parameters.H = l.OutH;
                    parameters.W = l.OutW;
                    parameters.C = l.OutC;
                    parameters.Inputs = l.Outputs;
                }
            }
            net.Outputs = Network.get_network_output_size(net);
            net.Output = Network.get_network_output(net);
            if (workspaceSize != 0)
            {
                if (CudaUtils.UseGpu)
                {
                    net.Workspace = new float[(workspaceSize - 1) / sizeof(float) + 1];
                }
                else
                {
                    net.Workspace = new float[1];
                }
            }
            return net;
        }

        private static Section[] read_cfg(string filename)
        {
            if (!File.Exists(filename)) Utils.file_error(filename);
            int nu = 0;
            List<Section> sections = new List<Section>();
            var lines = File.ReadAllLines(filename);
            Section current = null;
            foreach (var line in lines)
            {
                ++nu;
                Utils.Strip(line);
                if (string.IsNullOrEmpty(line))
                {
                    continue;
                }
                switch (line[0])
                {
                    case '[':
                        current = new Section(line);
                        sections.Add(current);
                        break;
                    case '\0':
                    case '#':
                    case ';':
                        break;
                    default:
                        if (!OptionList.read_option(line, current?.Options.ToList()))
                        {
                            Console.Error.Write("Config file Utils.Error line %d, could parse: %s\n", nu, line);
                        }
                        break;
                }
            }
            return sections.ToArray();
        }

        private static byte[] FloatArrayToByteArray(float[] floats, int start = 0)
        {
            var bytes = new byte[(floats.Length - start) * 4];
            //todo may be this should be start * 4
            Buffer.BlockCopy(floats, start, bytes, 0, bytes.Length);
            return bytes;
        }

        private static void save_convolutional_weights(Layer l, FileStream fstream)
        {
            if (CudaUtils.UseGpu)
            {
                l.pull_convolutional_layer();
            }

            int num = l.N * l.C * l.Size * l.Size;
            var biases = FloatArrayToByteArray(l.BiasesComplete, l.BiasesIndex);
            fstream.Write(biases, 0, biases.Length);

            if (l.BatchNormalize)
            {
                var scales = FloatArrayToByteArray(l.Scales);
                var mean = FloatArrayToByteArray(l.RollingMean);
                var variance = FloatArrayToByteArray(l.RollingVariance);
                fstream.Write(scales, 0, scales.Length);
                fstream.Write(mean, 0, mean.Length);
                fstream.Write(variance, 0, variance.Length);
            }

            var weights = FloatArrayToByteArray(l.WeightsComplete, l.WeightsIndex);
            fstream.Write(weights, 0, weights.Length);

            if (l.Adam )
            {
                var m = FloatArrayToByteArray(l.M);
                var v = FloatArrayToByteArray(l.V);
                fstream.Write(m, 0, m.Length);
                fstream.Write(v, 0, v.Length);
            }
        }

        private static void save_batchnorm_weights(Layer l, FileStream fread)
        {
            if (CudaUtils.UseGpu)
            {
                l.pull_batchnorm_layer();
            }

            var scales = FloatArrayToByteArray(l.Scales);
            fread.Write(scales, 0, scales.Length);

            var mean = FloatArrayToByteArray(l.RollingMean);
            fread.Write(mean, 0, mean.Length);

            var variance = FloatArrayToByteArray(l.RollingVariance);
            fread.Write(variance, 0, variance.Length);
        }

        private static void save_connected_weights(Layer l, FileStream fread)
        {
            if (CudaUtils.UseGpu)
            {
                l.pull_connected_layer();
            }

            var biases = FloatArrayToByteArray(l.BiasesComplete, l.BiasesIndex);
            fread.Write(biases, 0, biases.Length);

            var weights = FloatArrayToByteArray(l.WeightsComplete, l.WeightsIndex);
            fread.Write(weights, 0, weights.Length);
            if (l.BatchNormalize )
            {
                var scales = FloatArrayToByteArray(l.Scales);
                fread.Write(scales, 0, scales.Length);

                var mean = FloatArrayToByteArray(l.RollingMean);
                fread.Write(mean, 0, mean.Length);

                var variance = FloatArrayToByteArray(l.RollingVariance);
                fread.Write(variance, 0, variance.Length);
            }
        }

        public static void save_weights_upto(Network net, string filename, int cutoff)
        {
            Console.Error.Write($"Saving weights to {filename}\n");
            if (!File.Exists(filename)) Utils.file_error(filename);

            using (var fstream = File.OpenWrite(filename))
            {
                var major = BitConverter.GetBytes(0);
                var minor = BitConverter.GetBytes(1);
                var revision = BitConverter.GetBytes(0);
                var seen = BitConverter.GetBytes(net.Seen);
                fstream.Write(major, 0, major.Length);
                fstream.Write(minor, 0, minor.Length);
                fstream.Write(revision, 0, revision.Length);
                fstream.Write(seen, 0, seen.Length);

                int i;
                for (i = 0; i < net.N && i < cutoff; ++i)
                {
                    Layer l = net.Layers[i];
                    if (l.LayerType == LayerType.Convolutional)
                    {
                        save_convolutional_weights(l, fstream);
                    }

                    if (l.LayerType == LayerType.Connected)
                    {
                        save_connected_weights(l, fstream);
                    }

                    if (l.LayerType == LayerType.Batchnorm)
                    {
                        save_batchnorm_weights(l, fstream);
                    }

                    if (l.LayerType == LayerType.Rnn)
                    {
                        save_connected_weights((l.InputLayer), fstream);
                        save_connected_weights((l.SelfLayer), fstream);
                        save_connected_weights((l.OutputLayer), fstream);
                    }

                    if (l.LayerType == LayerType.Gru)
                    {
                        save_connected_weights((l.InputZLayer), fstream);
                        save_connected_weights((l.InputRLayer), fstream);
                        save_connected_weights((l.InputHLayer), fstream);
                        save_connected_weights((l.StateZLayer), fstream);
                        save_connected_weights((l.StateRLayer), fstream);
                        save_connected_weights((l.StateHLayer), fstream);
                    }

                    if (l.LayerType == LayerType.Crnn)
                    {
                        save_convolutional_weights((l.InputLayer), fstream);
                        save_convolutional_weights((l.SelfLayer), fstream);
                        save_convolutional_weights((l.OutputLayer), fstream);
                    }

                    if (l.LayerType == LayerType.Local)
                    {
                        if (CudaUtils.UseGpu)
                        {
                            l.pull_local_layer();
                        }

                        int locations = l.OutW * l.OutH;
                        int size = l.Size * l.Size * l.C * l.N * locations;

                        var biases = FloatArrayToByteArray(l.BiasesComplete, l.BiasesIndex);

                        var weights = FloatArrayToByteArray(l.WeightsComplete, l.WeightsIndex);

                        fstream.Write(biases, 0, biases.Length);
                        fstream.Write(weights, 0, weights.Length);
                    }
                }
            }
        }

        public static void save_weights(Network net, string filename)
        {
            save_weights_upto(net, filename, net.N);
        }

        private static void transpose_matrix(float[] a, int rows, int cols, int aStart = 0)
        {
            float[] transpose = new float[rows * cols];
            int x, y;
            for (x = 0; x < rows; ++x)
            {
                for (y = 0; y < cols; ++y)
                {
                    transpose[y * rows + x] = a[aStart + x * cols + y];
                }
            }
            Array.Copy(transpose, 0, a, aStart, rows * cols);
        }

        private static void load_connected_weights(Layer l, FileStream fread, int transpose)
        {
            l.BiasesComplete = ReadFloat(fread, l.Outputs);
            l.WeightsComplete = ReadFloat(fread, l.Outputs * l.Inputs);
            l.BiasesIndex = l.WeightsIndex = 0;

            if (transpose != 0)
            {
                transpose_matrix(l.WeightsComplete, l.Inputs, l.Outputs, l.WeightsIndex);
            }
            if (l.BatchNormalize  && !l.Dontloadscales)
            {
                l.Scales = ReadFloat(fread, l.Outputs);
                l.RollingMean = ReadFloat(fread, l.Outputs);
                l.RollingVariance = ReadFloat(fread, l.Outputs);
            }
            if (CudaUtils.UseGpu)
            {
                l.push_connected_layer();
            }
        }

        private static float[] ReadFloat(FileStream fstream, int size)
        {
            var bytes = new byte[size * sizeof(float)];
            fstream.Read(bytes, 0, bytes.Length);
            var floats = new float[size];
            Buffer.BlockCopy(bytes, 0, floats, 0, floats.Length);
            return floats;
        }

        private static int[] ReadInt(FileStream fstream, int size)
        {
            var bytes = new byte[size * sizeof(int)];
            fstream.Read(bytes, 0, bytes.Length);
            var ints = new int[size];
            Buffer.BlockCopy(bytes, 0, ints, 0, ints.Length);
            return ints;
        }

        private static void load_batchnorm_weights(Layer l, FileStream fread)
        {
            l.Scales = ReadFloat(fread, l.C);
            l.RollingMean = ReadFloat(fread, l.C);
            l.RollingVariance = ReadFloat(fread, l.C);
            if (CudaUtils.UseGpu)
            {
                l.push_batchnorm_layer();
            }
        }

        private static void load_convolutional_weights(Layer l, FileStream fread)
        {
            int num = l.N * l.C * l.Size * l.Size;
            l.BiasesComplete = ReadFloat(fread, l.N);
            l.BiasesIndex = 0;
            if (l.BatchNormalize  && (!l.Dontloadscales))
            {
                l.Scales = ReadFloat(fread, l.N);
                l.RollingMean = ReadFloat(fread, l.N);
                l.RollingVariance = ReadFloat(fread, l.N);
            }

            l.WeightsComplete = ReadFloat(fread, num);
            l.WeightsIndex = 0;
            if (l.Adam )
            {
                l.M = ReadFloat(fread, num);
                l.V = ReadFloat(fread, num);
            }
            if (l.Flipped != 0)
            {
                transpose_matrix(l.WeightsComplete, l.C * l.Size * l.Size, l.N, l.WeightsIndex);
            }
            if (CudaUtils.UseGpu)
            {
                l.push_convolutional_layer();
            }
        }

        public static void load_weights_upto(Network net, string filename, int cutoff)
        {
            Console.Error.Write($"Loading weights from {filename}...");
            if (!File.Exists(filename))
            {
                Utils.file_error(filename);
            }

            using (var fread = File.OpenRead(filename))
            {
                int major = ReadInt(fread, 1)[0];
                int minor = ReadInt(fread, 1)[0];
                int revision = ReadInt(fread, 1)[0];
                net.Seen = ReadInt(fread, 1)[0];

                int transpose = (major > 1000) || (minor > 1000) ? 1 : 0;

                int i;
                for (i = 0; i < net.N && i < cutoff; ++i)
                {
                    Layer l = net.Layers[i];
                    if (l.Dontload) continue;
                    if (l.LayerType == LayerType.Convolutional)
                    {
                        load_convolutional_weights(l, fread);
                    }

                    if (l.LayerType == LayerType.Connected)
                    {
                        load_connected_weights(l, fread, transpose);
                    }

                    if (l.LayerType == LayerType.Batchnorm)
                    {
                        load_batchnorm_weights(l, fread);
                    }

                    if (l.LayerType == LayerType.Crnn)
                    {
                        load_convolutional_weights((l.InputLayer), fread);
                        load_convolutional_weights((l.SelfLayer), fread);
                        load_convolutional_weights((l.OutputLayer), fread);
                    }

                    if (l.LayerType == LayerType.Rnn)
                    {
                        load_connected_weights((l.InputLayer), fread, transpose);
                        load_connected_weights((l.SelfLayer), fread, transpose);
                        load_connected_weights((l.OutputLayer), fread, transpose);
                    }

                    if (l.LayerType == LayerType.Gru)
                    {
                        load_connected_weights((l.InputZLayer), fread, transpose);
                        load_connected_weights((l.InputRLayer), fread, transpose);
                        load_connected_weights((l.InputHLayer), fread, transpose);
                        load_connected_weights((l.StateZLayer), fread, transpose);
                        load_connected_weights((l.StateRLayer), fread, transpose);
                        load_connected_weights((l.StateHLayer), fread, transpose);
                    }

                    if (l.LayerType == LayerType.Local)
                    {
                        int locations = l.OutW * l.OutH;
                        int size = l.Size * l.Size * l.C * l.N * locations;
                        l.BiasesComplete = ReadFloat(fread, l.Outputs);
                        l.WeightsComplete = ReadFloat(fread, size);
                        l.BiasesIndex = 0;
                        l.WeightsIndex = 0;
                        if (CudaUtils.UseGpu)
                        {
                            l.push_local_layer();
                        }
                    }
                }
            }

            Console.Error.Write("Done!\n");
        }

        public static void load_weights(Network net, string filename)
        {
            load_weights_upto(net, filename, net.N);
        }
    }
}
