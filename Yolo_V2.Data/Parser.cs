using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public static class Parser
    {
        public static LayerType string_to_layer_type(string type)
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

        public static void parse_data(string data, ref float[] a)
        {
            if (string.IsNullOrEmpty(data)) return;
            var numbs = data.Split(',');
            a = numbs.Select(float.Parse).ToArray();
        }

        public static Layer parse_local(KeyValuePair[] options, SizeParams parameters)
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

            Layer Layer = new Layer(batch, h, w, c, n, size, stride, pad, activation);

            return Layer;
        }

        public static Layer parse_convolutional(KeyValuePair[] options, SizeParams parameters)
        {
            int n = OptionList.option_find_int(options, "filters", 1);
            int size = OptionList.option_find_int(options, "size", 1);
            int stride = OptionList.option_find_int(options, "stride", 1);
            int pad = OptionList.option_find_int_quiet(options, "pad", 0);
            int padding = OptionList.option_find_int_quiet(options, "padding", 0);
            if (pad != 0) padding = size / 2;

            string activation_s = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activation_s);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before convolutional Layer must output image.");
            int batch_normalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0);
            int binary = OptionList.option_find_int_quiet(options, "binary", 0);
            int xnor = OptionList.option_find_int_quiet(options, "xnor", 0);

            Layer Layer = make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, binary, xnor, parameters.Net.Adam);
            Layer.Flipped = OptionList.option_find_int_quiet(options, "flipped", 0);
            Layer.Dot = OptionList.option_find_float_quiet(options, "dot", 0);
            if (parameters.Net.Adam != 0)
            {
                Layer.B1 = parameters.Net.B1;
                Layer.B2 = parameters.Net.B2;
                Layer.Eps = parameters.Net.Eps;
            }

            return Layer;
        }

        public static Layer parse_crnn(KeyValuePair[] options, SizeParams parameters)
        {
            int output_filters = OptionList.option_find_int(options, "output_filters", 1);
            int hidden_filters = OptionList.option_find_int(options, "hidden_filters", 1);
            string activation_s = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activation_s);
            int batch_normalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0);

            Layer l = make_crnn_layer(parameters.Batch, parameters.W, parameters.H, parameters.C, hidden_filters, output_filters, parameters.TimeSteps, activation, batch_normalize);

            l.Shortcut = OptionList.option_find_int_quiet(options, "shortcut", 0);

            return l;
        }

        public static Layer parse_rnn(KeyValuePair[] options, SizeParams parameters)
        {
            int output = OptionList.option_find_int(options, "output", 1);
            int hidden = OptionList.option_find_int(options, "hidden", 1);
            string activation_s = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activation_s);
            int batch_normalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0);
            int logistic = OptionList.option_find_int_quiet(options, "logistic", 0);

            Layer l = make_rnn_layer(parameters.Batch, parameters.Inputs, hidden, output, parameters.TimeSteps, activation, batch_normalize, logistic);

            l.Shortcut = OptionList.option_find_int_quiet(options, "shortcut", 0);

            return l;
        }

        public static Layer parse_gru(KeyValuePair[] options, SizeParams parameters)
        {
            int output = OptionList.option_find_int(options, "output", 1);
            int batch_normalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0);

            Layer l = make_gru_layer(parameters.Batch, parameters.Inputs, output, parameters.TimeSteps, batch_normalize);

            return l;
        }

        public static Layer parse_connected(KeyValuePair[] options, SizeParams parameters)
        {
            int output = OptionList.option_find_int(options, "output", 1);
            string activation_s = OptionList.option_find_str(options, "activation", "logistic");
            Activation activation = ActivationsHelper.Get_activation(activation_s);
            int batch_normalize = OptionList.option_find_int_quiet(options, "batch_normalize", 0);

            Layer Layer = make_connected_layer(parameters.Batch, parameters.Inputs, output, activation, batch_normalize);

            return Layer;
        }

        public static Layer parse_softmax(KeyValuePair[] options, SizeParams parameters)
        {
            int groups = OptionList.option_find_int_quiet(options, "groups", 1);
            Layer Layer = make_softmax_layer(parameters.Batch, parameters.Inputs, groups);
            Layer.temperature = OptionList.option_find_float_quiet(options, "temperature", 1);
            string tree_file = OptionList.option_find_str(options, "tree", "");
            if (!string.IsNullOrEmpty(tree_file)) Layer.softmax_tree = new Tree(tree_file);
            return Layer;
        }

        public static Layer parse_region(KeyValuePair[] options, SizeParams parameters)
        {
            int coords = OptionList.option_find_int(options, "coords", 4);
            int classes = OptionList.option_find_int(options, "classes", 20);
            int num = OptionList.option_find_int(options, "num", 1);

            Layer l = make_region_layer(parameters.Batch, parameters.W, parameters.H, num, classes, coords);

            l.Log = OptionList.option_find_int_quiet(options, "log", 0);
            l.Sqrt = OptionList.option_find_int_quiet(options, "sqrt", 0);

            l.Softmax = OptionList.option_find_int(options, "softmax", 0);
            l.MaxBoxes = OptionList.option_find_int_quiet(options, "max", 30);
            l.Jitter = OptionList.option_find_float(options, "jitter", .2);
            l.Rescore = OptionList.option_find_int_quiet(options, "rescore", 0);

            l.Thresh = OptionList.option_find_float(options, "thresh", .5);
            l.Classfix = OptionList.option_find_int_quiet(options, "classfix", 0);
            l.Absolute = OptionList.option_find_int_quiet(options, "absolute", 0);
            l.Random = OptionList.option_find_int_quiet(options, "random", 0);

            l.CoordScale = OptionList.option_find_float(options, "coord_scale", 1);
            l.ObjectScale = OptionList.option_find_float(options, "object_scale", 1);
            l.NoobjectScale = OptionList.option_find_float(options, "noobject_scale", 1);
            l.ClassScale = OptionList.option_find_float(options, "class_scale", 1);
            l.BiasMatch = OptionList.option_find_int_quiet(options, "bias_match", 0);

            string tree_file = OptionList.option_find_str(options, "tree", "");
            if (!string.IsNullOrEmpty(tree_file)) l.SoftmaxTree = new Tree(tree_file);
            string map_file = OptionList.option_find_str(options, "map", "");
            if (!string.IsNullOrEmpty(map_file)) l.Map = Utils.read_map(map_file);

            string a = OptionList.option_find_str(options, "anchors", null);
            if (!string.IsNullOrEmpty(a))
            {
                var lines = a.Split(',');
                for (int i = 0; i < lines.Length; ++i)
                {
                    l.Biases[i] = float.Parse(lines[i]);
                }
            }
            return l;
        }

        public static Layer parse_detection(KeyValuePair[] options, SizeParams parameters)
        {
            int coords = OptionList.option_find_int(options, "coords", 1);
            int classes = OptionList.option_find_int(options, "classes", 1);
            int rescore = OptionList.option_find_int(options, "rescore", 0);
            int num = OptionList.option_find_int(options, "num", 1);
            int side = OptionList.option_find_int(options, "side", 7);
            Layer Layer = make_detection_layer(parameters.Batch, parameters.Inputs, num, side, classes, coords, rescore);

            Layer.Softmax = OptionList.option_find_int(options, "softmax", 0);
            Layer.Sqrt = OptionList.option_find_int(options, "sqrt", 0);

            Layer.MaxBoxes = OptionList.option_find_int_quiet(options, "max", 30);
            Layer.CoordScale = OptionList.option_find_float(options, "coord_scale", 1);
            Layer.Forced = OptionList.option_find_int(options, "forced", 0);
            Layer.ObjectScale = OptionList.option_find_float(options, "object_scale", 1);
            Layer.NoobjectScale = OptionList.option_find_float(options, "noobject_scale", 1);
            Layer.ClassScale = OptionList.option_find_float(options, "class_scale", 1);
            Layer.Jitter = OptionList.option_find_float(options, "jitter", .2);
            Layer.Random = OptionList.option_find_int_quiet(options, "random", 0);
            Layer.Reorg = OptionList.option_find_int_quiet(options, "reorg", 0);
            return Layer;
        }

        public static Layer parse_cost(KeyValuePair[] options, SizeParams parameters)
        {
            string type_s = OptionList.option_find_str(options, "type", "sse");
            CostType type = get_cost_type(type_s);
            float scale = OptionList.option_find_float_quiet(options, "scale", 1);
            Layer Layer = make_cost_layer(parameters.Batch, parameters.Inputs, type, scale);
            Layer.Ratio = OptionList.option_find_float_quiet(options, "ratio", 0);
            return Layer;
        }

        public static Layer parse_crop(KeyValuePair[] options, SizeParams parameters)
        {
            int crop_height = OptionList.option_find_int(options, "crop_height", 1);
            int crop_width = OptionList.option_find_int(options, "crop_width", 1);
            int flip = OptionList.option_find_int(options, "flip", 0);
            float angle = OptionList.option_find_float(options, "angle", 0);
            float saturation = OptionList.option_find_float(options, "saturation", 1);
            float exposure = OptionList.option_find_float(options, "exposure", 1);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before crop Layer must output image.");

            int noadjust = OptionList.option_find_int_quiet(options, "noadjust", 0);

            Layer l = make_crop_layer(batch, h, w, c, crop_height, crop_width, flip, angle, saturation, exposure);
            l.Shift = OptionList.option_find_float(options, "shift", 0);
            l.Noadjust = noadjust;
            return l;
        }

        public static Layer parse_reorg(KeyValuePair[] options, SizeParams parameters)
        {
            int stride = OptionList.option_find_int(options, "stride", 1);
            int reverse = OptionList.option_find_int_quiet(options, "reverse", 0);

            int batch, h, w, c;
            h = parameters.H;
            w = parameters.W;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before reorg Layer must output image.");

            Layer Layer = make_reorg_layer(batch, w, h, c, stride, reverse);
            return Layer;
        }

        public static Layer parse_maxpool(KeyValuePair[] options, SizeParams parameters)
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

            Layer Layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
            return Layer;
        }

        public static Layer parse_avgpool(KeyValuePair[] options, SizeParams parameters)
        {
            int batch, w, h, c;
            w = parameters.W;
            h = parameters.H;
            c = parameters.C;
            batch = parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before avgpool Layer must output image.");

            Layer Layer = make_avgpool_layer(batch, w, h, c);
            return Layer;
        }

        public static Layer parse_dropout(KeyValuePair[] options, SizeParams parameters)
        {
            float probability = OptionList.option_find_float(options, "probability", .5);
            Layer Layer = make_dropout_layer(parameters.Batch, parameters.Inputs, probability);
            Layer.OutW = parameters.W;
            Layer.OutH = parameters.H;
            Layer.OutC = parameters.C;
            return Layer;
        }

        public static Layer parse_normalization(KeyValuePair[] options, SizeParams parameters)
        {
            float alpha = OptionList.option_find_float(options, "alpha", .0001f);
            float beta = OptionList.option_find_float(options, "beta", .75f);
            float kappa = OptionList.option_find_float(options, "kappa", 1);
            int size = OptionList.option_find_int(options, "size", 5);
            Layer l = make_normalization_layer(parameters.Batch, parameters.W, parameters.H, parameters.C, size, alpha, beta, kappa);
            return l;
        }

        public static Layer parse_batchnorm(KeyValuePair[] options, SizeParams parameters)
        {
            Layer l = make_batchnorm_layer(parameters.Batch, parameters.W, parameters.H, parameters.C);
            return l;
        }

        public static Layer parse_shortcut(KeyValuePair[] options, SizeParams parameters, Network net)
        {
            string l = OptionList.option_find(options, "from");
            int index = int.Parse(l);
            if (index < 0) index = parameters.Index + index;

            int batch = parameters.Batch;
            Layer from = net.Layers[index];

            Layer s = make_shortcut_layer(batch, index, parameters.W, parameters.H, parameters.C, from.OutW, from.OutH, from.OutC);

            string activation_s = OptionList.option_find_str(options, "activation", "linear");
            Activation activation = ActivationsHelper.Get_activation(activation_s);
            s.Activation = activation;
            return s;
        }

        public static Layer parse_activation(KeyValuePair[] options, SizeParams parameters)
        {
            string activation_s = OptionList.option_find_str(options, "activation", "linear");
            Activation activation = ActivationsHelper.Get_activation(activation_s);

            Layer l = make_activation_layer(parameters.Batch, parameters.Inputs, activation);

            l.OutH = parameters.H;
            l.OutW = parameters.W;
            l.OutC = parameters.C;
            l.H = parameters.H;
            l.W = parameters.W;
            l.C = parameters.C;

            return l;
        }

        public static Layer parse_route(KeyValuePair[] options, SizeParams parameters, Network net)
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

            Layer Layer = make_route_layer(batch, n, layers, sizes);

            Layer first = net.Layers[layers[0]];
            Layer.OutW = first.OutW;
            Layer.OutH = first.OutH;
            Layer.OutC = first.OutC;
            for (var i = 1; i < n; ++i)
            {
                int index = layers[i];
                Layer next = net.Layers[index];
                if (next.OutW == first.OutW && next.OutH == first.OutH)
                {
                    Layer.OutC += next.OutC;
                }
                else
                {
                    Layer.OutH = Layer.OutW = Layer.OutC = 0;
                }
            }

            return Layer;
        }

        public static LearningRatePolicy get_policy(string s)
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

        public static void parse_net_options(KeyValuePair[] options, Network net)
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

            net.Adam = OptionList.option_find_int_quiet(options, "adam", 0);
            if (net.Adam != 0)
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

            string policy_s = OptionList.option_find_str(options, "policy", "constant");
            net.Policy = get_policy(policy_s);
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

        public static int is_network(section s)
        {
            return (s.type, "[net]") == 0
                    || s.type, "[Network]") == 0);
        }

        public static Network parse_network_cfg(string filename)
        {
            KeyValuePair[] sections = read_cfg(filename);
            if (sections.Length < 1) Utils.Error("Config file has no sections");
            var n = sections[0];
            Network net = make_network(sections.Length - 1);
            SizeParams parameters = new SizeParams();

            var s = new section(n.Val);
            KeyValuePair[] options = s.Options;
            if (!is_network(s)) Utils.Error("First section must be [net] or [Network]");
            parse_net_options(options, net);

            parameters.H = net.H;
            parameters.W = net.W;
            parameters.C = net.C;
            parameters.Inputs = net.Inputs;
            parameters.Batch = net.Batch;
            parameters.TimeSteps = net.TimeSteps;
            parameters.Net = net;

            ulong workspace_size = 0;
            var index = 1;
            int count = 0;
            Console.Error.Write("Layer     filters    size              input                output\n");
            while (index < sections.Length)
            {
                n = sections[index];
                index++;
                parameters.Index = count;
                Console.Error.Write($"{count:5} ");
                s = new section(n.Val);
                options = s.options;
                Layer l = new Layer();
                LayerType lt = string_to_layer_type(s.type);
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
                    Console.Error.Write($"LayerType not recognized: {s.type}\n");
                }
                l.Dontload = OptionList.option_find_int_quiet(options, "dontload", 0);
                l.Dontloadscales = OptionList.option_find_int_quiet(options, "dontloadscales", 0);
                OptionList.option_unused(options);
                net.Layers[count] = l;
                if (l.WorkspaceSize > workspace_size) workspace_size = l.WorkspaceSize;
                ++count;
                if (index + 1 < sections.Length)
                {
                    parameters.H = l.OutH;
                    parameters.W = l.OutW;
                    parameters.C = l.OutC;
                    parameters.Inputs = l.Outputs;
                }
            }
            net.Outputs = get_network_output_size(net);
            net.Output = get_network_output(net);
            if (workspace_size != 0)
            {
                if (CudaUtils.UseGpu)
                {
                    net.Workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
                }
                else
                {
                    net.Workspace = new float[1];
                }
            }
            return net;
        }

        public static KeyValuePair[] read_cfg(string filename)
        {
            FILE* file = fopen(filename, "r");
            if (file == 0) file_error(filename);
            string line;
            int nu = 0;
            KeyValuePair[] sections = make_list();
            section* current = 0;
            while ((line = fgetl(file)) != 0)
            {
                ++nu;
                strip(line);
                switch (line[0])
                {
                    case '[':
                        current = malloc(sizeof(section));
                        list_insert(sections, current);
                        current.options = make_list();
                        current.type = line;
                        break;
                    case '\0':
                    case '#':
                    case ';':
                        free(line);
                        break;
                    default:
                        if (!read_option(line, current.options))
                        {
                            Console.Error.Write("Config file Utils.Error line %d, could parse: %s\n", nu, line);
                            free(line);
                        }
                        break;
                }
            }
            fclose(file);
            return sections;
        }

        public static void save_convolutional_weights_binary(Layer l, string filename)
        {
            if (CudaUtils.UseGpu)
            {
                pull_convolutional_layer(l);
            }
            binarize_weights(l.Weights, l.N, l.C * l.Size * l.Size, l.BinaryWeights);
            int size = l.C * l.Size * l.Size;
            int i, j;
            using (var fstream = File.OpenWrite(filename))
            {
                var biases = FloatArrayToByteArray(l.Biases);
                fstream.Write(biases, 0, biases.Length);
                if (l.BatchNormalize != 0)
                {
                    var scales = FloatArrayToByteArray(l.Scales);
                    var mean = FloatArrayToByteArray(l.RollingMean);
                    var variance = FloatArrayToByteArray(l.RollingVariance);
                    fstream.Write(scales, 0, scales.Length);
                    fstream.Write(mean, 0, mean.Length);
                    fstream.Write(variance, 0, variance.Length);
                }
                for (i = 0; i < l.N; ++i)
                {
                    float mean = l.BinaryWeights[i * size];
                    if (mean < 0) mean = -mean;
                    var writeMean = BitConverter.GetBytes(mean);
                    fstream.Write(writeMean, 0, writeMean.Length);
                    for (j = 0; j < size / 8; ++j)
                    {
                        int index = i * size + j * 8;
                        byte[] c = new byte[1];
                        c[0] = 0;
                        for (byte k = 0; k < 8; ++k)
                        {
                            if (j * 8 + k >= size) break;
                            if (l.BinaryWeights[index + k] > 0) c[0] = (byte)(c[0] | 0x1 << k);
                        }
                        fstream.Write(c, 0, 1);
                    }
                }
            }
        }

        private static byte[] FloatArrayToByteArray(float[] floats)
        {
            var bytes = new byte[floats.Length * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        public static void save_convolutional_weights(Layer l, string filename)
        {
            if (CudaUtils.UseGpu)
            {
                pull_convolutional_layer(l);
            }

            int num = l.N * l.C * l.Size * l.Size;
            using (var fstream = File.OpenWrite(filename))
            {
                var biases = FloatArrayToByteArray(l.Biases);
                fstream.Write(biases, 0, biases.Length);

                if (l.BatchNormalize != 0)
                {
                    var scales = FloatArrayToByteArray(l.Scales);
                    var mean = FloatArrayToByteArray(l.RollingMean);
                    var variance = FloatArrayToByteArray(l.RollingVariance);
                    fstream.Write(scales, 0, scales.Length);
                    fstream.Write(mean, 0, mean.Length);
                    fstream.Write(variance, 0, variance.Length);
                }

                var weights = FloatArrayToByteArray(l.Weights);
                fstream.Write(weights, 0, weights.Length);

                if (l.Adam != 0)
                {
                    var m = FloatArrayToByteArray(l.M);
                    var v = FloatArrayToByteArray(l.V);
                    fstream.Write(m, 0, m.Length);
                    fstream.Write(v, 0, v.Length);
                }
            }
        }

        public static void save_batchnorm_weights(Layer l, FileStream fread)
        {
            if (CudaUtils.UseGpu)
            {
                pull_batchnorm_layer(l);
            }
            fwrite(l.Scales, sizeof(float), l.C, fp);
            fwrite(l.RollingMean, sizeof(float), l.C, fp);
            fwrite(l.RollingVariance, sizeof(float), l.C, fp);
        }

        public static void save_connected_weights(Layer l, FileStream fread)
        {
            if (CudaUtils.UseGpu)
            {
                pull_connected_layer(l);
            }
            fwrite(l.Biases, sizeof(float), l.outputs, fp);
            fwrite(l.Weights, sizeof(float), l.outputs * l.Inputs, fp);
            if (l.BatchNormalize)
            {
                fwrite(l.Scales, sizeof(float), l.outputs, fp);
                fwrite(l.RollingMean, sizeof(float), l.outputs, fp);
                fwrite(l.RollingVariance, sizeof(float), l.outputs, fp);
            }
        }

        public static void save_weights_upto(Network net, string filename, int cutoff)
        {
            if (net.CudaUtils.UseGpu)
            {
                cuda_set_device(net.gpu_index);
            }
            Console.Error.Write("Saving weights to %s\n", filename);
            FileStream fread = fopen(filename, "wb");
            if (!fp) file_error(filename);

            int major = 0;
            int minor = 1;
            int revision = 0;
            fwrite(&major, sizeof(int), 1, fp);
            fwrite(&minor, sizeof(int), 1, fp);
            fwrite(&revision, sizeof(int), 1, fp);
            fwrite(net.seen, sizeof(int), 1, fp);

            int i;
            for (i = 0; i < net.N && i < cutoff; ++i)
            {
                Layer l = net.layers[i];
                if (l.type == CONVOLUTIONAL)
                {
                    save_convolutional_weights(l, fp);
                }
                if (l.type == CONNECTED)
                {
                    save_connected_weights(l, fp);
                }
                if (l.type == BATCHNORM)
                {
                    save_batchnorm_weights(l, fp);
                }
                if (l.type == RNN)
                {
                    save_connected_weights(*(l.InputLayer), fp);
                    save_connected_weights(*(l.SelfLayer), fp);
                    save_connected_weights(*(l.OutputLayer), fp);
                }
                if (l.type == GRU)
                {
                    save_connected_weights(*(l.InputZLayer), fp);
                    save_connected_weights(*(l.InputRLayer), fp);
                    save_connected_weights(*(l.InputHLayer), fp);
                    save_connected_weights(*(l.StateZLayer), fp);
                    save_connected_weights(*(l.StateRLayer), fp);
                    save_connected_weights(*(l.StateHLayer), fp);
                }
                if (l.type == CRNN)
                {
                    save_convolutional_weights(*(l.InputLayer), fp);
                    save_convolutional_weights(*(l.SelfLayer), fp);
                    save_convolutional_weights(*(l.OutputLayer), fp);
                }
                if (l.type == LOCAL)
                {
                    if (CudaUtils.UseGpu)
                    {
                        pull_local_layer(l);
                    }
                    int locations = l.OutW * l.OutH;
                    int size = l.Size * l.Size * l.C * l.N * locations;
                    fwrite(l.Biases, sizeof(float), l.outputs, fp);
                    fwrite(l.Weights, sizeof(float), size, fp);
                }
            }
            fclose(fp);
        }

        public static void save_weights(Network net, string filename)
        {
            save_weights_upto(net, filename, net.N);
        }

        public static void transpose_matrix(float[] a, int rows, int cols)
        {
            float[] transpose = calloc(rows * cols, sizeof(float));
            int x, y;
            for (x = 0; x < rows; ++x)
            {
                for (y = 0; y < cols; ++y)
                {
                    transpose[y * rows + x] = a[x * cols + y];
                }
            }
            memcpy(a, transpose, rows * cols * sizeof(float));
            free(transpose);
        }

        public static void load_connected_weights(Layer l, FileStream fread, int transpose)
        {
            fread(l.Biases, sizeof(float), l.outputs, fp);
            fread(l.Weights, sizeof(float), l.outputs * l.Inputs, fp);
            if (transpose)
            {
                transpose_matrix(l.Weights, l.Inputs, l.outputs);
            }
            if (l.BatchNormalize && (!l.dontloadscales))
            {
                fread(l.Scales, sizeof(float), l.outputs, fp);
                fread(l.RollingMean, sizeof(float), l.outputs, fp);
                fread(l.RollingVariance, sizeof(float), l.outputs, fp);
            }
            if (CudaUtils.UseGpu)
            {
                push_connected_layer(l);
            }
        }

        public static void load_batchnorm_weights(Layer l, FileStream fread)
        {
            fread(l.Scales, sizeof(float), l.C, fp);
            fread(l.RollingMean, sizeof(float), l.C, fp);
            fread(l.RollingVariance, sizeof(float), l.C, fp);
            if (CudaUtils.UseGpu)
            {
                push_batchnorm_layer(l);
            }
        }

        public static void load_convolutional_weights_binary(Layer l, FileStream fread)
        {
            fread(l.Biases, sizeof(float), l.N, fp);
            if (l.BatchNormalize && (!l.dontloadscales))
            {
                fread(l.Scales, sizeof(float), l.N, fp);
                fread(l.RollingMean, sizeof(float), l.N, fp);
                fread(l.RollingVariance, sizeof(float), l.N, fp);
            }
            int size = l.C * l.Size * l.Size;
            int i, j, k;
            for (i = 0; i < l.N; ++i)
            {
                float mean = 0;
                fread(&mean, sizeof(float), 1, fp);
                for (j = 0; j < size / 8; ++j)
                {
                    int index = i * size + j * 8;
                    unsigned char c = 0;
                    fread(&c, sizeof(char), 1, fp);
                    for (k = 0; k < 8; ++k)
                    {
                        if (j * 8 + k >= size) break;
                        l.Weights[index + k] = (c & 1 << k) ? mean : -mean;
                    }
                }
            }
            if (CudaUtils.UseGpu)
            {
                push_convolutional_layer(l);
            }
        }

        public static void load_convolutional_weights(Layer l, FileStream fread)
        {
            int num = l.N * l.C * l.Size * l.Size;
            fread(l.Biases, sizeof(float), l.N, fp);
            if (l.BatchNormalize != 0 && (!l.Dontloadscales))
            {
                fread(l.Scales, sizeof(float), l.N, fp);
                fread(l.RollingMean, sizeof(float), l.N, fp);
                fread(l.RollingVariance, sizeof(float), l.N, fp);
                if (0)
                {
                    int i;
                    for (i = 0; i < l.N; ++i)
                    {
                        printf("%g, ", l.RollingMean[i]);
                    }
                    printf("\n");
                    for (i = 0; i < l.N; ++i)
                    {
                        printf("%g, ", l.RollingVariance[i]);
                    }
                    printf("\n");
                }
                if (0)
                {
                    fill_cpu(l.N, 0, l.RollingMean, 1);
                    fill_cpu(l.N, 0, l.RollingVariance, 1);
                }
            }
            fread(l.Weights, sizeof(float), num, fp);
            if (l.Adam)
            {
                fread(l.m, sizeof(float), num, fp);
                fread(l.v, sizeof(float), num, fp);
            }
            if (l.Flipped)
            {
                transpose_matrix(l.Weights, l.C * l.Size * l.Size, l.N);
            }
            if (CudaUtils.UseGpu)
            {
                push_convolutional_layer(l);
            }
        }

        public static void load_weights_upto(Network net, string filename, int cutoff)
        {
            Console.Error.Write($"Loading weights from {filename}...");
            fflush(stdout);
            if (!File.Exists(filename))
            {
                Utils.file_error(filename);
            }

            FileStream fread = fopen(filename, "rb");

            int major;
            int minor;
            int revision;
            fread(&major, sizeof(int), 1, fp);
            fread(&minor, sizeof(int), 1, fp);
            fread(&revision, sizeof(int), 1, fp);
            fread(net.seen, sizeof(int), 1, fp);
            int transpose = (major > 1000) || (minor > 1000) ? 1 : 0;

            int i;
            for (i = 0; i < net.N && i < cutoff; ++i)
            {
                Layer l = net.Layers[i];
                if (l.Dontload) continue;
                if (l.LayerType == LayerType.Convolutional)
                {
                    load_convolutional_weights(l, fp);
                }

                if (l.LayerType == LayerType.Connected)
                {
                    load_connected_weights(l, fp, transpose);
                }

                if (l.LayerType == LayerType.Batchnorm)
                {
                    load_batchnorm_weights(l, fstream);
                }

                if (l.LayerType == LayerType.Crnn)
                {
                    load_convolutional_weights((l.InputLayer), fp);
                    load_convolutional_weights((l.SelfLayer), fp);
                    load_convolutional_weights((l.OutputLayer), fp);
                }

                if (l.LayerType == LayerType.Rnn)
                {
                    load_connected_weights((l.InputLayer), fp, transpose);
                    load_connected_weights((l.SelfLayer), fp, transpose);
                    load_connected_weights((l.OutputLayer), fp, transpose);
                }

                if (l.LayerType == LayerType.Gru)
                {
                    load_connected_weights((l.InputZLayer), fp, transpose);
                    load_connected_weights((l.InputRLayer), fp, transpose);
                    load_connected_weights((l.InputHLayer), fp, transpose);
                    load_connected_weights((l.StateZLayer), fp, transpose);
                    load_connected_weights((l.StateRLayer), fp, transpose);
                    load_connected_weights((l.StateHLayer), fp, transpose);
                }

                if (l.LayerType == LayerType.Local)
                {
                    int locations = l.OutW * l.OutH;
                    int size = l.Size * l.Size * l.C * l.N * locations;
                    fread(l.Biases, sizeof(float), l.outputs, fp);
                    fread(l.Weights, sizeof(float), size, fp);
                    if (CudaUtils.UseGpu)
                    {
                        push_local_layer(l);
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
