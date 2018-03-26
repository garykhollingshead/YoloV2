using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public static class Parser
    {
        LayerType string_to_layer_type(string type)
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
                case "[network]":
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
        
        void parse_data(string data, float[] a)
        {
            if (string.IsNullOrEmpty(data)) return;
            var numbs = data.Split(',');
            a.AddRange(numbs.Select(float.Parse));
        }

        Layer parse_local(KeyValuePair[] options, SizeParams parameters)
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
            batch =parameters.Batch;
            if (!(h != 0 && w != 0 && c != 0)) Utils.Error("Layer before local layer must output image.");

            Layer layer = make_local_layer(batch, h, w, c, n, size, stride, pad, activation);

            return layer;
        }

        Layer parse_convolutional(list* options, SizeParams parameters)
        {
            int n = option_find_int(options, "filters", 1);
            int size = option_find_int(options, "size", 1);
            int stride = option_find_int(options, "stride", 1);
            int pad = option_find_int_quiet(options, "pad", 0);
            int padding = option_find_int_quiet(options, "padding", 0);
            if (pad) padding = size / 2;

            string activation_s = option_find_str(options, "activation", "logistic");
            ACTIVATION activation = get_activation(activation_s);

            int batch, h, w, c;
            h = parameters.h;
            w = parameters.w;
            c = parameters.c;
            batch =parameters.batch;
            if (!(h && w && c)) error("Layer before convolutional layer must output image.");
            int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
            int binary = option_find_int_quiet(options, "binary", 0);
            int xnor = option_find_int_quiet(options, "xnor", 0);

            Layer layer = make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, binary, xnor, parameters.net.adam);
            layer.flipped = option_find_int_quiet(options, "flipped", 0);
            layer.dot = option_find_float_quiet(options, "dot", 0);
            if (parameters.net.adam){
                layer.B1 = parameters.net.B1;
                layer.B2 = parameters.net.B2;
                layer.eps = parameters.net.eps;
            }

            return layer;
        }

        layer parse_crnn(list* options, SizeParams parameters)
        {
            int output_filters = option_find_int(options, "output_filters", 1);
            int hidden_filters = option_find_int(options, "hidden_filters", 1);
            string activation_s = option_find_str(options, "activation", "logistic");
            ACTIVATION activation = get_activation(activation_s);
            int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

            layer l = make_crnn_layer(parameters.batch, parameters.w, parameters.h, parameters.c, hidden_filters, output_filters, parameters.time_steps, activation, batch_normalize);

            l.shortcut = option_find_int_quiet(options, "shortcut", 0);

            return l;
        }

        layer parse_rnn(list* options, SizeParams parameters)
        {
            int output = option_find_int(options, "output", 1);
            int hidden = option_find_int(options, "hidden", 1);
            string activation_s = option_find_str(options, "activation", "logistic");
            ACTIVATION activation = get_activation(activation_s);
            int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
            int logistic = option_find_int_quiet(options, "logistic", 0);

            layer l = make_rnn_layer(parameters.batch, parameters.inputs, hidden, output, parameters.time_steps, activation, batch_normalize, logistic);

            l.shortcut = option_find_int_quiet(options, "shortcut", 0);

            return l;
        }

        layer parse_gru(list* options, SizeParams parameters)
        {
            int output = option_find_int(options, "output", 1);
            int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

            layer l = make_gru_layer(parameters.batch, parameters.inputs, output, parameters.time_steps, batch_normalize);

            return l;
        }

        connected_layer parse_connected(list* options, SizeParams parameters)
        {
            int output = option_find_int(options, "output", 1);
            string activation_s = option_find_str(options, "activation", "logistic");
            ACTIVATION activation = get_activation(activation_s);
            int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

            connected_layer layer = make_connected_layer(parameters.batch, parameters.inputs, output, activation, batch_normalize);

            return layer;
        }

        softmax_layer parse_softmax(list* options, SizeParams parameters)
        {
            int groups = option_find_int_quiet(options, "groups", 1);
            softmax_layer layer = make_softmax_layer(parameters.batch, parameters.inputs, groups);
            layer.temperature = option_find_float_quiet(options, "temperature", 1);
            string tree_file = option_find_str(options, "tree", 0);
            if (tree_file) layer.softmax_tree = read_tree(tree_file);
            return layer;
        }

        layer parse_region(list* options, SizeParams parameters)
        {
            int coords = option_find_int(options, "coords", 4);
            int classes = option_find_int(options, "classes", 20);
            int num = option_find_int(options, "num", 1);

            layer l = make_region_layer(parameters.batch, parameters.w, parameters.h, num, classes, coords);
            assert(l.outputs == parameters.inputs);

            l.log = option_find_int_quiet(options, "log", 0);
            l.sqrt = option_find_int_quiet(options, "sqrt", 0);

            l.softmax = option_find_int(options, "softmax", 0);
            l.max_boxes = option_find_int_quiet(options, "max", 30);
            l.jitter = option_find_float(options, "jitter", .2);
            l.rescore = option_find_int_quiet(options, "rescore", 0);

            l.thresh = option_find_float(options, "thresh", .5);
            l.classfix = option_find_int_quiet(options, "classfix", 0);
            l.absolute = option_find_int_quiet(options, "absolute", 0);
            l.random = option_find_int_quiet(options, "random", 0);

            l.coord_scale = option_find_float(options, "coord_scale", 1);
            l.object_scale = option_find_float(options, "object_scale", 1);
            l.noobject_scale = option_find_float(options, "noobject_scale", 1);
            l.class_scale = option_find_float(options, "class_scale", 1);
            l.bias_match = option_find_int_quiet(options, "bias_match", 0);

            string tree_file = option_find_str(options, "tree", 0);
            if (tree_file) l.softmax_tree = read_tree(tree_file);
            string map_file = option_find_str(options, "map", 0);
            if (map_file) l.map = read_map(map_file);

            string a = option_find_str(options, "anchors", 0);
            if (a)
            {
                int len = strlen(a);
                int n = 1;
                int i;
                for (i = 0; i < len; ++i)
                {
                    if (a[i] == ',') ++n;
                }
                for (i = 0; i < n; ++i)
                {
                    float bias = atof(a);
                    l.biases[i] = bias;
                    a = strchr(a, ',') + 1;
                }
            }
            return l;
        }
        detection_layer parse_detection(list* options, SizeParams parameters)
        {
            int coords = option_find_int(options, "coords", 1);
            int classes = option_find_int(options, "classes", 1);
            int rescore = option_find_int(options, "rescore", 0);
            int num = option_find_int(options, "num", 1);
            int side = option_find_int(options, "side", 7);
            detection_layer layer = make_detection_layer(parameters.batch, parameters.inputs, num, side, classes, coords, rescore);

            layer.softmax = option_find_int(options, "softmax", 0);
            layer.sqrt = option_find_int(options, "sqrt", 0);

            layer.max_boxes = option_find_int_quiet(options, "max", 30);
            layer.coord_scale = option_find_float(options, "coord_scale", 1);
            layer.forced = option_find_int(options, "forced", 0);
            layer.object_scale = option_find_float(options, "object_scale", 1);
            layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
            layer.class_scale = option_find_float(options, "class_scale", 1);
            layer.jitter = option_find_float(options, "jitter", .2);
            layer.random = option_find_int_quiet(options, "random", 0);
            layer.reorg = option_find_int_quiet(options, "reorg", 0);
            return layer;
        }

        cost_layer parse_cost(list* options, SizeParams parameters)
        {
            string type_s = option_find_str(options, "type", "sse");
            COST_TYPE type = get_cost_type(type_s);
            float scale = option_find_float_quiet(options, "scale", 1);
            cost_layer layer = make_cost_layer(parameters.batch, parameters.inputs, type, scale);
            layer.ratio = option_find_float_quiet(options, "ratio", 0);
            return layer;
        }

        crop_layer parse_crop(list* options, SizeParams parameters)
        {
            int crop_height = option_find_int(options, "crop_height", 1);
            int crop_width = option_find_int(options, "crop_width", 1);
            int flip = option_find_int(options, "flip", 0);
            float angle = option_find_float(options, "angle", 0);
            float saturation = option_find_float(options, "saturation", 1);
            float exposure = option_find_float(options, "exposure", 1);

            int batch, h, w, c;
            h = parameters.h;
            w = parameters.w;
            c = parameters.c;
            batch =parameters.batch;
            if (!(h && w && c)) error("Layer before crop layer must output image.");

            int noadjust = option_find_int_quiet(options, "noadjust", 0);

            crop_layer l = make_crop_layer(batch, h, w, c, crop_height, crop_width, flip, angle, saturation, exposure);
            l.shift = option_find_float(options, "shift", 0);
            l.noadjust = noadjust;
            return l;
        }

        layer parse_reorg(list* options, SizeParams parameters)
        {
            int stride = option_find_int(options, "stride", 1);
            int reverse = option_find_int_quiet(options, "reverse", 0);

            int batch, h, w, c;
            h = parameters.h;
            w = parameters.w;
            c = parameters.c;
            batch =parameters.batch;
            if (!(h && w && c)) error("Layer before reorg layer must output image.");

            layer layer = make_reorg_layer(batch, w, h, c, stride, reverse);
            return layer;
        }

        maxpool_layer parse_maxpool(list* options, SizeParams parameters)
        {
            int stride = option_find_int(options, "stride", 1);
            int size = option_find_int(options, "size", stride);
            int padding = option_find_int_quiet(options, "padding", (size - 1) / 2);

            int batch, h, w, c;
            h = parameters.h;
            w = parameters.w;
            c = parameters.c;
            batch =parameters.batch;
            if (!(h && w && c)) error("Layer before maxpool layer must output image.");

            maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
            return layer;
        }

        avgpool_layer parse_avgpool(list* options, SizeParams parameters)
        {
            int batch, w, h, c;
            w = parameters.w;
            h = parameters.h;
            c = parameters.c;
            batch =parameters.batch;
            if (!(h && w && c)) error("Layer before avgpool layer must output image.");

            avgpool_layer layer = make_avgpool_layer(batch, w, h, c);
            return layer;
        }

        dropout_layer parse_dropout(list* options, SizeParams parameters)
        {
            float probability = option_find_float(options, "probability", .5);
            dropout_layer layer = make_dropout_layer(parameters.batch, parameters.inputs, probability);
            layer.out_w = parameters.w;
            layer.out_h = parameters.h;
            layer.out_c = parameters.c;
            return layer;
        }

        layer parse_normalization(list* options, SizeParams parameters)
        {
            float alpha = option_find_float(options, "alpha", .0001);
            float beta = option_find_float(options, "beta", .75);
            float kappa = option_find_float(options, "kappa", 1);
            int size = option_find_int(options, "size", 5);
            layer l = make_normalization_layer(parameters.batch, parameters.w, parameters.h, parameters.c, size, alpha, beta, kappa);
            return l;
        }

        layer parse_batchnorm(list* options, SizeParams parameters)
        {
            layer l = make_batchnorm_layer(parameters.batch, parameters.w, parameters.h, parameters.c);
            return l;
        }

        layer parse_shortcut(list* options, SizeParams parameters, network net)
        {
            string l = option_find(options, "from");
            int index = atoi(l);
            if (index < 0) index = parameters.index + index;

            int batch = parameters.batch;
            layer from = net.layers[index];

            layer s = make_shortcut_layer(batch, index, parameters.w, parameters.h, parameters.c, from.out_w, from.out_h, from.out_c);

            string activation_s = option_find_str(options, "activation", "linear");
            ACTIVATION activation = get_activation(activation_s);
            s.activation = activation;
            return s;
        }


        layer parse_activation(list* options, SizeParams parameters)
        {
            string activation_s = option_find_str(options, "activation", "linear");
            ACTIVATION activation = get_activation(activation_s);

            layer l = make_activation_layer(parameters.batch, parameters.inputs, activation);

            l.out_h = parameters.h;
            l.out_w = parameters.w;
            l.out_c = parameters.c;
            l.h = parameters.h;
            l.w = parameters.w;
            l.c = parameters.c;

            return l;
        }

        route_layer parse_route(list* options, SizeParams parameters, network net)
        {
            string l = option_find(options, "layers");
            int len = strlen(l);
            if (!l) error("Route Layer must specify input layers");
            int n = 1;
            int i;
            for (i = 0; i < len; ++i)
            {
                if (l[i] == ',') ++n;
            }

            int* layers = calloc(n, sizeof(int));
            int* sizes = calloc(n, sizeof(int));
            for (i = 0; i < n; ++i)
            {
                int index = atoi(l);
                l = strchr(l, ',') + 1;
                if (index < 0) index = parameters.index + index;
                layers[i] = index;
                sizes[i] = net.layers[index].outputs;
            }
            int batch = parameters.batch;

            route_layer layer = make_route_layer(batch, n, layers, sizes);

            Layer first = net.layers[layers[0]];
            layer.out_w = first.out_w;
            layer.out_h = first.out_h;
            layer.out_c = first.out_c;
            for (i = 1; i < n; ++i)
            {
                int index = layers[i];
                Layer next = net.layers[index];
                if (next.out_w == first.out_w && next.out_h == first.out_h)
                {
                    layer.out_c += next.out_c;
                }
                else
                {
                    layer.out_h = layer.out_w = layer.out_c = 0;
                }
            }

            return layer;
        }

        learning_rate_policy get_policy(string s)
        {
            if (s, "random") == 0) return RANDOM;
            if (s, "poly") == 0) return POLY;
            if (s, "constant") == 0) return CONSTANT;
            if (s, "step") == 0) return STEP;
            if (s, "exp") == 0) return EXP;
            if (s, "sigmoid") == 0) return SIG;
            if (s, "steps") == 0) return STEPS;
            fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
            return CONSTANT;
        }

        void parse_net_options(list* options, network* net)
        {
            net->batch = option_find_int(options, "batch", 1);
            net->learning_rate = option_find_float(options, "learning_rate", .001);
            net->momentum = option_find_float(options, "momentum", .9);
            net->decay = option_find_float(options, "decay", .0001);
            int subdivs = option_find_int(options, "subdivisions", 1);
            net->time_steps = option_find_int_quiet(options, "time_steps", 1);
            net->batch /= subdivs;
            net->batch *= net->time_steps;
            net->subdivisions = subdivs;

            net->adam = option_find_int_quiet(options, "adam", 0);
            if (net->adam)
            {
                net->B1 = option_find_float(options, "B1", .9);
                net->B2 = option_find_float(options, "B2", .999);
                net->eps = option_find_float(options, "eps", .000001);
            }

            net->h = option_find_int_quiet(options, "height", 0);
            net->w = option_find_int_quiet(options, "width", 0);
            net->c = option_find_int_quiet(options, "channels", 0);
            net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
            net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
            net->min_crop = option_find_int_quiet(options, "min_crop", net->w);

            net->angle = option_find_float_quiet(options, "angle", 0);
            net->aspect = option_find_float_quiet(options, "aspect", 1);
            net->saturation = option_find_float_quiet(options, "saturation", 1);
            net->exposure = option_find_float_quiet(options, "exposure", 1);
            net->hue = option_find_float_quiet(options, "hue", 0);

            if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

            string policy_s = option_find_str(options, "policy", "constant");
            net->policy = get_policy(policy_s);
            net->burn_in = option_find_int_quiet(options, "burn_in", 0);
            if (net->policy == STEP)
            {
                net->step = option_find_int(options, "step", 1);
                net->scale = option_find_float(options, "scale", 1);
            }
            else if (net->policy == STEPS)
            {
                string l = option_find(options, "steps");
                string p = option_find(options, "scales");
                if (!l || !p) error("STEPS policy must have steps and scales in cfg file");

                int len = strlen(l);
                int n = 1;
                int i;
                for (i = 0; i < len; ++i)
                {
                    if (l[i] == ',') ++n;
                }
                int* steps = calloc(n, sizeof(int));
                float[] scales = calloc(n, sizeof(float));
                for (i = 0; i < n; ++i)
                {
                    int step = atoi(l);
                    float scale = atof(p);
                    l = strchr(l, ',') + 1;
                    p = strchr(p, ',') + 1;
                    steps[i] = step;
                    scales[i] = scale;
                }
                net->scales = scales;
                net->steps = steps;
                net->num_steps = n;
            }
            else if (net->policy == EXP)
            {
                net->gamma = option_find_float(options, "gamma", 1);
            }
            else if (net->policy == SIG)
            {
                net->gamma = option_find_float(options, "gamma", 1);
                net->step = option_find_int(options, "step", 1);
            }
            else if (net->policy == POLY || net->policy == RANDOM)
            {
                net->power = option_find_float(options, "power", 1);
            }
            net->max_batches = option_find_int(options, "max_batches", 0);
        }

        int is_network(section* s)
        {
            return (s->type, "[net]") == 0
                    || s->type, "[network]") == 0);
        }

        network parse_network_cfg(string filename)
        {
            list* sections = read_cfg(filename);
            node* n = sections->front;
            if (!n) error("Config file has no sections");
            network net = make_network(sections->size - 1);
            net.gpu_index = gpu_index;
            SizeParams parameters;

            section* s = (section*)n->val;
            list* options = s->options;
            if (!is_network(s)) error("First section must be [net] or [network]");
            parse_net_options(options, &net);

    parameters.h = net.h;
    parameters.w = net.w;
    parameters.c = net.c;
    parameters.inputs = net.inputs;
    parameters.batch = net.batch;
    parameters.time_steps = net.time_steps;
    parameters.net = net;

            size_t workspace_size = 0;
            n = n->next;
            int count = 0;
            free_section(s);
            fprintf(stderr, "layer     filters    size              input                output\n");
            while (n)
            {
        parameters.index = count;
                fprintf(stderr, "%5d ", count);
                s = (section*)n->val;
                options = s->options;
                layer l = { 0 };
                LayerType lt = string_to_layer_type(s->type);
                if (lt == CONVOLUTIONAL)
                {
                    l = parse_convolutional(options, parameters);
                }
                else if (lt == LOCAL)
                {
                    l = parse_local(options, parameters);
                }
                else if (lt == ACTIVE)
                {
                    l = parse_activation(options, parameters);
                }
                else if (lt == RNN)
                {
                    l = parse_rnn(options, parameters);
                }
                else if (lt == GRU)
                {
                    l = parse_gru(options, parameters);
                }
                else if (lt == CRNN)
                {
                    l = parse_crnn(options, parameters);
                }
                else if (lt == CONNECTED)
                {
                    l = parse_connected(options, parameters);
                }
                else if (lt == CROP)
                {
                    l = parse_crop(options, parameters);
                }
                else if (lt == COST)
                {
                    l = parse_cost(options, parameters);
                }
                else if (lt == REGION)
                {
                    l = parse_region(options, parameters);
                }
                else if (lt == DETECTION)
                {
                    l = parse_detection(options, parameters);
                }
                else if (lt == SOFTMAX)
                {
                    l = parse_softmax(options, parameters);
                    net.hierarchy = l.softmax_tree;
                }
                else if (lt == NORMALIZATION)
                {
                    l = parse_normalization(options, parameters);
                }
                else if (lt == BATCHNORM)
                {
                    l = parse_batchnorm(options, parameters);
                }
                else if (lt == MAXPOOL)
                {
                    l = parse_maxpool(options, parameters);
                }
                else if (lt == REORG)
                {
                    l = parse_reorg(options, parameters);
                }
                else if (lt == AVGPOOL)
                {
                    l = parse_avgpool(options, parameters);
                }
                else if (lt == ROUTE)
                {
                    l = parse_route(options, parameters, net);
                }
                else if (lt == SHORTCUT)
                {
                    l = parse_shortcut(options, parameters, net);
                }
                else if (lt == DROPOUT)
                {
                    l = parse_dropout(options, parameters);
                    l.output = net.layers[count - 1].output;
                    l.delta = net.layers[count - 1].delta;
                    l.output_gpu = net.layers[count - 1].output_gpu;
                    l.delta_gpu = net.layers[count - 1].delta_gpu;
                }
                else
                {
                    fprintf(stderr, "Type not recognized: %s\n", s->type);
                }
                l.dontload = option_find_int_quiet(options, "dontload", 0);
                l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
                option_unused(options);
                net.layers[count] = l;
                if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
                free_section(s);
                n = n->next;
                ++count;
                if (n)
                {
            parameters.h = l.out_h;
            parameters.w = l.out_w;
            parameters.c = l.out_c;
            parameters.inputs = l.outputs;
                }
            }
            free_list(sections);
            net.outputs = get_network_output_size(net);
            net.output = get_network_output(net);
            if (workspace_size)
            {
                if (gpu_index >= 0)
                {
                    net.workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
                }
                else
                {
                    net.workspace = calloc(1, workspace_size);
                }
            }
            return net;
        }

        list* read_cfg(string filename)
        {
            FILE* file = fopen(filename, "r");
            if (file == 0) file_error(filename);
            string line;
            int nu = 0;
            list* sections = make_list();
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
                        current->options = make_list();
                        current->type = line;
                        break;
                    case '\0':
                    case '#':
                    case ';':
                        free(line);
                        break;
                    default:
                        if (!read_option(line, current->options))
                        {
                            fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                            free(line);
                        }
                        break;
                }
            }
            fclose(file);
            return sections;
        }

        void save_convolutional_weights_binary(layer l, FILE* fp)
        {
            if (gpu_index >= 0)
            {
                pull_convolutional_layer(l);
            }
            binarize_weights(l.weights, l.n, l.c * l.size * l.size, l.binary_weights);
            int size = l.c * l.size * l.size;
            int i, j, k;
            fwrite(l.biases, sizeof(float), l.n, fp);
            if (l.batch_normalize)
            {
                fwrite(l.scales, sizeof(float), l.n, fp);
                fwrite(l.rolling_mean, sizeof(float), l.n, fp);
                fwrite(l.rolling_variance, sizeof(float), l.n, fp);
            }
            for (i = 0; i < l.n; ++i)
            {
                float mean = l.binary_weights[i * size];
                if (mean < 0) mean = -mean;
                fwrite(&mean, sizeof(float), 1, fp);
                for (j = 0; j < size / 8; ++j)
                {
                    int index = i * size + j * 8;
                    unsigned char c = 0;
                    for (k = 0; k < 8; ++k)
                    {
                        if (j * 8 + k >= size) break;
                        if (l.binary_weights[index + k] > 0) c = (c | 1 << k);
                    }
                    fwrite(&c, sizeof(char), 1, fp);
                }
            }
        }

        void save_convolutional_weights(layer l, FILE* fp)
        {
            if (l.binary)
            {
                //save_convolutional_weights_binary(l, fp);
                //return;
            }
            if (gpu_index >= 0)
            {
                pull_convolutional_layer(l);
            }
            int num = l.n * l.c * l.size * l.size;
            fwrite(l.biases, sizeof(float), l.n, fp);
            if (l.batch_normalize)
            {
                fwrite(l.scales, sizeof(float), l.n, fp);
                fwrite(l.rolling_mean, sizeof(float), l.n, fp);
                fwrite(l.rolling_variance, sizeof(float), l.n, fp);
            }
            fwrite(l.weights, sizeof(float), num, fp);
            if (l.adam)
            {
                fwrite(l.m, sizeof(float), num, fp);
                fwrite(l.v, sizeof(float), num, fp);
            }
        }

        void save_batchnorm_weights(layer l, FILE* fp)
        {
            if (gpu_index >= 0)
            {
                pull_batchnorm_layer(l);
            }
            fwrite(l.scales, sizeof(float), l.c, fp);
            fwrite(l.rolling_mean, sizeof(float), l.c, fp);
            fwrite(l.rolling_variance, sizeof(float), l.c, fp);
        }

        void save_connected_weights(layer l, FILE* fp)
        {
            if (gpu_index >= 0)
            {
                pull_connected_layer(l);
            }
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), l.outputs * l.inputs, fp);
            if (l.batch_normalize)
            {
                fwrite(l.scales, sizeof(float), l.outputs, fp);
                fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
                fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
            }
        }

        void save_weights_upto(network net, string filename, int cutoff)
        {
            if (net.gpu_index >= 0)
            {
                cuda_set_device(net.gpu_index);
            }
            fprintf(stderr, "Saving weights to %s\n", filename);
            FILE* fp = fopen(filename, "wb");
            if (!fp) file_error(filename);

            int major = 0;
            int minor = 1;
            int revision = 0;
            fwrite(&major, sizeof(int), 1, fp);
            fwrite(&minor, sizeof(int), 1, fp);
            fwrite(&revision, sizeof(int), 1, fp);
            fwrite(net.seen, sizeof(int), 1, fp);

            int i;
            for (i = 0; i < net.n && i < cutoff; ++i)
            {
                layer l = net.layers[i];
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
                    save_connected_weights(*(l.input_layer), fp);
                    save_connected_weights(*(l.self_layer), fp);
                    save_connected_weights(*(l.output_layer), fp);
                }
                if (l.type == GRU)
                {
                    save_connected_weights(*(l.input_z_layer), fp);
                    save_connected_weights(*(l.input_r_layer), fp);
                    save_connected_weights(*(l.input_h_layer), fp);
                    save_connected_weights(*(l.state_z_layer), fp);
                    save_connected_weights(*(l.state_r_layer), fp);
                    save_connected_weights(*(l.state_h_layer), fp);
                }
                if (l.type == CRNN)
                {
                    save_convolutional_weights(*(l.input_layer), fp);
                    save_convolutional_weights(*(l.self_layer), fp);
                    save_convolutional_weights(*(l.output_layer), fp);
                }
                if (l.type == LOCAL)
                {
                    if (gpu_index >= 0)
                    {
                        pull_local_layer(l);
                    }
                    int locations = l.out_w * l.out_h;
                    int size = l.size * l.size * l.c * l.n * locations;
                    fwrite(l.biases, sizeof(float), l.outputs, fp);
                    fwrite(l.weights, sizeof(float), size, fp);
                }
            }
            fclose(fp);
        }
        void save_weights(network net, string filename)
        {
            save_weights_upto(net, filename, net.n);
        }

        void transpose_matrix(float[] a, int rows, int cols)
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

        void load_connected_weights(layer l, FILE* fp, int transpose)
        {
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), l.outputs * l.inputs, fp);
            if (transpose)
            {
                transpose_matrix(l.weights, l.inputs, l.outputs);
            }
            if (l.batch_normalize && (!l.dontloadscales))
            {
                fread(l.scales, sizeof(float), l.outputs, fp);
                fread(l.rolling_mean, sizeof(float), l.outputs, fp);
                fread(l.rolling_variance, sizeof(float), l.outputs, fp);
            }
            if (gpu_index >= 0)
            {
                push_connected_layer(l);
            }
        }

        void load_batchnorm_weights(layer l, FILE* fp)
        {
            fread(l.scales, sizeof(float), l.c, fp);
            fread(l.rolling_mean, sizeof(float), l.c, fp);
            fread(l.rolling_variance, sizeof(float), l.c, fp);
            if (gpu_index >= 0)
            {
                push_batchnorm_layer(l);
            }
        }

        void load_convolutional_weights_binary(layer l, FILE* fp)
        {
            fread(l.biases, sizeof(float), l.n, fp);
            if (l.batch_normalize && (!l.dontloadscales))
            {
                fread(l.scales, sizeof(float), l.n, fp);
                fread(l.rolling_mean, sizeof(float), l.n, fp);
                fread(l.rolling_variance, sizeof(float), l.n, fp);
            }
            int size = l.c * l.size * l.size;
            int i, j, k;
            for (i = 0; i < l.n; ++i)
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
                        l.weights[index + k] = (c & 1 << k) ? mean : -mean;
                    }
                }
            }
            if (gpu_index >= 0)
            {
                push_convolutional_layer(l);
            }
        }

        void load_convolutional_weights(layer l, FILE* fp)
        {
            int num = l.n * l.c * l.size * l.size;
            fread(l.biases, sizeof(float), l.n, fp);
            if (l.batch_normalize && (!l.dontloadscales))
            {
                fread(l.scales, sizeof(float), l.n, fp);
                fread(l.rolling_mean, sizeof(float), l.n, fp);
                fread(l.rolling_variance, sizeof(float), l.n, fp);
                if (0)
                {
                    int i;
                    for (i = 0; i < l.n; ++i)
                    {
                        printf("%g, ", l.rolling_mean[i]);
                    }
                    printf("\n");
                    for (i = 0; i < l.n; ++i)
                    {
                        printf("%g, ", l.rolling_variance[i]);
                    }
                    printf("\n");
                }
                if (0)
                {
                    fill_cpu(l.n, 0, l.rolling_mean, 1);
                    fill_cpu(l.n, 0, l.rolling_variance, 1);
                }
            }
            fread(l.weights, sizeof(float), num, fp);
            if (l.adam)
            {
                fread(l.m, sizeof(float), num, fp);
                fread(l.v, sizeof(float), num, fp);
            }
            if (l.flipped)
            {
                transpose_matrix(l.weights, l.c * l.size * l.size, l.n);
            }
            if (gpu_index >= 0)
            {
                push_convolutional_layer(l);
            }
        }


        void load_weights_upto(network* net, string filename, int cutoff)
        {
            if (net->gpu_index >= 0)
            {
                cuda_set_device(net->gpu_index);
            }
            fprintf(stderr, "Loading weights from %s...", filename);
            fflush(stdout);
            FILE* fp = fopen(filename, "rb");
            if (!fp) file_error(filename);

            int major;
            int minor;
            int revision;
            fread(&major, sizeof(int), 1, fp);
            fread(&minor, sizeof(int), 1, fp);
            fread(&revision, sizeof(int), 1, fp);
            fread(net->seen, sizeof(int), 1, fp);
            int transpose = (major > 1000) || (minor > 1000);

            int i;
            for (i = 0; i < net->n && i < cutoff; ++i)
            {
                layer l = net->layers[i];
                if (l.dontload) continue;
                if (l.type == CONVOLUTIONAL)
                {
                    load_convolutional_weights(l, fp);
                }
                if (l.type == CONNECTED)
                {
                    load_connected_weights(l, fp, transpose);
                }
                if (l.type == BATCHNORM)
                {
                    load_batchnorm_weights(l, fp);
                }
                if (l.type == CRNN)
                {
                    load_convolutional_weights(*(l.input_layer), fp);
                    load_convolutional_weights(*(l.self_layer), fp);
                    load_convolutional_weights(*(l.output_layer), fp);
                }
                if (l.type == RNN)
                {
                    load_connected_weights(*(l.input_layer), fp, transpose);
                    load_connected_weights(*(l.self_layer), fp, transpose);
                    load_connected_weights(*(l.output_layer), fp, transpose);
                }
                if (l.type == GRU)
                {
                    load_connected_weights(*(l.input_z_layer), fp, transpose);
                    load_connected_weights(*(l.input_r_layer), fp, transpose);
                    load_connected_weights(*(l.input_h_layer), fp, transpose);
                    load_connected_weights(*(l.state_z_layer), fp, transpose);
                    load_connected_weights(*(l.state_r_layer), fp, transpose);
                    load_connected_weights(*(l.state_h_layer), fp, transpose);
                }
                if (l.type == LOCAL)
                {
                    int locations = l.out_w * l.out_h;
                    int size = l.size * l.size * l.c * l.n * locations;
                    fread(l.biases, sizeof(float), l.outputs, fp);
                    fread(l.weights, sizeof(float), size, fp);
                    if (gpu_index >= 0)
                    {
                        push_local_layer(l);
                    }
                }
            }
            fprintf(stderr, "Done!\n");
            fclose(fp);
        }

        void load_weights(network* net, string filename)
        {
            load_weights_upto(net, filename, net->n);
        }

    }
}
