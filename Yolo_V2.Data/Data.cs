using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class Data
    {
        public int W;
        public int H;
        public Matrix X;
        public Matrix Y;
        public int Shallow;
        public List<int> NumBoxes;
        public Box[][] Boxes;

        private static readonly object mutexLock = new object();

        public static List<string> GetPaths(string filename)
        {
            return !File.Exists(filename)
                ? new List<string>()
                : File.ReadAllLines(filename).ToList();
        }

        string[] get_random_paths(string[] paths, int n, int m)
        {
            string[] random_paths = new string[n];
            int i;
            lock (mutexLock)
            {
                for (i = 0; i < n; ++i)
                {
                    int index = Utils.Rand.Next() % m;
                    random_paths[i] = paths[index];
                }
            }
            return random_paths;
        }

        string[] find_replace_paths(string[] paths, int n, string find, string replace)
        {
            string[] replace_paths = new string[n];
            int i;
            for (i = 0; i < n; ++i)
            {
                replace_paths[i] = paths[i].Replace(find, replace);
            }

            return replace_paths;
        }

        Matrix load_image_paths_gray(string[] paths, int n, int w, int h)
        {
            int i;
            Matrix X = new Matrix(n);

            for (i = 0; i < n; ++i)
            {
                Image im = load_image(paths[i], w, h, 3);

                Image gray = grayscale_image(im);
                im = gray;

                X.Vals[i] = im.Data;
                X.Cols = im.H * im.W * im.C;
            }
            return X;
        }

        Matrix load_image_paths(string[] paths, int n, int w, int h)
        {
            int i;
            Matrix X = new Matrix(n);

            for (i = 0; i < n; ++i)
            {
                Image im = LoadArgs.load_image_color(paths[i], w, h);
                X.Vals[i] = im.Data;
                X.Cols = im.H * im.W * im.C;
            }
            return X;
        }

        Matrix load_image_augment_paths(string[] paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
        {
            int i;
            Matrix X = new Matrix(n);

            for (i = 0; i < n; ++i)
            {
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                Image crop = random_augment_image(im, angle, aspect, min, max, size);
                int flip = Utils.Rand.Next() % 2;
                if (flip != 0) flip_image(crop);
                random_distort_image(crop, hue, saturation, exposure);

                X.Vals[i] = crop.Data;
                X.Cols = crop.H * crop.W * crop.C;
            }
            return X;
        }


        BoxLabel[] read_boxes(string filename, ref int n)
        {
            if (!File.Exists(filename))
            {
                Utils.file_error(filename);
            }

            float x, y, h, w;
            int id;
            int count = 0;
            var lines = File.ReadAllLines(filename);
            BoxLabel[] boxes = new BoxLabel[lines.Length];
            for (var i = 0; i < lines.Length; ++i)
            {
                var numbers = lines[i].Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (numbers.Length != 5)
                {
                    continue;
                }
                id = int.Parse(numbers[0]);
                x = float.Parse(numbers[1]);
                y = float.Parse(numbers[2]);
                w = float.Parse(numbers[3]);
                h = float.Parse(numbers[4]);
                boxes[i] = new BoxLabel();
                boxes[i].Id = id;
                boxes[i].X = x;
                boxes[i].Y = y;
                boxes[i].H = h;
                boxes[i].W = w;
                boxes[i].Left = x - w / 2;
                boxes[i].Right = x + w / 2;
                boxes[i].Top = y - h / 2;
                boxes[i].Bottom = y + h / 2;
                ++count;
            }
            n = count;
            return boxes;
        }

        void randomize_boxes(BoxLabel[] b, int n)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                BoxLabel swap = b[i];
                int index = Utils.Rand.Next() % n;
                b[i] = b[index];
                b[index] = swap;
            }
        }

        void correct_boxes(BoxLabel[] boxes, int n, float dx, float dy, float sx, float sy, int flip)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                if (boxes[i].X == 0 && boxes[i].Y == 0)
                {
                    boxes[i].X = 999999;
                    boxes[i].Y = 999999;
                    boxes[i].W = 999999;
                    boxes[i].H = 999999;
                    continue;
                }
                boxes[i].Left = boxes[i].Left * sx - dx;
                boxes[i].Right = boxes[i].Right * sx - dx;
                boxes[i].Top = boxes[i].Top * sy - dy;
                boxes[i].Bottom = boxes[i].Bottom * sy - dy;

                if (flip != 0)
                {
                    float swap = boxes[i].Left;
                    boxes[i].Left = 1.0f - boxes[i].Right;
                    boxes[i].Right = 1.0f - swap;
                }

                boxes[i].Left = Utils.Constrain(0, 1, boxes[i].Left);
                boxes[i].Right = Utils.Constrain(0, 1, boxes[i].Right);
                boxes[i].Top = Utils.Constrain(0, 1, boxes[i].Top);
                boxes[i].Bottom = Utils.Constrain(0, 1, boxes[i].Bottom);

                boxes[i].X = (boxes[i].Left + boxes[i].Right) / 2;
                boxes[i].Y = (boxes[i].Top + boxes[i].Bottom) / 2;
                boxes[i].W = (boxes[i].Right - boxes[i].Left);
                boxes[i].H = (boxes[i].Bottom - boxes[i].Top);

                boxes[i].W = Utils.Constrain(0, 1, boxes[i].W);
                boxes[i].H = Utils.Constrain(0, 1, boxes[i].H);
            }
        }

        void fill_truth_swag(string path, float[] truth, int classes, int flip, float dx, float dy, float sx, float sy)
        {
            string labelpath = path.Replace("images", "labels");
            labelpath = path.Replace("JPEGImages", "labels");
            labelpath = path.Replace(".jpg", ".txt");
            labelpath = path.Replace(".JPG", ".txt");
            labelpath = path.Replace(".JPEG", ".txt");

            int count = 0;
            BoxLabel[] boxes = read_boxes(labelpath, ref count);
            randomize_boxes(boxes, count);
            correct_boxes(boxes, count, dx, dy, sx, sy, flip);
            float x, y, w, h;
            int id;
            int i;

            for (i = 0; i < count && i < 30; ++i)
            {
                x = boxes[i].X;
                y = boxes[i].Y;
                w = boxes[i].W;
                h = boxes[i].H;
                id = boxes[i].Id;

                if (w < .0 || h < .0) continue;

                int index = (4 + classes) * i;

                truth[index++] = x;
                truth[index++] = y;
                truth[index++] = w;
                truth[index++] = h;

                if (id < classes) truth[index + id] = 1;
            }
        }

        void fill_truth_region(string path, float[] truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
        {
            string labelpath = path.Replace("images", "labels");
            labelpath = path.Replace("JPEGImages", "labels");
            labelpath = path.Replace(".jpg", ".txt");
            labelpath = path.Replace(".JPG", ".txt");
            labelpath = path.Replace(".JPEG", ".txt");
            labelpath = path.Replace(".png", ".txt");
            int count = 0;
            BoxLabel[] boxes = read_boxes(labelpath, ref count);
            randomize_boxes(boxes, count);
            correct_boxes(boxes, count, dx, dy, sx, sy, flip);
            float x, y, w, h;
            int id;
            int i;

            for (i = 0; i < count; ++i)
            {
                x = boxes[i].X;
                y = boxes[i].Y;
                w = boxes[i].W;
                h = boxes[i].H;
                id = boxes[i].Id;

                if (w < .01 || h < .01) continue;

                int col = (int)(x * num_boxes);
                int row = (int)(y * num_boxes);

                x = x * num_boxes - col;
                y = y * num_boxes - row;

                int index = (col + row * num_boxes) * (5 + classes);
                if (index >= truth.Length) continue;
                truth[index++] = 1;

                if (id < classes) truth[index + id] = 1;
                index += classes;

                truth[index++] = x;
                truth[index++] = y;
                truth[index++] = w;
                truth[index++] = h;
            }
        }

        void fill_truth_detection(string path, int num_boxes, float[] truth, int classes, int flip, float dx, float dy, float sx, float sy)
        {
            string labelpath = path.Replace("images", "labels");
            labelpath = path.Replace("JPEGImages", "labels");
            labelpath = path.Replace(".jpg", ".txt");
            labelpath = path.Replace(".JPG", ".txt");
            labelpath = path.Replace(".JPEG", ".txt");
            labelpath = path.Replace(".raw", ".txt");
            labelpath = path.Replace(".png", ".txt");
            int count = 0;
            BoxLabel[] boxes = read_boxes(labelpath, ref count);
            randomize_boxes(boxes, count);
            correct_boxes(boxes, count, dx, dy, sx, sy, flip);
            if (count > num_boxes) count = num_boxes;
            float x, y, w, h;
            int id;
            int i;

            for (i = 0; i < count; ++i)
            {
                x = boxes[i].X;
                y = boxes[i].Y;
                w = boxes[i].W;
                h = boxes[i].H;
                id = boxes[i].Id;

                if ((w < .01 || h < .01)) continue;

                truth[i * 5 + 0] = x;
                truth[i * 5 + 1] = y;
                truth[i * 5 + 2] = w;
                truth[i * 5 + 3] = h;
                truth[i * 5 + 4] = id;
            }
        }

        public static readonly int NumChars = 37;

        void print_letters(float[] pred, int n)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                var p = pred.Skip(Int32.MaxValue * NumChars).ToArray();
                int index = Utils.max_index(p, NumChars);
                Console.Write($"{index}");
            }
            Console.Write("\n");
        }

        void fill_truth_captcha(string path, int n, float[] truth)
        {
            string begin = path.Split('/').Last();
            int i;
            for (i = 0; i < begin.Length && i < n && begin[i] != '.'; ++i)
            {
                int index = int.Parse(begin[i].ToString());
                if (index > 35) Console.Write($"Bad {begin[1]}\n");
                truth[i * NumChars + index] = 1;
            }
            for (; i < n; ++i)
            {
                truth[i * NumChars + NumChars - 1] = 1;
            }
        }

        Data load_data_captcha(string[] paths, int n, int m, int k, int w, int h)
        {
            if (m != 0)
            {
                paths = get_random_paths(paths, n, m);
            }
            Data d = new Data();
            d.Shallow = 0;
            d.X = load_image_paths(paths, n, w, h);
            d.Y = new Matrix(n, k * NumChars);
            int i;
            for (i = 0; i < n; ++i)
            {
                fill_truth_captcha(paths[i], k, d.Y.Vals[i]);
            }
            return d;
        }

        Data load_data_captcha_encode(string[] paths, int n, int m, int w, int h)
        {
            if (m != 0) paths = get_random_paths(paths, n, m);
            Data d = new Data();
            d.Shallow = 0;
            d.X = load_image_paths(paths, n, w, h);
            d.X.Cols = 17100;
            d.Y = d.X;
            return d;
        }

        void fill_truth(string path, string[] labels, int k, float[] truth)
        {
            int i;
            truth = new float[truth.Length];
            int count = 0;
            for (i = 0; i < k; ++i)
            {
                if (path.Contains(labels[i]))
                {
                    truth[i] = 1;
                    ++count;
                }
            }
            if (count != 1) Console.Write($"Too many or too few labels: {count}, {path}\n", count, path);
        }

        void fill_hierarchy(float[] truth, int k, Tree hierarchy)
        {
            int j;
            for (j = 0; j < k; ++j)
            {
                if (j < truth.Length)
                {
                    int parent = hierarchy.Parent[j];
                    while (parent >= 0)
                    {
                        truth[parent] = 1;
                        parent = hierarchy.Parent[parent];
                    }
                }
            }
            int i;
            int count = 0;
            for (j = 0; j < hierarchy.Groups; ++j)
            {
                int mask = 1;
                for (i = 0; i < hierarchy.GroupSize[j]; ++i)
                {
                    if (truth[count + i] != 0)
                    {
                        mask = 0;
                        break;
                    }
                }
                if (mask != 0)
                {
                    for (i = 0; i < hierarchy.GroupSize[j]; ++i)
                    {
                        truth[count + i] = SECRET_NUM;
                    }
                }
                count += hierarchy.GroupSize[j];
            }
        }

        Matrix load_labels_paths(string[] paths, int n, string[] labels, int k, Tree hierarchy)
        {
            Matrix y = new Matrix(n, k);
            int i;
            for (i = 0; i < n && labels.Any(); ++i)
            {
                fill_truth(paths[i], labels, k, y.Vals[i]);
                if (hierarchy != null)
                {
                    fill_hierarchy(y.Vals[i], k, hierarchy);
                }
            }
            return y;
        }

        Matrix load_tags_paths(string[] paths, int n, int k)
        {
            Matrix y = new Matrix(n, k);
            int i;
            int count = 0;
            for (i = 0; i < n; ++i)
            {
                string label = paths[i].Replace("imgs", "labels");
                label = label.Replace("_icon1.jpeg", ".txt");

                if (!File.Exists(label))
                {
                    label = label.Replace("labels", "labels2");

                    if (!File.Exists(label))
                    {
                        continue;
                    }
                }

                var lines = File.ReadAllLines(label);
                count = lines.Length;
                int tag;
                foreach (var line in lines)
                {
                    tag = int.Parse(line.Split(' ').ToArray()[0]);
                    if (tag < k)
                    {
                        y.Vals[i][tag] = 1;
                    }
                }
            }
            Console.Write($"{count}/{n}\n", count, n);
            return y;
        }

        string[] get_labels(string filename)
        {
            return GetPaths(filename).ToArray();
        }

        Data load_data_region(int n, string[] paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
        {
            string[] random_paths = get_random_paths(paths, n, m);
            int i;
            Data d = new Data();
            d.Shallow = 0;

            d.X.Rows = n;
            d.X.Vals = new float[d.X.Rows][];
            d.X.Cols = h * w * 3;


            int k = size * size * (5 + classes);
            d.Y = new Matrix(n, k);
            for (i = 0; i < n; ++i)
            {
                Image orig = LoadArgs.load_image_color(random_paths[i], 0, 0);

                int oh = orig.H;
                int ow = orig.W;

                int dw = (int)(ow * jitter);
                int dh = (int)(oh * jitter);

                int pleft = (int)Utils.rand_uniform(-dw, dw);
                int pright = (int)Utils.rand_uniform(-dw, dw);
                int ptop = (int)Utils.rand_uniform(-dh, dh);
                int pbot = (int)Utils.rand_uniform(-dh, dh);

                int swidth = ow - pleft - pright;
                int sheight = oh - ptop - pbot;

                float sx = (float)swidth / ow;
                float sy = (float)sheight / oh;

                int flip = Utils.Rand.Next() % 2;
                Image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

                float dx = ((float)pleft / ow) / sx;
                float dy = ((float)ptop / oh) / sy;

                Image sized = resize_image(cropped, w, h);
                if (flip != 0) flip_image(sized);
                random_distort_image(sized, hue, saturation, exposure);
                d.X.Vals[i] = sized.Data;

                fill_truth_region(random_paths[i], d.Y.Vals[i], classes, size, flip, dx, dy, 1.0f / sx, 1.0f / sy);
            }
            return d;
        }

        Data load_data_compare(int n, string[] paths, int m, int classes, int w, int h)
        {
            if (m != 0) paths = get_random_paths(paths, 2 * n, m);
            int i, j;
            Data d = new Data();
            d.Shallow = 0;

            d.X.Rows = n;
            d.X.Vals = new float[d.X.Rows][];
            d.X.Cols = h * w * 6;

            int k = 2 * (classes);
            d.Y = new Matrix(n, k);
            for (i = 0; i < n; ++i)
            {
                Image im1 = LoadArgs.load_image_color(paths[i * 2], w, h);
                Image im2 = LoadArgs.load_image_color(paths[i * 2 + 1], w, h);

                d.X.Vals[i] = new float[d.X.Cols];
                memcpy(d.X.Vals[i], im1.Data, h * w * 3 * sizeof(float));
                memcpy(d.X.Vals[i] + h * w * 3, im2.Data, h * w * 3 * sizeof(float));

                int id;
                float iou;

                char imlabel1[4096];
                char imlabel2[4096];
                find_replace(paths[i * 2], "imgs", "labels", imlabel1);
                find_replace(imlabel1, "jpg", "txt", imlabel1);
                FILE* fp1 = fopen(imlabel1, "r");

                while (fscanf(fp1, "%d %f", &id, &iou) == 2)
                {
                    if (d.Y.Vals[i][2 * id] < iou) d.Y.Vals[i][2 * id] = iou;
                }

                find_replace(paths[i * 2 + 1], "imgs", "labels", imlabel2);
                find_replace(imlabel2, "jpg", "txt", imlabel2);
                FILE* fp2 = fopen(imlabel2, "r");

                while (fscanf(fp2, "%d %f", &id, &iou) == 2)
                {
                    if (d.Y.Vals[i][2 * id + 1] < iou) d.Y.Vals[i][2 * id + 1] = iou;
                }

                for (j = 0; j < classes; ++j)
                {
                    if (d.Y.Vals[i][2 * j] > .5 && d.Y.Vals[i][2 * j + 1] < .5)
                    {
                        d.Y.Vals[i][2 * j] = 1;
                        d.Y.Vals[i][2 * j + 1] = 0;
                    }
                    else if (d.Y.Vals[i][2 * j] < .5 && d.Y.Vals[i][2 * j + 1] > .5)
                    {
                        d.Y.Vals[i][2 * j] = 0;
                        d.Y.Vals[i][2 * j + 1] = 1;
                    }
                    else
                    {
                        d.Y.Vals[i][2 * j] = SECRET_NUM;
                        d.Y.Vals[i][2 * j + 1] = SECRET_NUM;
                    }
                }
                fclose(fp1);
                fclose(fp2);

                free_image(im1);
                free_image(im2);
            }
            if (m) free(paths);
            return d;
        }

        Data load_data_swag(string[] paths, int n, int classes, float jitter)
        {
            int index = random_gen() % n;
            string random_path = paths[index];

            Image orig = LoadArgs.load_image_color(random_path, 0, 0);
            int h = orig.H;
            int w = orig.W;

            Data d = { 0 };
            d.Shallow = 0;
            d.W = w;
            d.H = h;

            d.X.Rows = 1;
            d.X.Vals = (float[] *)calloc(d.X.Rows, sizeof(float[]));
            d.X.Cols = h * w * 3;

            int k = (4 + classes) * 30;
            d.Y = new Matrix(1, k);

            int dw = w * jitter;
            int dh = h * jitter;

            int pleft = (int)Utils.rand_uniform(-dw, dw);
            int pright = (int)Utils.rand_uniform(-dw, dw);
            int ptop = (int)Utils.rand_uniform(-dh, dh);
            int pbot = (int)Utils.rand_uniform(-dh, dh);

            int swidth = w - pleft - pright;
            int sheight = h - ptop - pbot;

            float sx = (float)swidth / w;
            float sy = (float)sheight / h;

            int flip = random_gen() % 2;
            Image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

            float dx = ((float)pleft / w) / sx;
            float dy = ((float)ptop / h) / sy;

            Image sized = resize_image(cropped, w, h);
            if (flip) flip_image(sized);
            d.X.Vals[0] = sized.Data;

            fill_truth_swag(random_path, d.Y.Vals[0], classes, flip, dx, dy, 1./ sx, 1./ sy);

            free_image(orig);
            free_image(cropped);

            return d;
        }

        Data load_data_detection(int n, string[] paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
        {
            string[] random_paths = get_random_paths(paths, n, m);
            int i;
            Data d = { 0 };
            d.Shallow = 0;

            d.X.Rows = n;
            d.X.Vals = (float[] *)calloc(d.X.Rows, sizeof(float[]));
            d.X.Cols = h * w * 3;

            d.Y = new Matrix(n, 5 * boxes);
            for (i = 0; i < n; ++i)
            {
                Image orig = LoadArgs.load_image_color(random_paths[i], 0, 0);

                int oh = orig.H;
                int ow = orig.W;

                int dw = (ow * jitter);
                int dh = (oh * jitter);

                int pleft = (int)Utils.rand_uniform(-dw, dw);
                int pright = (int)Utils.rand_uniform(-dw, dw);
                int ptop = (int)Utils.rand_uniform(-dh, dh);
                int pbot = (int)Utils.rand_uniform(-dh, dh);

                int swidth = ow - pleft - pright;
                int sheight = oh - ptop - pbot;

                float sx = (float)swidth / ow;
                float sy = (float)sheight / oh;

                int flip = random_gen() % 2;
                Image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

                float dx = ((float)pleft / ow) / sx;
                float dy = ((float)ptop / oh) / sy;

                Image sized = resize_image(cropped, w, h);
                if (flip) flip_image(sized);
                random_distort_image(sized, hue, saturation, exposure);
                d.X.Vals[i] = sized.Data;

                fill_truth_detection(random_paths[i], boxes, d.Y.Vals[i], classes, flip, dx, dy, 1./ sx, 1./ sy);

                free_image(orig);
                free_image(cropped);
            }
            free(random_paths);
            return d;
        }


        void* load_thread(void* ptr)
        {
            srand(time(0));
            //Console.Write("Loading Data: %d\n", random_gen());
            load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.W, a.H);
    } else if (a.type == CLASSIFICATION_DATA){
        * a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
} else if (a.type == SUPER_DATA){
        * a.d = load_data_super(a.paths, a.n, a.m, a.W, a.H, a.scale);
    } else if (a.type == WRITING_DATA){
        * a.d = load_data_writing(a.paths, a.n, a.m, a.W, a.H, a.out_w, a.out_h);
    } else if (a.type == REGION_DATA){
        * a.d = load_data_region(a.n, a.paths, a.m, a.W, a.H, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){
        * a.d = load_data_detection(a.n, a.paths, a.m, a.W, a.H, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == SWAG_DATA){
        * a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        * a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.W, a.H);
    } else if (a.type == IMAGE_DATA){
        * (a.im) = LoadArgs.load_image_color(a.path, 0, 0);
        * (a.resized) = resize_image(*(a.im), a.W, a.H);
    } else if (a.type == TAG_DATA){
        * a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

Thread load_data_in_thread(load_args args)
{
    Thread thread;
    struct load_args * ptr = (load_args*)calloc(1, sizeof(struct load_args));
    * ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

void* load_threads(void* ptr)
{
    srand(time(0));
    int i;
    load_args args = *(load_args*)ptr;
    if (args.threads == 0) args.threads = 1;
    Data *out = args.d;
    int total = args.n;
    free(ptr);
    Data* buffers = (Data*)calloc(args.threads, sizeof(Data));
    Thread* threads = (Thread*)calloc(args.threads, sizeof(Thread));
    for (i = 0; i < args.threads; ++i)
    {
        args.d = buffers + i;
        args.n = (i + 1) * total / args.threads - i * total / args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for (i = 0; i < args.threads; ++i)
    {
        pthread_join(threads[i], 0);
    }
    *out = concat_datas(buffers, args.threads);
    out.shallow = 0;
    for (i = 0; i < args.threads; ++i)
    {
        buffers[i].Shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}

Thread load_data(load_args args)
{
    Thread thread;
    struct load_args * ptr = (load_args*)calloc(1, sizeof(struct load_args));
    * ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

Data load_data_writing(string[] paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if (m) paths = get_random_paths(paths, n, m);
    string[] replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    Data d = { 0 };
    d.Shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.Y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if (m) free(paths);
    int i;
    for (i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

Data load_data_old(string[] paths, int n, int m, string[] labels, int k, int w, int h)
{
    if (m) paths = get_random_paths(paths, n, m);
    Data d = { 0 };
    d.Shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.Y = load_labels_paths(paths, n, labels, k, 0);
    if (m) free(paths);
    return d;
}

Data load_data_super(string[] paths, int n, int m, int w, int h, int scale)
{
    if (m) paths = get_random_paths(paths, n, m);
    Data d = { 0 };
    d.Shallow = 0;

    int i;
    d.X.Rows = n;
    d.X.Vals = (float[] *)calloc(n, sizeof(float[]));
    d.X.Cols = w * h * 3;

    d.Y.Rows = n;
    d.Y.Vals = (float[] *)calloc(n, sizeof(float[]));
    d.Y.Cols = w * scale * h * scale * 3;

    for (i = 0; i < n; ++i)
    {
        Image im = LoadArgs.load_image_color(paths[i], 0, 0);
        Image crop = random_crop_image(im, w * scale, h * scale);
        int flip = random_gen() % 2;
        if (flip) flip_image(crop);
        Image resize = resize_image(crop, w, h);
        d.X.Vals[i] = resize.Data;
        d.Y.Vals[i] = crop.Data;
        free_image(im);
    }

    if (m) free(paths);
    return d;
}

Data load_data_augment(string[] paths, int n, int m, string[] labels, int k, Tree* hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if (m) paths = get_random_paths(paths, n, m);
    Data d = { 0 };
    d.Shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
    d.Y = load_labels_paths(paths, n, labels, k, hierarchy);
    if (m) free(paths);
    return d;
}

Data load_data_tag(string[] paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if (m) paths = get_random_paths(paths, n, m);
    Data d = { 0 };
    d.W = size;
    d.H = size;
    d.Shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
    d.Y = load_tags_paths(paths, n, k);
    if (m) free(paths);
    return d;
}

Matrix concat_matrix(Matrix m1, Matrix m2)
{
    int i, count = 0;
    Matrix m;
    m.Cols = m1.Cols;
    m.Rows = m1.Rows + m2.Rows;
    m.Vals = (float[] *)calloc(m1.Rows + m2.Rows, sizeof(float[]));
    for (i = 0; i < m1.Rows; ++i)
    {
        m.Vals[count++] = m1.Vals[i];
    }
    for (i = 0; i < m2.Rows; ++i)
    {
        m.Vals[count++] = m2.Vals[i];
    }
    return m;
}

Data concat_data(Data d1, Data d2)
{
    Data d = { 0 };
    d.Shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.Y = concat_matrix(d1.Y, d2.Y);
    return d;
}

Data concat_datas(Data* d, int n)
{
    int i;
    Data out = { 0};
    for (i = 0; i < n; ++i)
    {
        Data snew = concat_data(d[i], out);
        free_data(out);
        out = snew;
    }
    return out;
}

Data load_categorical_data_csv(string filename, int target, int k)
{
    Data d = { 0 };
    d.Shallow = 0;
    Matrix X = csv_to_matrix(filename);
    float[] truth_1d = pop_column(&X, target);
    float[]*truth = one_hot_encode(truth_1d, X.Rows, k);
    Matrix y;
    y.Rows = X.Rows;
    y.Cols = k;
    y.Vals = truth;
    d.X = X;
    d.Y = y;
    free(truth_1d);
    return d;
}

Data load_cifar10_data(string filename)
{
    Data d = { 0 };
    d.Shallow = 0;
    long i, j;
    Matrix X = new Matrix(10000, 3072);
    Matrix y = new Matrix(10000, 10);
    d.X = X;
    d.Y = y;

    FILE* fp = fopen(filename, "rb");
    if (!fp) file_error(filename);
    for (i = 0; i < 10000; ++i)
    {
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int sclass = bytes[0];
        y.Vals[i][sclass] = 1;
        for (j = 0; j < X.Cols; ++j)
        {
            X.Vals[i][j] = (double)bytes[j + 1];
        }
    }
    //translate_data_rows(d, -128);
    scale_data_rows(d, 1./ 255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(Data d, int n, float[] X, float[] y)
{
    int j;
    for (j = 0; j < n; ++j)
    {
        int index = random_gen() % d.X.Rows;
        memcpy(X + j * d.X.Cols, d.X.Vals[index], d.X.Cols * sizeof(float));
        memcpy(y + j * d.Y.Cols, d.Y.Vals[index], d.Y.Cols * sizeof(float));
    }
}

void get_next_batch(Data d, int n, int offset, float[] X, float[] y)
{
    int j;
    for (j = 0; j < n; ++j)
    {
        int index = offset + j;
        memcpy(X + j * d.X.Cols, d.X.Vals[index], d.X.Cols * sizeof(float));
        memcpy(y + j * d.Y.Cols, d.Y.Vals[index], d.Y.Cols * sizeof(float));
    }
}

void smooth_data(Data d)
{
    int i, j;
    float scale = 1. / d.Y.Cols;
    float eps = .1;
    for (i = 0; i < d.Y.Rows; ++i)
    {
        for (j = 0; j < d.Y.Cols; ++j)
        {
            d.Y.Vals[i][j] = eps * scale + (1 - eps) * d.Y.Vals[i][j];
        }
    }
}

Data load_all_cifar10()
{
    Data d = { 0 };
    d.Shallow = 0;
    int i, j, b;
    Matrix X = new Matrix(50000, 3072);
    Matrix y = new Matrix(50000, 10);
    d.X = X;
    d.Y = y;


    for (b = 0; b < 5; ++b)
    {
        char buff[256];
        sprintf(buff, "Data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b + 1);
        FILE* fp = fopen(buff, "rb");
        if (!fp) file_error(buff);
        for (i = 0; i < 10000; ++i)
        {
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int sclass = bytes[0];
            y.Vals[i + b * 10000][sclass] = 1;
            for (j = 0; j < X.Cols; ++j)
            {
                X.Vals[i + b * 10000][j] = (double)bytes[j + 1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    //translate_data_rows(d, -128);
    scale_data_rows(d, 1./ 255);
    smooth_data(d);
    return d;
}

Data load_go(string filename)
{
    FILE* fp = fopen(filename, "rb");
    Matrix X = new Matrix(3363059, 361);
    Matrix y = new Matrix(3363059, 361);
    int row, col;

    if (!fp) file_error(filename);
    string label;
    int count = 0;
    while ((label = fgetl(fp)))
    {
        int i;
        if (count == X.Rows)
        {
            X = resize_matrix(X, count * 2);
            y = resize_matrix(y, count * 2);
        }
        sscanf(label, "%d %d", &row, &col);
        string board = fgetl(fp);

        int index = row * 19 + col;
        y.Vals[count][index] = 1;

        for (i = 0; i < 19 * 19; ++i)
        {
            float val = 0;
            if (board[i] == '1') val = 1;
            else if (board[i] == '2') val = -1;
            X.Vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    Data d = { 0 };
    d.Shallow = 0;
    d.X = X;
    d.Y = y;


    fclose(fp);

    return d;
}


void randomize_data(Data d)
{
    int i;
    for (i = d.X.Rows - 1; i > 0; --i)
    {
        int index = random_gen() % i;
        float[] swap = d.X.Vals[index];
        d.X.Vals[index] = d.X.Vals[i];
        d.X.Vals[i] = swap;

        swap = d.Y.Vals[index];
        d.Y.Vals[index] = d.Y.Vals[i];
        d.Y.Vals[i] = swap;
    }
}

void scale_data_rows(Data d, float s)
{
    int i;
    for (i = 0; i < d.X.Rows; ++i)
    {
        scale_array(d.X.Vals[i], d.X.Cols, s);
    }
}

void translate_data_rows(Data d, float s)
{
    int i;
    for (i = 0; i < d.X.Rows; ++i)
    {
        translate_array(d.X.Vals[i], d.X.Cols, s);
    }
}

void normalize_data_rows(Data d)
{
    int i;
    for (i = 0; i < d.X.Rows; ++i)
    {
        normalize_array(d.X.Vals[i], d.X.Cols);
    }
}

Data get_data_part(Data d, int part, int total)
{
    Data p = { 0 };
    p.Shallow = 1;
    p.X.Rows = d.X.Rows * (part + 1) / total - d.X.Rows * part / total;
    p.Y.Rows = d.Y.Rows * (part + 1) / total - d.Y.Rows * part / total;
    p.X.Cols = d.X.Cols;
    p.Y.Cols = d.Y.Cols;
    p.X.Vals = d.X.Vals + d.X.Rows * part / total;
    p.Y.Vals = d.Y.Vals + d.Y.Rows * part / total;
    return p;
}

Data get_random_data(Data d, int num)
{
    Data r = { 0 };
    r.Shallow = 1;

    r.X.Rows = num;
    r.Y.Rows = num;

    r.X.Cols = d.X.Cols;
    r.Y.Cols = d.Y.Cols;

    r.X.Vals = (float[] *)calloc(num, sizeof(float[]));
    r.Y.Vals = (float[] *)calloc(num, sizeof(float[]));

    int i;
    for (i = 0; i < num; ++i)
    {
        int index = random_gen() % d.X.Rows;
        r.X.Vals[i] = d.X.Vals[index];
        r.Y.Vals[i] = d.Y.Vals[index];
    }
    return r;
}

Data* split_data(Data d, int part, int total)
{
    Data* split = (Data*)calloc(2, sizeof(Data));
    int i;
    int start = part * d.X.Rows / total;
    int end = (part + 1) * d.X.Rows / total;
    Data train;
    Data test;
    train.Shallow = test.Shallow = 1;

    test.X.Rows = test.Y.Rows = end - start;
    train.X.Rows = train.Y.Rows = d.X.Rows - (end - start);
    train.X.Cols = test.X.Cols = d.X.Cols;
    train.Y.Cols = test.Y.Cols = d.Y.Cols;

    train.X.Vals = (float[] *)calloc(train.X.Rows, sizeof(float[]));
    test.X.Vals = (float[] *)calloc(test.X.Rows, sizeof(float[]));
    train.Y.Vals = (float[] *)calloc(train.Y.Rows, sizeof(float[]));
    test.Y.Vals = (float[] *)calloc(test.Y.Rows, sizeof(float[]));

    for (i = 0; i < start; ++i)
    {
        train.X.Vals[i] = d.X.Vals[i];
        train.Y.Vals[i] = d.Y.Vals[i];
    }
    for (i = start; i < end; ++i)
    {
        test.X.Vals[i - start] = d.X.Vals[i];
        test.Y.Vals[i - start] = d.Y.Vals[i];
    }
    for (i = end; i < d.X.Rows; ++i)
    {
        train.X.Vals[i - (end - start)] = d.X.Vals[i];
        train.Y.Vals[i - (end - start)] = d.Y.Vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

    }
}