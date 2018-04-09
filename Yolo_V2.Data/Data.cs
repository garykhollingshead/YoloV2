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
        private int W;
        private int H;
        public Matrix X;
        public Matrix Y;
        private int Shallow;
        private List<int> NumBoxes;

        private static readonly object mutexLock = new object();

        public Data()
        {
            NumBoxes = new List<int>();
            X = new Matrix();
            Y = new Matrix();
        }

        public static string[] GetPaths(string filename)
        {
            return !File.Exists(filename)
                ? new string[0]
                : File.ReadAllLines(filename);
        }

        private static string[] get_random_paths(string[] paths, int n, int m)
        {
            string[] randomPaths = new string[n];
            int i;
            lock (mutexLock)
            {
                for (i = 0; i < n; ++i)
                {
                    int index = Utils.Rand.Next() % m;
                    randomPaths[i] = paths[index];
                }
            }
            return randomPaths;
        }

        private static string[] find_replace_paths(string[] paths, int n, string find, string replace)
        {
            string[] replacePaths = new string[n];
            int i;
            for (i = 0; i < n; ++i)
            {
                replacePaths[i] = paths[i].Replace(find, replace);
            }

            return replacePaths;
        }

        private static Matrix load_image_paths_gray(string[] paths, int n, int w, int h)
        {
            int i;
            Matrix x = new Matrix(n);

            for (i = 0; i < n; ++i)
            {
                Image im = LoadArgs.load_image(paths[i], w, h, 3);

                Image gray = LoadArgs.grayscale_image(im);
                im = gray;

                x.Vals[i] = im.Data;
                x.Cols = im.Height * im.Width * im.NumberOfChannels;
            }
            return x;
        }

        private static Matrix load_image_paths(string[] paths, int n, int w, int h)
        {
            int i;
            Matrix x = new Matrix(n);

            for (i = 0; i < n; ++i)
            {
                Image im = LoadArgs.load_image_color(paths[i], w, h);
                x.Vals[i] = im.Data;
                x.Cols = im.Height * im.Width * im.NumberOfChannels;
            }
            return x;
        }

        private static Matrix load_image_augment_paths(string[] paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
        {
            int i;
            Matrix x = new Matrix(n);

            for (i = 0; i < n; ++i)
            {
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                Image crop = LoadArgs.random_augment_image(im, angle, aspect, min, max, size);
                int flip = Utils.Rand.Next() % 2;
                if (flip != 0) LoadArgs.flip_image(ref crop);
                LoadArgs.random_distort_image(crop, hue, saturation, exposure);

                x.Vals[i] = crop.Data;
                x.Cols = crop.Height * crop.Width * crop.NumberOfChannels;
            }
            return x;
        }

        public static BoxLabel[] read_boxes(string filename, ref int n)
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
                var numbers = lines[i].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
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

        private static void randomize_boxes(BoxLabel[] b, int n)
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

        private static void correct_boxes(BoxLabel[] boxes, int n, float dx, float dy, float sx, float sy, int flip)
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

        private static void fill_truth_swag(string path, float[] truth, int classes, int flip, float dx, float dy, float sx, float sy)
        {
            string labelpath = path.Replace("images", "labels");
            labelpath = labelpath.Replace("JPEGImages", "labels");
            labelpath = labelpath.Replace(".jpg", ".txt");
            labelpath = labelpath.Replace(".JPG", ".txt");
            labelpath = labelpath.Replace(".JPEG", ".txt");

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

        private static void fill_truth_region(string path, float[] truth, int classes, int numBoxes, int flip, float dx, float dy, float sx, float sy)
        {
            string labelpath = path.Replace("images", "labels");
            labelpath = labelpath.Replace("JPEGImages", "labels");
            labelpath = labelpath.Replace(".jpg", ".txt");
            labelpath = labelpath.Replace(".JPG", ".txt");
            labelpath = labelpath.Replace(".JPEG", ".txt");
            labelpath = labelpath.Replace(".png", ".txt");
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

                int col = (int)(x * numBoxes);
                int row = (int)(y * numBoxes);

                x = x * numBoxes - col;
                y = y * numBoxes - row;

                int index = (col + row * numBoxes) * (5 + classes);
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

        private static void fill_truth_detection(string path, int numBoxes, float[] truth, int classes, int flip, float dx, float dy, float sx, float sy)
        {
            string labelpath = path.Replace("images", "labels");
            labelpath = labelpath.Replace("JPEGImages", "labels");
            labelpath = labelpath.Replace(".jpg", ".txt");
            labelpath = labelpath.Replace(".JPG", ".txt");
            labelpath = labelpath.Replace(".JPEG", ".txt");
            labelpath = labelpath.Replace(".raw", ".txt");
            labelpath = labelpath.Replace(".png", ".txt");
            int count = 0;
            BoxLabel[] boxes = read_boxes(labelpath, ref count);
            randomize_boxes(boxes, count);
            correct_boxes(boxes, count, dx, dy, sx, sy, flip);
            if (count > numBoxes) count = numBoxes;
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

        private static void fill_truth(string path, string[] labels, int k, float[] truth)
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

        private static void fill_hierarchy(float[] truth, int k, Tree hierarchy)
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
                        truth[count + i] = Utils.SecretNum;
                    }
                }
                count += hierarchy.GroupSize[j];
            }
        }

        private static Matrix load_labels_paths(string[] paths, int n, string[] labels, int k, Tree hierarchy)
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

        private static Matrix load_tags_paths(string[] paths, int n, int k)
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

        public static string[] get_labels(string filename)
        {
            return GetPaths(filename).ToArray();
        }

        private static Data load_data_region(int n, string[] paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
        {
            string[] randomPaths = get_random_paths(paths, n, m);
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
                Image orig = LoadArgs.load_image_color(randomPaths[i], 0, 0);

                int oh = orig.Height;
                int ow = orig.Width;

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
                Image cropped = LoadArgs.crop_image(orig, pleft, ptop, swidth, sheight);

                float dx = ((float)pleft / ow) / sx;
                float dy = ((float)ptop / oh) / sy;

                Image sized = LoadArgs.resize_image(cropped, w, h);
                if (flip != 0) LoadArgs.flip_image(ref sized);
                LoadArgs.random_distort_image(sized, hue, saturation, exposure);
                d.X.Vals[i] = sized.Data;

                fill_truth_region(randomPaths[i], d.Y.Vals[i], classes, size, flip, dx, dy, 1.0f / sx, 1.0f / sy);
            }
            return d;
        }

        private static Data load_data_compare(int n, string[] paths, int m, int classes, int w, int h)
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
                Array.Copy(im1.Data, d.X.Vals[i], im1.Data.Length);
                Array.Copy(im2.Data, 0, d.X.Vals[i], im1.Data.Length, im2.Data.Length);

                int id;
                float iou;

                string imlabel1 = paths[i * 2].Replace("imgs", "labels");
                imlabel1 = imlabel1.Replace("jpg", "txt");

                var lines1 = File.ReadAllLines(imlabel1);
                foreach (var line in lines1)
                {
                    var parts = line.Split(' ');
                    if (parts.Length != 2)
                    {
                        continue;
                    }
                    id = int.Parse(parts[0]);
                    iou = float.Parse(parts[1]);
                    if (d.Y.Vals[i][2 * id] < iou)
                    {
                        d.Y.Vals[i][2 * id] = iou;
                    }
                }

                string imlabel2 = paths[i * 2 + 1].Replace("imgs", "labels");
                imlabel2 = imlabel2.Replace("jpg", "txt");

                var lines2 = File.ReadAllLines(imlabel2);
                foreach (var line in lines2)
                {
                    var parts = line.Split(' ');
                    if (parts.Length != 2)
                    {
                        continue;
                    }
                    id = int.Parse(parts[0]);
                    iou = float.Parse(parts[1]);

                    if (d.Y.Vals[i][2 * id + 1] < iou)
                    {
                        d.Y.Vals[i][2 * id + 1] = iou;
                    }
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
                        d.Y.Vals[i][2 * j] = Utils.SecretNum;
                        d.Y.Vals[i][2 * j + 1] = Utils.SecretNum;
                    }
                }
            }
            return d;
        }

        private static Data load_data_swag(string[] paths, int n, int classes, float jitter)
        {
            int index = Utils.Rand.Next() % n;
            string randomPath = paths[index];

            Image orig = LoadArgs.load_image_color(randomPath, 0, 0);
            int h = orig.Height;
            int w = orig.Width;

            Data d = new Data();
            d.Shallow = 0;
            d.W = w;
            d.H = h;

            d.X.Rows = 1;
            d.X.Vals = new float[d.X.Rows][];
            d.X.Cols = h * w * 3;

            int k = (4 + classes) * 30;
            d.Y = new Matrix(1, k);

            int dw = (int)(w * jitter);
            int dh = (int)(h * jitter);

            int pleft = (int)Utils.rand_uniform(-dw, dw);
            int pright = (int)Utils.rand_uniform(-dw, dw);
            int ptop = (int)Utils.rand_uniform(-dh, dh);
            int pbot = (int)Utils.rand_uniform(-dh, dh);

            int swidth = w - pleft - pright;
            int sheight = h - ptop - pbot;

            float sx = (float)swidth / w;
            float sy = (float)sheight / h;

            int flip = Utils.Rand.Next() % 2;
            Image cropped = LoadArgs.crop_image(orig, pleft, ptop, swidth, sheight);

            float dx = ((float)pleft / w) / sx;
            float dy = ((float)ptop / h) / sy;

            Image sized = LoadArgs.resize_image(cropped, w, h);
            if (flip != 0) LoadArgs.flip_image(ref sized);
            d.X.Vals[0] = sized.Data;

            fill_truth_swag(randomPath, d.Y.Vals[0], classes, flip, dx, dy, 1.0f / sx, 1.0f / sy);

            return d;
        }

        private static Data load_data_detection(int n, string[] paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
        {
            string[] randomPaths = get_random_paths(paths, n, m);
            int i;
            Data d = new Data();
            d.Shallow = 0;

            d.X.Rows = n;
            d.X.Vals = new float[d.X.Rows][];
            d.X.Cols = h * w * 3;

            d.Y = new Matrix(n, 5 * boxes);
            for (i = 0; i < n; ++i)
            {
                Image orig = LoadArgs.load_image_color(randomPaths[i], 0, 0);

                int oh = orig.Height;
                int ow = orig.Width;

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
                Image cropped = LoadArgs.crop_image(orig, pleft, ptop, swidth, sheight);

                float dx = ((float)pleft / ow) / sx;
                float dy = ((float)ptop / oh) / sy;

                Image sized = LoadArgs.resize_image(cropped, w, h);
                if (flip != 0) LoadArgs.flip_image(ref sized);
                LoadArgs.random_distort_image(sized, hue, saturation, exposure);
                d.X.Vals[i] = sized.Data;

                fill_truth_detection(randomPaths[i], boxes, d.Y.Vals[i], classes, flip, dx, dy, 1.0f / sx, 1.0f / sy);
            }
            return d;
        }

        private static void load_thread(LoadArgs ptr)
        {
            if (ptr.Exposure == 0) ptr.Exposure = 1;
            if (ptr.Saturation == 0) ptr.Saturation = 1;
            if (ptr.Aspect == 0) ptr.Aspect = 1;

            if (ptr.Type == DataType.OldClassificationData)
            {
                ptr.D = load_data_old(ptr.Paths, ptr.N, ptr.M, ptr.Labels, ptr.Classes, ptr.W, ptr.H);
            }
            else if (ptr.Type == DataType.ClassificationData)
            {
                ptr.D = load_data_augment(ptr.Paths, ptr.N, ptr.M, ptr.Labels, ptr.Classes, ptr.Hierarchy, ptr.Min, ptr.Max, ptr.Size, ptr.Angle, ptr.Aspect, ptr.Hue, ptr.Saturation, ptr.Exposure);
            }
            else if (ptr.Type == DataType.SuperData)
            {
                ptr.D = load_data_super(ptr.Paths, ptr.N, ptr.M, ptr.W, ptr.H, ptr.Scale);
            }
            else if (ptr.Type == DataType.WritingData)
            {
                ptr.D = load_data_writing(ptr.Paths, ptr.N, ptr.M, ptr.W, ptr.H, ptr.OutW, ptr.OutH);
            }
            else if (ptr.Type == DataType.RegionData)
            {
                ptr.D = load_data_region(ptr.N, ptr.Paths, ptr.M, ptr.W, ptr.H, ptr.NumBoxes, ptr.Classes, ptr.Jitter, ptr.Hue, ptr.Saturation, ptr.Exposure);
            }
            else if (ptr.Type == DataType.DetectionData)
            {
                ptr.D = load_data_detection(ptr.N, ptr.Paths, ptr.M, ptr.W, ptr.H, ptr.NumBoxes, ptr.Classes, ptr.Jitter, ptr.Hue, ptr.Saturation, ptr.Exposure);
            }
            else if (ptr.Type == DataType.SwagData)
            {
                ptr.D = load_data_swag(ptr.Paths, ptr.N, ptr.Classes, ptr.Jitter);
            }
            else if (ptr.Type == DataType.CompareData)
            {
                ptr.D = load_data_compare(ptr.N, ptr.Paths, ptr.M, ptr.Classes, ptr.W, ptr.H);
            }
            else if (ptr.Type == DataType.ImageData)
            {
                ptr.Im = LoadArgs.load_image_color(ptr.Path, 0, 0);
                ptr.Resized = LoadArgs.resize_image((ptr.Im), ptr.W, ptr.H);
            }
            else if (ptr.Type == DataType.TagData)
            {
                ptr.D = load_data_tag(ptr.Paths, ptr.N, ptr.M, ptr.Classes, ptr.Min, ptr.Max, ptr.Size, ptr.Angle, ptr.Aspect, ptr.Hue, ptr.Saturation, ptr.Exposure);
            }
        }

        public static Thread load_data_in_thread(LoadArgs args)
        {
            Thread thread = new Thread(() => load_thread(args));
            thread.Start();
            return thread;
        }

        private static void load_threads(LoadArgs ptr)
        {
            int i;
            if (ptr.Threads == 0) ptr.Threads = 1;
            Data output = ptr.D;
            int total = ptr.N;
            Data[] buffers = new Data[ptr.Threads];
            Thread[] threads = new Thread[ptr.Threads];
            for (i = 0; i < ptr.Threads; ++i)
            {
                ptr.D = buffers[i];
                ptr.N = (i + 1) * total / ptr.Threads - i * total / ptr.Threads;
                threads[i] = load_data_in_thread(ptr);
            }
            for (i = 0; i < ptr.Threads; ++i)
            {
                threads[i].Join();
            }
            output = concat_datas(buffers, ptr.Threads);
            output.Shallow = 0;
            for (i = 0; i < ptr.Threads; ++i)
            {
                buffers[i].Shallow = 1;
            }
        }

        public static Thread load_data(LoadArgs args)
        {
            Thread thread = new Thread(() => load_threads(args));
            return thread;
        }

        private static Data load_data_writing(string[] paths, int n, int m, int w, int h, int outW, int outH)
        {
            if (m != 0) paths = get_random_paths(paths, n, m);
            string[] replacePaths = find_replace_paths(paths, n, ".png", "-label.png");
            Data d = new Data();
            d.Shallow = 0;
            d.X = load_image_paths(paths, n, w, h);
            d.Y = load_image_paths_gray(replacePaths, n, outW, outH);
            return d;
        }

        private static Data load_data_old(string[] paths, int n, int m, string[] labels, int k, int w, int h)
        {
            if (m != 0) paths = get_random_paths(paths, n, m);
            Data d = new Data();
            d.Shallow = 0;
            d.X = load_image_paths(paths, n, w, h);
            d.Y = load_labels_paths(paths, n, labels, k, null);
            return d;
        }

        private static Data load_data_super(string[] paths, int n, int m, int w, int h, int scale)
        {
            if (m != 0) paths = get_random_paths(paths, n, m);
            Data d = new Data();
            d.Shallow = 0;

            int i;
            d.X.Rows = n;
            d.X.Vals = new float[n][];
            d.X.Cols = w * h * 3;

            d.Y.Rows = n;
            d.Y.Vals = new float[n][];
            d.Y.Cols = w * scale * h * scale * 3;

            for (i = 0; i < n; ++i)
            {
                Image im = LoadArgs.load_image_color(paths[i], 0, 0);
                Image crop = LoadArgs.random_crop_image(im, w * scale, h * scale);
                int flip = Utils.Rand.Next() % 2;
                if (flip != 0) LoadArgs.flip_image(ref crop);
                Image resize = LoadArgs.resize_image(crop, w, h);
                d.X.Vals[i] = resize.Data;
                d.Y.Vals[i] = crop.Data;
            }
            return d;
        }

        private static Data load_data_augment(string[] paths, int n, int m, string[] labels, int k, Tree hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
        {
            if (m != 0)
            {
                paths = get_random_paths(paths, n, m);
            }
            Data d = new Data();
            d.Shallow = 0;
            d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
            d.Y = load_labels_paths(paths, n, labels, k, hierarchy);
            return d;
        }

        private static Data load_data_tag(string[] paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
        {
            if (m != 0) paths = get_random_paths(paths, n, m);
            Data d = new Data();
            d.W = size;
            d.H = size;
            d.Shallow = 0;
            d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
            d.Y = load_tags_paths(paths, n, k);
            return d;
        }

        private static Matrix concat_matrix(Matrix m1, Matrix m2)
        {
            int i, count = 0;
            Matrix m = new Matrix();
            m.Cols = m1.Cols;
            m.Rows = m1.Rows + m2.Rows;
            m.Vals = new float[m1.Rows + m2.Rows][];
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

        private static Data concat_data(Data d1, Data d2)
        {
            Data d = new Data();
            d.Shallow = 1;
            d.X = concat_matrix(d1.X, d2.X);
            d.Y = concat_matrix(d1.Y, d2.Y);
            return d;
        }

        private static Data concat_datas(Data[] d, int n)
        {
            int i;
            Data output = new Data();
            for (i = 0; i < n; ++i)
            {
                Data snew = concat_data(d[i], output);
                output = snew;
            }
            return output;
        }

        public static Data load_cifar10_data(string filename)
        {
            Data d = new Data();
            d.Shallow = 0;
            long i, j;
            Matrix x = new Matrix(10000, 3072);
            Matrix y = new Matrix(10000, 10);
            d.X = x;
            d.Y = y;

            if (!File.Exists(filename)) Utils.file_error(filename);
            using (var stream = File.OpenRead(filename))
            {
                for (i = 0; i < 10000; ++i)
                {
                    var bytes = new byte[3073];
                    stream.Read(bytes, 0, bytes.Length);
                    int sclass = bytes[0];
                    y.Vals[i][sclass] = 1;
                    for (j = 0; j < x.Cols; ++j)
                    {
                        x.Vals[i][j] = bytes[j + 1];
                    }
                }
            }
            d.scale_data_rows( 1.0f / 255);
            return d;
        }

        public void get_random_batch(int n, float[] x, float[] y)
        {
            int j;
            for (j = 0; j < n; ++j)
            {
                int index = Utils.Rand.Next() % X.Rows;
                for (var i = 0; i < X.Cols && i < Y.Cols; ++i)
                {
                    if (i < X.Cols)
                    {
                        x[j * X.Cols + i] = X.Vals[index][i];
                    }

                    if (i < Y.Cols)
                    {
                        y[j * Y.Cols + i] = Y.Vals[index][i];
                    }
                }
            }
        }

        public void get_next_batch(int n, int offset, float[] x, float[] y)
        {
            int j;
            for (j = 0; j < n; ++j)
            {
                int index = offset + j;
                for (var i = 0; i < X.Cols && i < Y.Cols; ++i)
                {
                    if (i < X.Cols)
                    {
                        x[j * X.Cols + i] = X.Vals[index][i];
                    }

                    if (i < Y.Cols)
                    {
                        y[j * Y.Cols + i] = Y.Vals[index][i];
                    }
                }
            }
        }

        private void smooth_data()
        {
            int i, j;
            float scale = 1.0f / Y.Cols;
            float eps = .1f;
            for (i = 0; i < Y.Rows; ++i)
            {
                for (j = 0; j < Y.Cols; ++j)
                {
                    Y.Vals[i][j] = eps * scale + (1 - eps) * Y.Vals[i][j];
                }
            }
        }

        public static Data load_all_cifar10()
        {
            Data d = new Data();
            d.Shallow = 0;
            int i, j, b;
            Matrix x = new Matrix(50000, 3072);
            Matrix y = new Matrix(50000, 10);
            d.X = x;
            d.Y = y;


            for (b = 0; b < 5; ++b)
            {
                var buff = $"Data/cifar/cifar-10-batches-bin/data_batch_{b + 1}.bin";

                if (!File.Exists(buff)) Utils.file_error(buff);
                using (var fstream = File.OpenRead(buff))
                {
                    for (i = 0; i < 10000; ++i)
                    {
                        var bytes = new byte[3073];
                        fstream.Read(bytes, 0, bytes.Length);
                        int sclass = bytes[0];
                        y.Vals[i + b * 10000][sclass] = 1;
                        for (j = 0; j < x.Cols; ++j)
                        {
                            x.Vals[i + b * 10000][j] = bytes[j + 1];
                        }
                    }
                }
            }
            d.scale_data_rows(1.0f / 255);
            d.smooth_data();
            return d;
        }

        private void scale_data_rows(float s)
        {
            int i;
            for (i = 0; i < X.Rows; ++i)
            {
                Utils.scale_array(X.Vals[i], X.Cols, s);
            }
        }

        public static Data get_data_part(Data d, int part, int total)
        {
            Data p = new Data();
            var n = part / total;
            p.Shallow = 1;
            p.X.Rows = d.X.Rows * (part + 1) / total - d.X.Rows * n;
            p.Y.Rows = d.Y.Rows * (part + 1) / total - d.Y.Rows * n;
            p.X.Cols = d.X.Cols;
            p.Y.Cols = d.Y.Cols;
            p.X.Vals = new float[d.X.Vals.Length - n][];
            p.Y.Vals = new float[d.Y.Vals.Length - n][];
            for (int i = 0; i < p.X.Vals.Length && i < p.Y.Vals.Length; ++i)
            {
                if (i < p.X.Vals.Length)
                {
                    p.X.Vals[i] = d.X.Vals[i + n];
                }

                if (i < p.Y.Vals.Length)
                {
                    p.Y.Vals[i] = d.Y.Vals[i + n];
                }
            }
            return p;
        }
    }
}