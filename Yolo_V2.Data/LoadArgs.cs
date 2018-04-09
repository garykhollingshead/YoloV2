using System;
using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public class LoadArgs
    {
        public int Threads;
        public string[] Paths;
        public string Path;
        public int N;
        public int M;
        public string[] Labels;
        public int H;
        public int W;
        public int OutW;
        public int OutH;
        public int Nh;
        public int Nw;
        public int NumBoxes;
        public int Min, Max, Size;
        public int Classes;
        public int Background;
        public int Scale;
        public float Jitter;
        public float Angle;
        public float Aspect;
        public float Saturation;
        public float Exposure;
        public float Hue;
        public Data D;
        public Image Im;
        public Image Resized;
        public DataType Type;
        public Tree Hierarchy;

        private static int windows;
        private static VideoWriter outputVideo;

        private static float get_color(int c, int x, int max)
        {
            float[][] colors =
            {
                new float[]{ 1, 0, 1 },
                new float[] { 0, 0, 1 },
                new float[] { 0, 1, 1 },
                new float[] { 0, 1, 0 },
                new float[] { 1, 1, 0 },
                new float[] { 1, 0, 0 }
            };

            float ratio = ((float)x / max) * 5;
            int i = (int)Math.Floor(ratio);
            int j = (int)Math.Ceiling(ratio);
            ratio -= i;
            return (1 - ratio) * colors[i][c] + ratio * colors[j][c];
        }

        private static void composite_image(Image source, Image dest, int dx, int dy)
        {
            int x, y, k;
            for (k = 0; k < source.NumberOfChannels; ++k)
            {
                for (y = 0; y < source.Height; ++y)
                {
                    for (x = 0; x < source.Width; ++x)
                    {
                        float val = get_pixel(source, x, y, k);
                        float val2 = get_pixel_extend(dest, dx + x, dy + y, k);
                        set_pixel(dest, dx + x, dy + y, k, val * val2);
                    }
                }
            }
        }

        private static Image border_image(Image a, int border)
        {
            Image b = new Image(a.Width + 2 * border, a.Height + 2 * border, a.NumberOfChannels);
            int x, y, k;
            for (k = 0; k < b.NumberOfChannels; ++k)
            {
                for (y = 0; y < b.Height; ++y)
                {
                    for (x = 0; x < b.Width; ++x)
                    {
                        float val = get_pixel_extend(a, x - border, y - border, k);
                        if (x - border < 0 || x - border >= a.Width || y - border < 0 || y - border >= a.Height) val = 1;
                        set_pixel(b, x, y, k, val);
                    }
                }
            }

            return b;
        }

        private static Image tile_images(Image a, Image b, int dx)
        {
            if (a.Width == 0)
            {
                return new Image(b);
            }
            Image c = new Image(a.Width + b.Width + dx, (a.Height > b.Height) ? a.Height : b.Height, (a.NumberOfChannels > b.NumberOfChannels) ? a.NumberOfChannels : b.NumberOfChannels);
            Blas.Fill_cpu(c.Width * c.Height * c.NumberOfChannels, 1, c.Data, 1);
            embed_image(a, ref c, 0, 0);
            composite_image(b, c, a.Width + dx, 0);
            return c;
        }

        private static Image get_label(Image[][] characters, string lbl, int size)
        {
            if (size > 7) size = 7;
            Image label = new Image();
            foreach (var c in lbl)
            {
                Image l = characters[size][c];
                Image n = tile_images(label, l, -size - 1 + (size + 1) / 2);
                label = n;
            }

            return border_image(label, (int)(label.Height * .25));
        }

        public static void draw_label(ref Image a, int x, int y, string label, float blue, float green, float red)
        {
            using (var img = a.ToMat(true))
            {
                CvInvoke.PutText(img, label, new Point(x, y - 20), FontFace.HersheyPlain, 1.25,
                    new MCvScalar(blue, green, red), 2);
                a = new Image(img);
            }
        }

        public static void draw_box(ref Image a, int x, int y, int width, int height, int thickness, float blue, float green, float red)
        {
            using (var img = a.ToMat(true))
            {
                CvInvoke.Rectangle(img, new Rectangle(x, y, width, height), new MCvScalar(blue, green, red), thickness);
                a = new Image(img);
            }
        }

        public static void draw_detections(ref Image im, int num, float thresh, Box[] boxes, float[][] probs, string[] names, int classes)
        {
            int i;

            for (i = 0; i < num; ++i)
            {
                int curClass = Utils.max_index(probs[i], classes);
                float prob = probs[i][curClass];
                if (prob > thresh)
                {

                    int width = (int)(im.Height * .012);

                    Console.WriteLine($"{names[curClass]}: {prob:P}");
                    int offset = curClass * 123457 % classes;
                    float red = get_color(2, offset, classes) * 255;
                    float green = get_color(1, offset, classes) * 255;
                    float blue = get_color(0, offset, classes) * 255;
                    var rgb = new[]{
                        red, green, blue
                    };

                    Box b = boxes[i];

                    int left = (int)((b.X - b.W / 2f) * im.Width);
                    int right = (int)((b.X + b.W / 2f) * im.Width);
                    int top = (int)((b.Y - b.H / 2f) * im.Height);
                    int bot = (int)((b.Y + b.H / 2f) * im.Height);

                    if (left < 0) left = 0;
                    if (right > im.Width - 1) right = im.Width - 1;
                    if (top < 0) top = 0;
                    if (bot > im.Height - 1) bot = im.Height - 1;

                    draw_box(ref im, left, top, right - left, bot - top, width, blue, green, red);
                    draw_label(ref im, top, left, names[curClass], blue, green, red);
                }
            }
        }

        public static void flip_image(ref Image a)
        {
            using (var img = a.ToMat(true))
            {
                CvInvoke.Flip(img, img, FlipType.Vertical);
            }
        }

        private static void embed_image(Image source, ref Image dest, int dx, int dy)
        {
            using (var srcImg = source.ToMat(true))
            using (var roiImg = new Mat(dest.ToMat(true), new Rectangle(dx, dy, srcImg.Width, srcImg.Height)))
            {
                srcImg.CopyTo(roiImg);
                dest = new Image(roiImg);
                return;
            }
            int x, y, k;
            for (k = 0; k < source.NumberOfChannels; ++k)
            {
                for (y = 0; y < source.Height; ++y)
                {
                    for (x = 0; x < source.Width; ++x)
                    {
                        float val = get_pixel(source, x, y, k);
                        set_pixel(dest, dx + x, dy + y, k, val);
                    }
                }
            }
        }

        public static Image collapse_image_layers(Image source, int border)
        {
            int h = source.Height;
            h = (h + border) * source.NumberOfChannels - border;
            Image dest = new Image(source.Width, h, 1);
            int i;
            for (i = 0; i < source.NumberOfChannels; ++i)
            {
                Image layer = get_image_layer(source, i);
                int hOffset = i * (source.Height + border);
                embed_image(layer, ref dest, 0, hOffset);
            }

            return dest;
        }

        public static void constrain_image(Image im)
        {
            int i;
            for (i = 0; i < im.Width * im.Height * im.NumberOfChannels; ++i)
            {
                if (im.Data[i] < 0)
                    im.Data[i] = 0;
                if (im.Data[i] > 255)
                    im.Data[i] = 255;
            }
        }

        private static void normalize_image(Image p)
        {
            int i;
            float min = 9999999;
            float max = -999999;

            for (i = 0; i < p.Height * p.Width * p.NumberOfChannels; ++i)
            {
                float v = p.Data[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            if (max - min < .000000001)
            {
                min = 0;
                max = 1;
            }

            for (i = 0; i < p.NumberOfChannels * p.Width * p.Height; ++i)
            {
                p.Data[i] = (p.Data[i] - min) / (max - min);
            }
        }

        public static void rgbgr_image(Image im)
        {
            int i;
            for (i = 0; i < im.Width * im.Height; ++i)
            {
                float swap = im.Data[i];
                im.Data[i] = im.Data[i + im.Width * im.Height * 2];
                im.Data[i + im.Width * im.Height * 2] = swap;
            }
        }

        private static void show_image_cv(Image p, string name)
        {
            int x, y, k;
            Image copy = new Image(p);
            constrain_image(copy);
            //if (p.NumberOfChannels == 3) rgbgr_image(copy);

            string buff = name;
            CvInvoke.NamedWindow(buff, NamedWindowType.Normal);

            using (Mat disp = copy.ToMat(true))
            {
                CvInvoke.Imshow(buff, disp);
                CvInvoke.WaitKey(1);
                Size size = new Size(disp.Width, disp.Height);

                if (outputVideo == null)
                {
                    Console.WriteLine($"\n SRC output_video = {outputVideo} ");
                    string outputName = "test_dnn_out.avi";
                    outputVideo = new VideoWriter(outputName, 25, size, true);
                    Console.WriteLine($"\n cvCreateVideoWriter, DST output_video = {outputVideo} ");
                }

                outputVideo.Write(disp);
            }

            Console.WriteLine("\n cvWriteFrame \n");
        }

        public static void show_image(Image p, string name)
        {
            show_image_cv(p, name);
        }

        private static Image ipl_to_image(Mat src)
        {
            return new Image(src);
        }

        private static void System(string command)
        {
            var cmd = new Process
            {
                StartInfo =
                {
                    FileName = "cmd.exe",
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                    UseShellExecute = false
                }
            };


            cmd.Start();

            cmd.StandardInput.WriteLine(command);
            cmd.StandardInput.Flush();
            cmd.StandardInput.Close();
            Console.WriteLine(cmd.StandardOutput.ReadToEnd());
        }

        private static Image load_image_cv(string filename, int channels)
        {
            var flag = (channels == 1)
                ? ImreadModes.Grayscale
                : ImreadModes.Color;
            try
            {
                using (Mat src = new Mat(filename, flag))
                {
                    Image retImage = ipl_to_image(src);
                    //rgbgr_image(retImage);
                    return retImage;
                }
            }
            catch
            {
                Console.Error.WriteLine($"Cannot load Image {filename}");
                System($"echo {filename} >> bad.list");
                return new Image(10, 10, 3);
            }

        }

        public static Image get_image_from_stream(VideoCapture cap)
        {
            using (Mat src = cap.QueryFrame())
            {
                if (src.IsEmpty) return new Image();

                Image im = ipl_to_image(src);

                //rgbgr_image(im);
                return im;
            }
        }

        private static void save_image_jpg(Image p, string name)
        {

            using (var img = p.ToMat(true))
            {
                img.Save($"{name}.jpg");
            }
        }

        public static void save_image_png(Image im, string name)
        {
            using (var img = im.ToMat(true))
            {
                img.Save($"{name}.png");
                //CvInvoke.Imwrite($"{name}.png", img, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 5));
            }
        }

        public static void save_image(Image im, string name)
        {
            save_image_jpg(im, name);
        }

        public static Image make_random_image(int w, int h, int c)
        {
            Image retImage = new Image(w, h, c);
            int i;
            for (i = 0; i < w * h * c; ++i)
            {
                retImage.Data[i] = Utils.rand_normal() * .25f + .5f;
            }
            return retImage;
        }

        private static Image rotate_crop_image(Image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
        {
            int x, y, c;
            float cx = im.Width / 2.0f;
            float cy = im.Height / 2.0f;
            Image rot = new Image(w, h, im.NumberOfChannels);
            for (c = 0; c < im.NumberOfChannels; ++c)
            {
                for (y = 0; y < h; ++y)
                {
                    for (x = 0; x < w; ++x)
                    {
                        var rx = Math.Cos(rad) * ((x - w / 2.0) / s * aspect + dx / s * aspect) - Math.Sin(rad) * ((y - h / 2.0) / s + dy / s) + cx;
                        var ry = Math.Sin(rad) * ((x - w / 2.0) / s * aspect + dx / s * aspect) + Math.Cos(rad) * ((y - h / 2.0) / s + dy / s) + cy;
                        var val = bilinear_interpolate(im, (float)rx, (float)ry, c);
                        set_pixel(rot, x, y, c, val);
                    }
                }
            }
            return rot;
        }

        public static Image rotate_image(Image im, float rad)
        {
            int x, y, c;
            float cx = im.Width / 2.0f;
            float cy = im.Height / 2.0f;
            Image rot = new Image(im.Width, im.Height, im.NumberOfChannels);
            for (c = 0; c < im.NumberOfChannels; ++c)
            {
                for (y = 0; y < im.Height; ++y)
                {
                    for (x = 0; x < im.Width; ++x)
                    {
                        var rx = Math.Cos(rad) * (x - cx) - Math.Sin(rad) * (y - cy) + cx;
                        var ry = Math.Sin(rad) * (x - cx) + Math.Cos(rad) * (y - cy) + cy;
                        float val = bilinear_interpolate(im, (float)rx, (float)ry, c);
                        set_pixel(rot, x, y, c, val);
                    }
                }
            }
            return rot;
        }

        public static void scale_image(Image m, float s)
        {
            int i;
            for (i = 0; i < m.Height * m.Width * m.NumberOfChannels; ++i) m.Data[i] *= s;
        }

        public static Image crop_image(Image im, int dx, int dy, int w, int h)
        {
            Image cropped = new Image(w, h, im.NumberOfChannels);
            int i, j, k;
            for (k = 0; k < im.NumberOfChannels; ++k)
            {
                for (j = 0; j < h; ++j)
                {
                    for (i = 0; i < w; ++i)
                    {
                        int r = j + dy;
                        int c = i + dx;
                        float val = 0;
                        r = Utils.constrain_int(r, 0, im.Height - 1);
                        c = Utils.constrain_int(c, 0, im.Width - 1);
                        if (r >= 0 && r < im.Height && c >= 0 && c < im.Width)
                        {
                            val = get_pixel(im, c, r, k);
                        }
                        set_pixel(cropped, i, j, k, val);
                    }
                }
            }
            return cropped;
        }

        public static int best_3d_shift_r(Image a, Image b, int min, int max)
        {
            if (min == max)
            {
                return min;
            }
            int mid = (int)Math.Floor((min + max) / 2.0);
            Image c1 = crop_image(b, 0, mid, b.Width, b.Height);
            Image c2 = crop_image(b, 0, mid + 1, b.Width, b.Height);
            float d1 = Utils.dist_array(c1.Data, a.Data, a.Width * a.Height * a.NumberOfChannels, 10);
            float d2 = Utils.dist_array(c2.Data, a.Data, a.Width * a.Height * a.NumberOfChannels, 10);
            return d1 < d2
                ? best_3d_shift_r(a, b, min, mid)
                : best_3d_shift_r(a, b, mid + 1, max);
        }

        public static void composite_3d(string f1, string f2, string output, int delta)
        {
            if (string.IsNullOrEmpty(output))
            {
                output = "out";
            }
            Image a = load_image(f1, 0, 0, 0);
            Image b = load_image(f2, 0, 0, 0);
            int shift = best_3d_shift_r(a, b, -a.Height / 100, a.Height / 100);

            Image c1 = crop_image(b, 10, shift, b.Width, b.Height);
            float d1 = Utils.dist_array(c1.Data, a.Data, a.Width * a.Height * a.NumberOfChannels, 100);
            Image c2 = crop_image(b, -10, shift, b.Width, b.Height);
            float d2 = Utils.dist_array(c2.Data, a.Data, a.Width * a.Height * a.NumberOfChannels, 100);

            Console.WriteLine($"{shift}\n");

            Image c = crop_image(b, delta, shift, a.Width, a.Height);
            int i;
            for (i = 0; i < c.Width * c.Height; ++i)
            {
                c.Data[i] = a.Data[i];
            }
            save_image_jpg(c, output);
        }

        public static Image resize_min(Image im, int min)
        {
            int w = im.Width;
            int h = im.Height;
            if (w < h)
            {
                h = (h * min) / w;
                w = min;
            }
            else
            {
                w = (w * min) / h;
                h = min;
            }
            if (w == im.Width && h == im.Height) return im;
            Image resized = resize_image(im, w, h);
            return resized;
        }

        public static Image random_crop_image(Image im, int w, int h)
        {
            int dx = Utils.rand_int(0, im.Width - w);
            int dy = Utils.rand_int(0, im.Height - h);
            Image crop = crop_image(im, dx, dy, w, h);
            return crop;
        }

        public static Image random_augment_image(Image im, float angle, float aspect, int low, int high, int size)
        {
            aspect = Utils.rand_scale(aspect);
            int r = Utils.rand_int(low, high);
            int min = (im.Height < im.Width * aspect)
                ? im.Height
                : (int)(im.Width * aspect);
            float scale = (float)r / min;

            float rad = (float)(Utils.rand_uniform(-angle, angle) * Math.PI * 2 / 360.0f);

            float dx = (im.Width * scale / aspect - size) / 2.0f;
            float dy = (im.Height * scale - size) / 2.0f;
            if (dx < 0) dx = 0;
            if (dy < 0) dy = 0;
            dx = Utils.rand_uniform(-dx, dx);
            dy = Utils.rand_uniform(-dy, dy);

            Image crop = rotate_crop_image(im, rad, scale, size, size, dx, dy, aspect);

            return crop;
        }

        private static float three_way_max(float a, float b, float c)
        {
            return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
        }

        private static float three_way_min(float a, float b, float c)
        {
            return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
        }

        private static void rgb_to_hsv(Image im)
        {
            int i, j;
            float r, g, b;
            float h, s, v;
            for (j = 0; j < im.Height; ++j)
            {
                for (i = 0; i < im.Width; ++i)
                {
                    r = get_pixel(im, i, j, 0);
                    g = get_pixel(im, i, j, 1);
                    b = get_pixel(im, i, j, 2);
                    float max = three_way_max(r, g, b);
                    float min = three_way_min(r, g, b);
                    float delta = max - min;
                    v = max;
                    if (max == 0)
                    {
                        s = 0;
                        h = 0;
                    }
                    else
                    {
                        s = delta / max;
                        if (r == max)
                        {
                            h = (g - b) / delta;
                        }
                        else if (g == max)
                        {
                            h = 2 + (b - r) / delta;
                        }
                        else
                        {
                            h = 4 + (r - g) / delta;
                        }
                        if (h < 0) h += 6;
                        h = h / 6.0f;
                    }
                    set_pixel(im, i, j, 0, h);
                    set_pixel(im, i, j, 1, s);
                    set_pixel(im, i, j, 2, v);
                }
            }
        }

        private static void hsv_to_rgb(Image im)
        {
            int i, j;
            float r, g, b;
            float h, s, v;
            float f, p, q, t;
            for (j = 0; j < im.Height; ++j)
            {
                for (i = 0; i < im.Width; ++i)
                {
                    h = 6 * get_pixel(im, i, j, 0);
                    s = get_pixel(im, i, j, 1);
                    v = get_pixel(im, i, j, 2);
                    if (s == 0)
                    {
                        r = g = b = v;
                    }
                    else
                    {
                        int index = (int)Math.Floor(h);
                        f = h - index;
                        p = v * (1 - s);
                        q = v * (1 - s * f);
                        t = v * (1 - s * (1 - f));
                        if (index == 0)
                        {
                            r = v; g = t; b = p;
                        }
                        else if (index == 1)
                        {
                            r = q; g = v; b = p;
                        }
                        else if (index == 2)
                        {
                            r = p; g = v; b = t;
                        }
                        else if (index == 3)
                        {
                            r = p; g = q; b = v;
                        }
                        else if (index == 4)
                        {
                            r = t; g = p; b = v;
                        }
                        else
                        {
                            r = v; g = p; b = q;
                        }
                    }
                    set_pixel(im, i, j, 0, r);
                    set_pixel(im, i, j, 1, g);
                    set_pixel(im, i, j, 2, b);
                }
            }
        }

        public static Image grayscale_image(Image im)
        {
            int i, j, k;
            Image gray = new Image(im.Width, im.Height, 1);
            float[] scale = { 0.587f, 0.299f, 0.114f };
            for (k = 0; k < im.NumberOfChannels; ++k)
            {
                for (j = 0; j < im.Height; ++j)
                {
                    for (i = 0; i < im.Width; ++i)
                    {
                        gray.Data[i + im.Width * j] += scale[k] * get_pixel(im, i, j, k);
                    }
                }
            }
            return gray;
        }

        public static Image threshold_image(Image im, float thresh)
        {
            int i;
            Image t = new Image(im.Width, im.Height, im.NumberOfChannels);
            for (i = 0; i < im.Width * im.Height * im.NumberOfChannels; ++i)
            {
                t.Data[i] = im.Data[i] > thresh ? 1 : 0;
            }
            return t;
        }

        private static void scale_image_channel(Image im, int c, float v)
        {
            int i, j;
            for (j = 0; j < im.Height; ++j)
            {
                for (i = 0; i < im.Width; ++i)
                {
                    float pix = get_pixel(im, i, j, c);
                    pix = pix * v;
                    set_pixel(im, i, j, c, pix);
                }
            }
        }

        private static void distort_image(Image im, float hue, float sat, float val)
        {
            rgb_to_hsv(im);
            scale_image_channel(im, 1, sat);
            scale_image_channel(im, 2, val);
            int i;
            for (i = 0; i < im.Width * im.Height; ++i)
            {
                im.Data[i] = im.Data[i] + hue;
                if (im.Data[i] > 1) im.Data[i] -= 1;
                if (im.Data[i] < 0) im.Data[i] += 1;
            }
            hsv_to_rgb(im);
            constrain_image(im);
        }

        public static void random_distort_image(Image im, float hue, float saturation, float exposure)
        {
            float dhue = Utils.rand_uniform(-hue, hue);
            float dsat = Utils.rand_scale(saturation);
            float dexp = Utils.rand_scale(exposure);
            distort_image(im, dhue, dsat, dexp);
        }

        private static float bilinear_interpolate(Image im, float x, float y, int c)
        {
            int ix = (int)Math.Floor(x);
            int iy = (int)Math.Floor(y);

            float dx = x - ix;
            float dy = y - iy;

            float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
                dy * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
                (1 - dy) * dx * get_pixel_extend(im, ix + 1, iy, c) +
                dy * dx * get_pixel_extend(im, ix + 1, iy + 1, c);
            return val;
        }

        public static Image resize_image(Image im, int w, int h)
        {
            using (var img = im.ToMat(true))
            {
                CvInvoke.Resize(img, img, new Size(w, h));
                return new Image(img);
            }
        }

        public static void test_resize(string filename)
        {
            Image im = load_image(filename, 0, 0, 3);
            float mag = Utils.mag_array(im.Data, im.Width * im.Height * im.NumberOfChannels);
            Console.Write($"L2 Norm: {mag}\n");
            var gray = grayscale_image(im);

            var c1 = new Image(im);
            var c2 = new Image(im);
            var c3 = new Image(im);
            var c4 = new Image(im);
            distort_image(c1, .1f, 1.5f, 1.5f);
            distort_image(c2, -.1f, .66666f, .66666f);
            distort_image(c3, .1f, 1.5f, .66666f);
            distort_image(c4, .1f, .66666f, 1.5f);


            show_image(im, "Original");
            show_image(gray, "Gray");
            show_image(c1, "C1");
            show_image(c2, "C2");
            show_image(c3, "C3");
            show_image(c4, "C4");
            while (true)
            {
                var aug = random_augment_image(im, 0, .75f, 320, 448, 320);
                show_image(aug, "aug");


                float exposure = 1.15f;
                float saturation = 1.15f;
                float hue = .05f;

                var c = new Image(im);

                float dexp = Utils.rand_scale(exposure);
                float dsat = Utils.rand_scale(saturation);
                float dhue = Utils.rand_uniform(-hue, hue);

                distort_image(c, dhue, dsat, dexp);
                show_image(c, "rand");
                Console.Write($"{dhue} {dsat} {dexp}\n");
                CvInvoke.WaitKey();
            }
        }

        public static Image load_image(string filename, int w, int h, int c)
        {
            Image img = load_image_cv(filename, c);

            if (h != img.Height || w != img.Width)
            {
                img = resize_image(img, w, h);
            }
            return img;
        }

        public static Image load_image_color(string filename, int w, int h)
        {
            return load_image(filename, w, h, 3);
        }

        private static Image get_image_layer(Image m, int l)
        {
            Image img = new Image(m.Width, m.Height, 1);

            for (var i = 0; i < m.Height * m.Width; ++i)
            {
                img.Data[i] = m.Data[i + l * m.Height * m.Width];
            }
            return img;
        }

        private static float get_pixel(Image m, int x, int y, int c)
        {
            return m.Data[c * m.Height * m.Width + y * m.Width + x];
        }

        private static float get_pixel_extend(Image m, int x, int y, int c)
        {
            if (x < 0) x = 0;
            if (x >= m.Width) x = m.Width - 1;
            if (y < 0) y = 0;
            if (y >= m.Height) y = m.Height - 1;
            if (c < 0 || c >= m.NumberOfChannels) return 0;
            return get_pixel(m, x, y, c);
        }

        private static void set_pixel(Image m, int x, int y, int c, float val)
        {
            if (x < 0 || y < 0 || c < 0 || x >= m.Width || y >= m.Height || c >= m.NumberOfChannels)
            {
                return;
            }
            m.Data[c * m.Height * m.Width + y * m.Width + x] = val;
        }

        private static void add_pixel(Image m, int x, int y, int c, float val)
        {
            m.Data[c * m.Height * m.Width + y * m.Width + x] += val;
        }

        private static Image collapse_images_vert(Image[] ims, int n)
        {
            int border = 1;
            var w = ims[0].Width;
            var h = (ims[0].Height + border) * n - border;
            var c = ims[0].NumberOfChannels;
            if (c != 3)
            {
                w = (w + border) * c - border;
                c = 1;
            }

            Image filters = new Image(w, h, c);
            for (var i = 0; i < n; ++i)
            {
                int hOffset = i * (ims[0].Height + border);
                Image copy = new Image(ims[i]);
                //normalize_image(copy);
                if (c == 3)
                {
                    embed_image(copy, ref filters, 0, hOffset);
                }
                else
                {
                    for (var j = 0; j < copy.NumberOfChannels; ++j)
                    {
                        int wOffset = j * (ims[0].Width + border);
                        Image layer = get_image_layer(copy, j);
                        embed_image(layer, ref filters, wOffset, hOffset);
                    }
                }
            }
            return filters;
        }

        public static void show_images(Image[] ims, int n, string window)
        {
            Image m = collapse_images_vert(ims, n);
            normalize_image(m);
            save_image(m, window);
            show_image(m, window);
        }
    }
}