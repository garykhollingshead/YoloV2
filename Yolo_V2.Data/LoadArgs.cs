﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.UI;
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
            for (k = 0; k < source.C; ++k)
            {
                for (y = 0; y < source.H; ++y)
                {
                    for (x = 0; x < source.W; ++x)
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
            Image b = new Image(a.W + 2 * border, a.H + 2 * border, a.C);
            int x, y, k;
            for (k = 0; k < b.C; ++k)
            {
                for (y = 0; y < b.H; ++y)
                {
                    for (x = 0; x < b.W; ++x)
                    {
                        float val = get_pixel_extend(a, x - border, y - border, k);
                        if (x - border < 0 || x - border >= a.W || y - border < 0 || y - border >= a.H) val = 1;
                        set_pixel(b, x, y, k, val);
                    }
                }
            }

            return b;
        }

        private static Image tile_images(Image a, Image b, int dx)
        {
            if (a.W == 0)
            {
                return new Image(b);
            }
            Image c = new Image(a.W + b.W + dx, (a.H > b.H) ? a.H : b.H, (a.C > b.C) ? a.C : b.C);
            Blas.Fill_cpu(c.W * c.H * c.C, 1, c.Data, 1);
            embed_image(a, c, 0, 0);
            composite_image(b, c, a.W + dx, 0);
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

            return border_image(label, (int)(label.H * .25));
        }

        private static void draw_label(Image a, int r, int c, Image label, float[] rgb)
        {
            int w = label.W;
            int h = label.H;
            if (r - h >= 0) r = r - h;

            int i, j, k;
            for (j = 0; j < h && j + r < a.H; ++j)
            {
                for (i = 0; i < w && i + c < a.W; ++i)
                {
                    for (k = 0; k < label.C; ++k)
                    {
                        float val = get_pixel(label, i, j, k);
                        set_pixel(a, i + c, j + r, k, rgb[k] * val);
                    }
                }
            }
        }

        private static void draw_box(Image a, int x1, int y1, int x2, int y2, float r, float g, float b)
        {
            //normalize_image(a);
            int i;
            if (x1 < 0) x1 = 0;
            if (x1 >= a.W) x1 = a.W - 1;
            if (x2 < 0) x2 = 0;
            if (x2 >= a.W) x2 = a.W - 1;

            if (y1 < 0) y1 = 0;
            if (y1 >= a.H) y1 = a.H - 1;
            if (y2 < 0) y2 = 0;
            if (y2 >= a.H) y2 = a.H - 1;

            for (i = x1; i <= x2; ++i)
            {
                a.Data[i + y1 * a.W + 0 * a.W * a.H] = r;
                a.Data[i + y2 * a.W + 0 * a.W * a.H] = r;

                a.Data[i + y1 * a.W + 1 * a.W * a.H] = g;
                a.Data[i + y2 * a.W + 1 * a.W * a.H] = g;

                a.Data[i + y1 * a.W + 2 * a.W * a.H] = b;
                a.Data[i + y2 * a.W + 2 * a.W * a.H] = b;
            }

            for (i = y1; i <= y2; ++i)
            {
                a.Data[x1 + i * a.W + 0 * a.W * a.H] = r;
                a.Data[x2 + i * a.W + 0 * a.W * a.H] = r;

                a.Data[x1 + i * a.W + 1 * a.W * a.H] = g;
                a.Data[x2 + i * a.W + 1 * a.W * a.H] = g;

                a.Data[x1 + i * a.W + 2 * a.W * a.H] = b;
                a.Data[x2 + i * a.W + 2 * a.W * a.H] = b;
            }
        }

        public static void draw_box_width(Image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
        {
            int i;
            for (i = 0; i < w; ++i)
            {
                draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
            }
        }

        public static Image[][] load_alphabet()
        {
            int i, j;
            int nsize = 8;
            var alphabets = new Image[nsize][];
            for (j = 0; j < nsize; ++j)
            {
                alphabets[j] = new Image[128];
                for (i = 32; i < 127; ++i)
                {
                    string buff = $"data/labels/{i}_{j}.png";
                    alphabets[i][j] = load_image_color(buff, 0, 0);
                }
            }

            return alphabets;
        }

        public static void draw_detections(Image im, int num, float thresh, Box[] boxes, float[][] probs, string[] names,
            Image[][] alphabet, int classes)
        {
            int i;

            for (i = 0; i < num; ++i)
            {
                int curClass = Utils.max_index(probs[i], classes);
                float prob = probs[i][curClass];
                if (prob > thresh)
                {

                    int width = (int)(im.H * .012);

                    Console.WriteLine($"{names[curClass]}: {prob * 100: .0}%");
                    int offset = curClass * 123457 % classes;
                    float red = get_color(2, offset, classes);
                    float green = get_color(1, offset, classes);
                    float blue = get_color(0, offset, classes);
                    var rgb = new[]{
                        red, green, blue
                    };

                    Box b = boxes[i];

                    int left = (int)(b.X - b.W / 2) * im.W;
                    int right = (int)(b.X + b.W / 2) * im.W;
                    int top = (int)(b.Y - b.H / 2) * im.H;
                    int bot = (int)(b.Y + b.H / 2) * im.H;

                    if (left < 0) left = 0;
                    if (right > im.W - 1) right = im.W - 1;
                    if (top < 0) top = 0;
                    if (bot > im.H - 1) bot = im.H - 1;

                    draw_box_width(im, left, top, right, bot, width, red, green, blue);
                    if (alphabet.Length != 0)
                    {
                        Image label = get_label(alphabet, names[curClass], (int)(im.H * .03) / 10);
                        draw_label(im, top + width, left, label, rgb);
                    }
                }
            }
        }

        public static void flip_image(Image a)
        {
            int i, j, k;
            for (k = 0; k < a.C; ++k)
            {
                for (i = 0; i < a.H; ++i)
                {
                    for (j = 0; j < a.W / 2; ++j)
                    {
                        int index = j + a.W * (i + a.H * (k));
                        int flip = (a.W - j - 1) + a.W * (i + a.H * (k));
                        float swap = a.Data[flip];
                        a.Data[flip] = a.Data[index];
                        a.Data[index] = swap;
                    }
                }
            }
        }

        private static void embed_image(Image source, Image dest, int dx, int dy)
        {
            int x, y, k;
            for (k = 0; k < source.C; ++k)
            {
                for (y = 0; y < source.H; ++y)
                {
                    for (x = 0; x < source.W; ++x)
                    {
                        float val = get_pixel(source, x, y, k);
                        set_pixel(dest, dx + x, dy + y, k, val);
                    }
                }
            }
        }

        public static Image collapse_image_layers(Image source, int border)
        {
            int h = source.H;
            h = (h + border) * source.C - border;
            Image dest = new Image(source.W, h, 1);
            int i;
            for (i = 0; i < source.C; ++i)
            {
                Image layer = get_image_layer(source, i);
                int hOffset = i * (source.H + border);
                embed_image(layer, dest, 0, hOffset);
            }

            return dest;
        }

        public static void constrain_image(Image im)
        {
            int i;
            for (i = 0; i < im.W * im.H * im.C; ++i)
            {
                if (im.Data[i] < 0) im.Data[i] = 0;
                if (im.Data[i] > 1) im.Data[i] = 1;
            }
        }

        private static void normalize_image(Image p)
        {
            int i;
            float min = 9999999;
            float max = -999999;

            for (i = 0; i < p.H * p.W * p.C; ++i)
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

            for (i = 0; i < p.C * p.W * p.H; ++i)
            {
                p.Data[i] = (p.Data[i] - min) / (max - min);
            }
        }

        public static void rgbgr_image(Image im)
        {
            int i;
            for (i = 0; i < im.W * im.H; ++i)
            {
                float swap = im.Data[i];
                im.Data[i] = im.Data[i + im.W * im.H * 2];
                im.Data[i + im.W * im.H * 2] = swap;
            }
        }

        private static void show_image_cv(Image p, string name)
        {
            int x, y, k;
            Image copy = new Image(p);
            constrain_image(copy);
            if (p.C == 3) rgbgr_image(copy);

            string buff = name;

            using (Mat disp = copy.ToMat())
            {
                ImageViewer.Show(disp, buff);

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
                    rgbgr_image(retImage);
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
                rgbgr_image(im);
                return im;
            }
        }

        private static void save_image_jpg(Image p, string name)
        {

            using (var img = p.ToMat())
            {
                img.Save($"{name}.jpg");
            }
        }

        public static void save_image_png(Image im, string name)
        {
            using (var img = im.ToMat())
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
            float cx = im.W / 2.0f;
            float cy = im.H / 2.0f;
            Image rot = new Image(w, h, im.C);
            for (c = 0; c < im.C; ++c)
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
            float cx = im.W / 2.0f;
            float cy = im.H / 2.0f;
            Image rot = new Image(im.W, im.H, im.C);
            for (c = 0; c < im.C; ++c)
            {
                for (y = 0; y < im.H; ++y)
                {
                    for (x = 0; x < im.W; ++x)
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
            for (i = 0; i < m.H * m.W * m.C; ++i) m.Data[i] *= s;
        }

        public static Image crop_image(Image im, int dx, int dy, int w, int h)
        {
            Image cropped = new Image(w, h, im.C);
            int i, j, k;
            for (k = 0; k < im.C; ++k)
            {
                for (j = 0; j < h; ++j)
                {
                    for (i = 0; i < w; ++i)
                    {
                        int r = j + dy;
                        int c = i + dx;
                        float val = 0;
                        r = Utils.constrain_int(r, 0, im.H - 1);
                        c = Utils.constrain_int(c, 0, im.W - 1);
                        if (r >= 0 && r < im.H && c >= 0 && c < im.W)
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
            Image c1 = crop_image(b, 0, mid, b.W, b.H);
            Image c2 = crop_image(b, 0, mid + 1, b.W, b.H);
            float d1 = Utils.dist_array(c1.Data, a.Data, a.W * a.H * a.C, 10);
            float d2 = Utils.dist_array(c2.Data, a.Data, a.W * a.H * a.C, 10);
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
            int shift = best_3d_shift_r(a, b, -a.H / 100, a.H / 100);

            Image c1 = crop_image(b, 10, shift, b.W, b.H);
            float d1 = Utils.dist_array(c1.Data, a.Data, a.W * a.H * a.C, 100);
            Image c2 = crop_image(b, -10, shift, b.W, b.H);
            float d2 = Utils.dist_array(c2.Data, a.Data, a.W * a.H * a.C, 100);

            Console.WriteLine("%d\n", shift);

            Image c = crop_image(b, delta, shift, a.W, a.H);
            int i;
            for (i = 0; i < c.W * c.H; ++i)
            {
                c.Data[i] = a.Data[i];
            }
            save_image_jpg(c, output);
        }

        public static Image resize_min(Image im, int min)
        {
            int w = im.W;
            int h = im.H;
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
            if (w == im.W && h == im.H) return im;
            Image resized = resize_image(im, w, h);
            return resized;
        }

        public static Image random_crop_image(Image im, int w, int h)
        {
            int dx = Utils.rand_int(0, im.W - w);
            int dy = Utils.rand_int(0, im.H - h);
            Image crop = crop_image(im, dx, dy, w, h);
            return crop;
        }

        public static Image random_augment_image(Image im, float angle, float aspect, int low, int high, int size)
        {
            aspect = Utils.rand_scale(aspect);
            int r = Utils.rand_int(low, high);
            int min = (im.H < im.W * aspect)
                ? im.H
                : (int)(im.W * aspect);
            float scale = (float)r / min;

            float rad = (float)(Utils.rand_uniform(-angle, angle) * Math.PI * 2 / 360.0f);

            float dx = (im.W * scale / aspect - size) / 2.0f;
            float dy = (im.H * scale - size) / 2.0f;
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
            for (j = 0; j < im.H; ++j)
            {
                for (i = 0; i < im.W; ++i)
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
            for (j = 0; j < im.H; ++j)
            {
                for (i = 0; i < im.W; ++i)
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
            Image gray = new Image(im.W, im.H, 1);
            float[] scale = { 0.587f, 0.299f, 0.114f };
            for (k = 0; k < im.C; ++k)
            {
                for (j = 0; j < im.H; ++j)
                {
                    for (i = 0; i < im.W; ++i)
                    {
                        gray.Data[i + im.W * j] += scale[k] * get_pixel(im, i, j, k);
                    }
                }
            }
            return gray;
        }

        public static Image threshold_image(Image im, float thresh)
        {
            int i;
            Image t = new Image(im.W, im.H, im.C);
            for (i = 0; i < im.W * im.H * im.C; ++i)
            {
                t.Data[i] = im.Data[i] > thresh ? 1 : 0;
            }
            return t;
        }

        private static void scale_image_channel(Image im, int c, float v)
        {
            int i, j;
            for (j = 0; j < im.H; ++j)
            {
                for (i = 0; i < im.W; ++i)
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
            for (i = 0; i < im.W * im.H; ++i)
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
            Image resized = new Image(w, h, im.C);
            Image part = new Image(w, im.H, im.C);
            int r, c, k;
            float wScale = (float)(im.W - 1) / (w - 1);
            float hScale = (float)(im.H - 1) / (h - 1);
            for (k = 0; k < im.C; ++k)
            {
                for (r = 0; r < im.H; ++r)
                {
                    for (c = 0; c < w; ++c)
                    {
                        float val = 0;
                        if (c == w - 1 || im.W == 1)
                        {
                            val = get_pixel(im, im.W - 1, r, k);
                        }
                        else
                        {
                            float sx = c * wScale;
                            int ix = (int)sx;
                            float dx = sx - ix;
                            val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                        }
                        set_pixel(part, c, r, k, val);
                    }
                }
            }
            for (k = 0; k < im.C; ++k)
            {
                for (r = 0; r < h; ++r)
                {
                    float sy = r * hScale;
                    int iy = (int)sy;
                    float dy = sy - iy;
                    for (c = 0; c < w; ++c)
                    {
                        float val = (1 - dy) * get_pixel(part, c, iy, k);
                        set_pixel(resized, c, r, k, val);
                    }
                    if (r == h - 1 || im.H == 1) continue;
                    for (c = 0; c < w; ++c)
                    {
                        float val = dy * get_pixel(part, c, iy + 1, k);
                        add_pixel(resized, c, r, k, val);
                    }
                }
            }

            return resized;
        }

        public static void test_resize(string filename)
        {
            Image im = load_image(filename, 0, 0, 3);
            float mag = Utils.mag_array(im.Data, im.W * im.H * im.C);
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

            if (h != img.H || w != img.W)
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
            Image img = new Image(m.W, m.H, 1);

            for (var i = 0; i < m.H * m.W; ++i)
            {
                img.Data[i] = m.Data[i + l * m.H * m.W];
            }
            return img;
        }

        private static float get_pixel(Image m, int x, int y, int c)
        {
            return m.Data[c * m.H * m.W + y * m.W + x];
        }

        private static float get_pixel_extend(Image m, int x, int y, int c)
        {
            if (x < 0) x = 0;
            if (x >= m.W) x = m.W - 1;
            if (y < 0) y = 0;
            if (y >= m.H) y = m.H - 1;
            if (c < 0 || c >= m.C) return 0;
            return get_pixel(m, x, y, c);
        }

        private static void set_pixel(Image m, int x, int y, int c, float val)
        {
            if (x < 0 || y < 0 || c < 0 || x >= m.W || y >= m.H || c >= m.C)
            {
                return;
            }
            m.Data[c * m.H * m.W + y * m.W + x] = val;
        }

        private static void add_pixel(Image m, int x, int y, int c, float val)
        {
            m.Data[c * m.H * m.W + y * m.W + x] += val;
        }

        private static Image collapse_images_vert(Image[] ims, int n)
        {
            int border = 1;
            var w = ims[0].W;
            var h = (ims[0].H + border) * n - border;
            var c = ims[0].C;
            if (c != 3)
            {
                w = (w + border) * c - border;
                c = 1;
            }

            Image filters = new Image(w, h, c);
            for (var i = 0; i < n; ++i)
            {
                int hOffset = i * (ims[0].H + border);
                Image copy = new Image(ims[i]);
                //normalize_image(copy);
                if (c == 3)
                {
                    embed_image(copy, filters, 0, hOffset);
                }
                else
                {
                    for (var j = 0; j < copy.C; ++j)
                    {
                        int wOffset = j * (ims[0].W + border);
                        Image layer = get_image_layer(copy, j);
                        embed_image(layer, filters, wOffset, hOffset);
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