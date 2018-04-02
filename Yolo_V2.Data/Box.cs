using System;

namespace Yolo_V2.Data
{
    public class Box
    {
        public float X;
        public float Y;
        public float W;
        public float H;

        public Box() { }

        public Box(float[] f, int offset = 0)
        {
            X = f[0 + offset];
            Y = f[1 + offset];
            W = f[2 + offset];
            H = f[3 + offset];
        }

        public Box(float x, float y, float w, float h)
        {
            X = x;
            Y = y;
            W = w;
            H = h;
        }

        public static DBox derivative(Box a, Box b)
        {
            DBox d = new DBox();
            d.DX = 0;
            d.DW = 0;
            float l1 = a.X - a.W / 2;
            float l2 = b.X - b.W / 2;
            if (l1 > l2)
            {
                d.DX -= 1;
                d.DW += .5f;
            }
            float r1 = a.X + a.W / 2;
            float r2 = b.X + b.W / 2;
            if (r1 < r2)
            {
                d.DX += 1;
                d.DW += .5f;
            }
            if (l1 > r2)
            {
                d.DX = -1;
                d.DW = 0;
            }
            if (r1 < l2)
            {
                d.DX = 1;
                d.DW = 0;
            }

            d.DY = 0;
            d.DH = 0;
            float t1 = a.Y - a.H / 2;
            float t2 = b.Y - b.H / 2;
            if (t1 > t2)
            {
                d.DY -= 1;
                d.DH += .5f;
            }
            float b1 = a.Y + a.H / 2;
            float b2 = b.Y + b.H / 2;
            if (b1 < b2)
            {
                d.DY += 1;
                d.DH += .5f;
            }
            if (t1 > b2)
            {
                d.DY = -1;
                d.DH = 0;
            }
            if (b1 < t2)
            {
                d.DY = 1;
                d.DH = 0;
            }
            return d;
        }
        
        public static float overlap(float x1, float w1, float x2, float w2)
        {
            float l1 = x1 - w1 / 2;
            float l2 = x2 - w2 / 2;
            float left = l1 > l2 ? l1 : l2;
            float r1 = x1 + w1 / 2;
            float r2 = x2 + w2 / 2;
            float right = r1 < r2 ? r1 : r2;
            return right - left;
        }
        
        public static float box_intersection(Box a, Box b)
        {
            float w = overlap(a.X, a.W, b.X, b.W);
            float h = overlap(a.Y, a.H, b.Y, b.H);
            if (w < 0 || h < 0) return 0;
            float area = w * h;
            return area;
        }
        
        public static float box_union(Box a, Box b)
        {
            float i = box_intersection(a, b);
            float u = a.W * a.H + b.W * b.H - i;
            return u;
        }
        
        public static float box_iou(Box a, Box b)
        {
            return box_intersection(a, b) / box_union(a, b);
        }
        
        public static float box_rmse(Box a, Box b)
        {
            return (float)Math.Sqrt(Math.Pow(a.X - b.X, 2) +
                        Math.Pow(a.Y - b.Y, 2) +
                        Math.Pow(a.W - b.W, 2) +
                        Math.Pow(a.H - b.H, 2));
        }
        
        public static DBox dintersect(Box a, Box b)
        {
            float w = overlap(a.X, a.W, b.X, b.W);
            float h = overlap(a.Y, a.H, b.Y, b.H);
            DBox dover = derivative(a, b);
            DBox di = new DBox();

            di.DW = dover.DW * h;
            di.DX = dover.DX * h;
            di.DH = dover.DH * w;
            di.DY = dover.DY * w;

            return di;
        }
        
        public static DBox dunion(Box a, Box b)
        {
            DBox du = new DBox();

            DBox di = dintersect(a, b);
            du.DW = a.H - di.DW;
            du.DH = a.W - di.DH;
            du.DX = -di.DX;
            du.DY = -di.DY;

            return du;
        }
        
        public static void test_dunion()
        {
            Box a = new Box(0, 0, 1, 1);
            Box dxa = new Box(0 + .0001f, 0, 1, 1);
            Box dya = new Box(0, 0 + .0001f, 1, 1);
            Box dwa = new Box(0, 0, 1 + .0001f, 1);
            Box dha = new Box(0, 0, 1, 1 + .0001f);

            Box b = new Box(.5f, .5f, .2f, .2f);
            DBox di = dunion(a, b);
            Console.Write($"Union: {di.DX} {di.DY} {di.DW} {di.DH}\n");
            float inter = box_union(a, b);
            float xinter = box_union(dxa, b);
            float yinter = box_union(dya, b);
            float winter = box_union(dwa, b);
            float hinter = box_union(dha, b);
            xinter = (xinter - inter) / (.0001f);
            yinter = (yinter - inter) / (.0001f);
            winter = (winter - inter) / (.0001f);
            hinter = (hinter - inter) / (.0001f);
            Console.Write($"Union Manual {xinter} {yinter} {winter} {hinter}\n");
        }
        
        public static void test_dintersect()
        {
            Box a = new Box(0, 0, 1, 1);
            Box dxa = new Box(0 + .0001f, 0, 1, 1);
            Box dya = new Box(0, 0 + .0001f, 1, 1);
            Box dwa = new Box(0, 0, 1 + .0001f, 1);
            Box dha = new Box(0, 0, 1, 1 + .0001f);

            Box b = new Box(.5f, .5f, .2f, .2f);
            DBox di = dintersect(a, b);
            Console.Write($"Inter: {di.DX} {di.DY} {di.DW} {di.DH}\n");
            float inter = box_intersection(a, b);
            float xinter = box_intersection(dxa, b);
            float yinter = box_intersection(dya, b);
            float winter = box_intersection(dwa, b);
            float hinter = box_intersection(dha, b);
            xinter = (xinter - inter) / (.0001f);
            yinter = (yinter - inter) / (.0001f);
            winter = (winter - inter) / (.0001f);
            hinter = (hinter - inter) / (.0001f);
            Console.Write($"Union Manual {xinter} {yinter} {winter} {hinter}\n");
        }
        
        public static void test_box()
        {
            test_dintersect();
            test_dunion();
            Box a = new Box(0, 0, 1, 1);
            Box dxa = new Box(0 + .00001f, 0, 1, 1);
            Box dya = new Box(0, 0 + .00001f, 1, 1);
            Box dwa = new Box(0, 0, 1 + .00001f, 1);
            Box dha = new Box(0, 0, 1, 1 + .00001f);

            Box b = new Box(.5f, 0, .2f, .2f);

            float iou = box_iou(a, b);
            iou = (1 - iou) * (1 - iou);
            Console.Write($"{iou}\n", iou);
            DBox d = diou(a, b);
            Console.Write($"{d.DX} {d.DY} {d.DW} {d.DH}\n");

            float xiou = box_iou(dxa, b);
            float yiou = box_iou(dya, b);
            float wiou = box_iou(dwa, b);
            float hiou = box_iou(dha, b);
            xiou = ((1 - xiou) * (1 - xiou) - iou) / (.00001f);
            yiou = ((1 - yiou) * (1 - yiou) - iou) / (.00001f);
            wiou = ((1 - wiou) * (1 - wiou) - iou) / (.00001f);
            hiou = ((1 - hiou) * (1 - hiou) - iou) / (.00001f);
            Console.Write($"manual {xiou} {yiou} {wiou} {hiou}\n");
        }
        
        public static DBox diou(Box a, Box b)
        {
            float u = box_union(a, b);
            float i = box_intersection(a, b);
            DBox di = dintersect(a, b);
            DBox du = dunion(a, b);
            DBox dd = new DBox();

            dd.DX = b.X - a.X;
            dd.DY = b.Y - a.Y;
            dd.DW = b.W - a.W;
            dd.DH = b.H - a.H;
            return dd;
        }
        
        public static int nms_comparator(SortableBbox a, SortableBbox b)
        {
            float diff = a.probs[a.index][b.sclass] - b.probs[b.index][b.sclass];
            if (diff < 0) return 1;
            else if (diff > 0) return -1;
            return 0;
        }
        
        public static void do_nms_sort(Box[] boxes, float[][] probs, int total, int classes, float thresh)
        {
            int i, j, k;
            SortableBbox[] s = new SortableBbox[total];

            for (i = 0; i < total; ++i)
            {
                s[i].index = i;
                s[i].sclass = 0;
                s[i].probs = probs;
            }

            for (k = 0; k < classes; ++k)
            {
                for (i = 0; i < total; ++i)
                {
                    s[i].sclass = k;
                }
                Array.Sort(s, nms_comparator);
                for (i = 0; i < total; ++i)
                {
                    if (probs[s[i].index][k] == 0) continue;
                    Box a = boxes[s[i].index];
                    for (j = i + 1; j < total; ++j)
                    {
                        Box b = boxes[s[j].index];
                        if (box_iou(a, b) > thresh)
                        {
                            probs[s[j].index][k] = 0;
                        }
                    }
                }
            }
        }
        
        public static void do_nms(Box[] boxes, float[][] probs, int total, int classes, float thresh)
        {
            int i, j, k;
            for (i = 0; i < total; ++i)
            {
                bool any = false;
                for (k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
                if (!any)
                {
                    continue;
                }
                for (j = i + 1; j < total; ++j)
                {
                    if (box_iou(boxes[i], boxes[j]) > thresh)
                    {
                        for (k = 0; k < classes; ++k)
                        {
                            if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                            else probs[j][k] = 0;
                        }
                    }
                }
            }
        }
        
        public static Box encode_box(Box b, Box anchor)
        {
            Box encode = new Box();
            encode.X = (b.X - anchor.X) / anchor.W;
            encode.Y = (b.Y - anchor.Y) / anchor.H;
            encode.W = (float)Math.Log(b.W / anchor.W, 2);
            encode.H = (float)Math.Log(b.H / anchor.H, 2);
            return encode;
        }
        
        public static Box decode_box(Box b, Box anchor)
        {
            Box decode = new Box();
            decode.X = b.X * anchor.W + anchor.X;
            decode.Y = b.Y * anchor.H + anchor.Y;
            decode.W = (float)Math.Pow(2.0, b.W) * anchor.W;
            decode.H = (float)Math.Pow(2.0, b.H) * anchor.H;
            return decode;
        }
    }
}