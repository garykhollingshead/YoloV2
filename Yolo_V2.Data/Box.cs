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

        private static float Overlap(float x1, float w1, float x2, float w2)
        {
            float l1 = x1 - w1 / 2;
            float l2 = x2 - w2 / 2;
            float left = l1 > l2 ? l1 : l2;
            float r1 = x1 + w1 / 2;
            float r2 = x2 + w2 / 2;
            float right = r1 < r2 ? r1 : r2;
            return right - left;
        }

        private static float box_intersection(Box a, Box b)
        {
            float w = Overlap(a.X, a.W, b.X, b.W);
            float h = Overlap(a.Y, a.H, b.Y, b.H);
            if (w < 0 || h < 0) return 0;
            float area = w * h;
            return area;
        }

        private static float box_union(Box a, Box b)
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

        private static int nms_comparator(SortableBbox a, SortableBbox b)
        {
            float diff = a.Probs[a.Index][b.Sclass] - b.Probs[b.Index][b.Sclass];
            if (diff < 0) return 1;
            if (diff > 0) return -1;
            return 0;
        }
        
        public static void do_nms_sort(Box[] boxes, float[][] probs, int total, int classes, float thresh)
        {
            int i, j, k;
            SortableBbox[] s = new SortableBbox[total];

            for (i = 0; i < total; ++i)
            {
                s[i].Index = i;
                s[i].Sclass = 0;
                s[i].Probs = probs;
            }

            for (k = 0; k < classes; ++k)
            {
                for (i = 0; i < total; ++i)
                {
                    s[i].Sclass = k;
                }
                Array.Sort(s, nms_comparator);
                for (i = 0; i < total; ++i)
                {
                    if (probs[s[i].Index][k] == 0) continue;
                    Box a = boxes[s[i].Index];
                    for (j = i + 1; j < total; ++j)
                    {
                        Box b = boxes[s[j].Index];
                        if (box_iou(a, b) > thresh)
                        {
                            probs[s[j].Index][k] = 0;
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
    }
}