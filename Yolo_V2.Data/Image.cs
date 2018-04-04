using System;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace Yolo_V2.Data
{
    public class Image
    {
        public int H;
        public int W;
        public int C;
        public float[] Data;
        
        public Image(int c = 0, int h = 0, int w = 0, float[] data = null, int start = 0)
        {
            C = c;
            H = h;
            W = w;
            if (data != null )
            {
                if (start != 0)
                {
                    Data = new float[data.Length - start];
                    Array.Copy(data, start, Data, 0, Data.Length);
                }
                else
                {
                    Data = data;
                }
            }
            else
            {
                Data = new float[h*w*c];
            }
        }

        public Image(Image p)
        {
            C = p.C;
            H = p.H;
            W = p.W;
            Data = p.Data.ToArray();
        }

        public Image(Mat src)
        {
            C = src.NumberOfChannels;
            H = src.Height;
            W = src.Width;
            Data = new float[C * W * H];
            GCHandle hand = GCHandle.Alloc(Data, GCHandleType.Pinned);
            using (var img2 = new Mat(src.Size, src.Depth, src.NumberOfChannels, hand.AddrOfPinnedObject(), src.Width * src.NumberOfChannels))
            {
                CvInvoke.BitwiseNot(src, img2);
                CvInvoke.BitwiseNot(img2, img2);
            }
            hand.Free();
        }

        public Mat ToMat()
        {
            GCHandle hand = GCHandle.Alloc(Data, GCHandleType.Pinned);
            var newMat = new Mat(new Size(W, H), DepthType.Cv8U, C, hand.AddrOfPinnedObject(), W * C);
            hand.Free();
            return newMat;
        }
    }
}