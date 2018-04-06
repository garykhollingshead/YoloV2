using System;
using System.Collections.Generic;
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

        public Image(int w = 0, int h = 0, int c = 0, float[] data = null, int start = 0)
        {
            C = c;
            H = h;
            W = w;
            if (data != null)
            {
                Data = new float[data.Length - start];
                Array.Copy(data, start, Data, 0, Data.Length);
            }
            else
            {
                Data = new float[h * w * c];
            }
        }

        public Image(Image p)
        {
            C = p.C;
            H = p.H;
            W = p.W;
            Data = new float[p.Data.Length];
            Array.Copy(p.Data, Data, Data.Length);
        }

        public Image(Mat src)
        {
            C = src.NumberOfChannels;
            H = src.Height;
            W = src.Width;
            Data = new float[C * W * H];
            var byteData = src.GetData();
            for (var i = 0; i < byteData.Length; ++i)
            {
                Data[i] = byteData[i];
            }
        }

        public Mat ToMat()
        {
            byte[] byteData = new byte[W * H * C];
            for (var i = 0; i < Data.Length; ++i)
            {
                byteData[i] = (byte)Data[i];
            }
            var newMat = new Mat(new Size(W, H), DepthType.Cv8U, C);
            return newMat;
        }
    }
}