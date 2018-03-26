﻿using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace Yolo_V2.Data.Enums
{
    public class Image
    {
        public int H;
        public int W;
        public int C;
        public float[] Data;
        
        public Image(int c = 0, int h = 0, int w = 0, float[] data = null)
        {
            C = c;
            H = h;
            W = w;
            Data = data ?? new float[h*w*c];
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
            var step = src.Step;
            var count = 0;

            for (var k = 0; k < C; ++k)
            {
                for (var i = 0; i < H; ++i)
                {
                    for (var j = 0; j < W; ++j)
                    {
                        //var value = new double[1];
                        //Marshal.Copy(src.DataPointer + (row * mat.Cols + col) * mat.ElementSize, value, 0, 1);
                        //Marshal.Copy(src.DataPointer + i * step + j * c + k, value, 0, 1);
                        //img.Data[count++] = data[i * step + j * c + k] / 255;
                        Data[count++] = (float)Image.GetPixel(src, i * step + j * C + k) / 255;

                    }
                }
            }
        }

        public Mat ToMat()
        {
            var newMat = new Mat(new System.Drawing.Size(W, H), DepthType.Cv8U, C);
            //TODO: get data from Data to mat
            return newMat;
        }

        public static void SetPixel(Mat img, int offset, double value)
        {
            var target = new[] { value };
            Marshal.Copy(target, 0, img.DataPointer + offset, 1);
        }

        public static void SetPixel(Mat mat, int row, int col, double value)
        {
            var target = new[] { value };
            Marshal.Copy(target, 0, mat.DataPointer + (row * mat.Cols + col) * mat.ElementSize, 1);
        }

        public static double GetPixel(Mat mat, int row, int col)
        {
            var value = new double[1];
            Marshal.Copy(mat.DataPointer + (row * mat.Cols + col) * mat.ElementSize, value, 0, 1);
            return value[0];
        }

        public static double GetPixel(Mat mat, int offset)
        {
            var value = new double[1];
            Marshal.Copy(mat.DataPointer + offset * mat.ElementSize, value, 0, 1);
            return value[0];
        }
    }
}