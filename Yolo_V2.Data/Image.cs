using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace Yolo_V2.Data
{
    public class Image
    {
        public readonly int Height;
        public readonly int Width;
        public readonly int NumberOfChannels;

        // Data in the form of BGR, ie 2x2 image would be:
        // BBBBGGGGRRRR
        public float[] Data;

        public Image(int width = 0, int height = 0, int numberOfChannels = 0, float[] data = null, int start = 0)
        {
            NumberOfChannels = numberOfChannels;
            Height = height;
            Width = width;
            if (data != null)
            {
                Data = new float[data.Length - start];
                Array.Copy(data, start, Data, 0, Data.Length);
            }
            else
            {
                Data = new float[height * width * numberOfChannels];
            }
        }

        public Image(Image p)
        {
            NumberOfChannels = p.NumberOfChannels;
            Height = p.Height;
            Width = p.Width;
            Data = new float[p.Data.Length];
            Array.Copy(p.Data, Data, Data.Length);
        }

        public Image(Mat src)
        {
            NumberOfChannels = src.NumberOfChannels;
            Height = src.Height;
            Width = src.Width;
            Data = new float[NumberOfChannels * Width * Height];
            var byteData = src.GetData();
            var channelSize = Width * Height;
            var red = 0;
            var green = channelSize;
            var blue = channelSize * 2;
            for (var i = 0; i < byteData.Length; i += 3)
            {
                Data[red + i / 3] = byteData[i + 2] / 255.0f;
                Data[green + i / 3] = byteData[i + 1] / 255.0f;
                Data[blue + i / 3] = byteData[i] / 255.0f;
            }
        }

        public Mat ToMat()
        {
            // RRRGGGBBB going to BGRBGRBGR
            byte[] byteData = new byte[Width * Height * NumberOfChannels];
            Mat newMat;
            var channelSize = Width * Height;
            var red = 0;
            var green = channelSize;
            var blue = channelSize * 2;
            // image might have been shrunk to values ~0-1
            var temp = new float[Data.Length];
            Array.Copy(Data, temp, Data.Length);
            for (var i = 0; i < Data.Length; i += 3)
            {
                if (temp[blue + i / 3] > 1)
                    temp[blue + i / 3] = 1;
                if (temp[blue + i / 3] < 0)
                    temp[blue + i / 3] = 0;
                byteData[i] = (byte)(temp[blue + i / 3] * 255);
                if (temp[green + i / 3] > 1)
                    temp[green + i / 3] = 1;
                if (temp[green + i / 3] < 0)
                    temp[green + i / 3] = 0;
                byteData[i + 1] = (byte)(temp[green + i / 3] * 255);
                if (temp[red + i / 3] > 1)
                    temp[red + i / 3] = 1;
                if (temp[red + i / 3] < 0)
                    temp[red + i / 3] = 0;
                byteData[i + 2] = (byte)(temp[red + i / 3] * 255);
            }
            newMat = new Mat(new Size(Width, Height), DepthType.Cv8U, NumberOfChannels);
            newMat.SetTo(byteData);

            return newMat;
        }
    }
}