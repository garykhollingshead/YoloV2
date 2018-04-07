namespace Yolo_V2.Data
{
    public class Image
    {
        //    public int H;
        //    public int W;
        //    public int C;
        //    public float[] Data;

        //    public Mat(int w = 0, int h = 0, int c = 0, float[] data = null, int start = 0)
        //    {
        //        C = c;
        //        H = h;
        //        W = w;
        //        if (data != null)
        //        {
        //            Data = new float[data.Length - start];
        //            Array.Copy(data, start, Data, 0, Data.Length);
        //        }
        //        else
        //        {
        //            Data = new float[h * w * c];
        //        }
        //    }
        //    public Mat(int w, int h, int c, byte[] data)
        //    {
        //        C = c;
        //        H = h;
        //        W = w;
        //        Data = new float[data.Length];
        //        for (var i = 0; i < data.Length; ++i)
        //        {
        //            Data[i] = data[i];
        //        }
        //    }

        //    public Mat(Mat p)
        //    {
        //        C = p.C;
        //        H = p.H;
        //        W = p.W;
        //        Data = new float[p.Data.Length];
        //        Array.Copy(p.Data, Data, Data.Length);
        //    }

        //    public Mat(Mat src)
        //    {
        //        C = src.NumberOfChannels;
        //        H = src.Height;
        //        W = src.Width;
        //        Data = GetFloats(src.GetData());
        //    }

        //    public Mat ToMat()
        //    {
        //        byte[] byteData = GetBytes(Data);
        //        GCHandle hand = GCHandle.Alloc(byteData, GCHandleType.Pinned);
        //        var newMat = new Mat(new Size(W, H), DepthType.Cv8U, C, hand.AddrOfPinnedObject(), C * W);
        //        hand.Free();
        //        return newMat;
        //    }

        public static float[] GetFloats(byte[] bytes)
        {
            var data = new float[bytes.Length];
            for (var i = 0; i < bytes.Length; ++i)
            {
                data[i] = bytes[i];
            }

            return data;
        }

        public static byte[] GetBytes(float[] floats)
        {
            byte[] byteData = new byte[floats.Length];
            for (var i = 0; i < floats.Length; ++i)
            {
                byteData[i] = (byte)floats[i];
            }

            return byteData;
        }
    }
}