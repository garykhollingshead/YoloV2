using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Emgu.CV;
using Emgu.CV.Util;

namespace Yolo_V2.Data
{
    public static class Utils
    {
        public const int SecretNum = -1234;

        public static int[] read_map(string filename)
        {
            if (!File.Exists(filename))
            {
                file_error(filename);
            }

            return File.ReadAllLines(filename).Select(int.Parse).ToArray();
        }

        public static void IncArray(ref byte[] array, ref byte[] backup, int oldIndex, int index)
        {
            if (backup == null || backup.Length == 0)
            {
                backup = new byte[array.Length];
                Array.Copy(array, backup, backup.Length);
            }
            else
            {
                Array.Copy(array, 0, backup, oldIndex, array.Length);
            }
            var newSize = backup.Length - index;
            array = new byte[newSize];
            Array.Copy(backup, index, array, 0, newSize);
        }

        public static void DecArray(ref byte[] array, ref byte[] backup, int oldIndex, int index)
        {
            Array.Copy(array, 0, backup, oldIndex, array.Length);

            var newSize = array.Length + index;
            array = new byte[newSize];
            Array.Copy(backup, index, array, 0, newSize);
        }

        public static void sorta_shuffle<T>(T[] arr, int sections)
        {
            int n = 0;
            while (n <= sections)
            {
                int k = Rand.Next(0, sections);
                T value = arr[k];
                arr[k] = arr[n];
                arr[n] = value;
                n++;
            }
        }

        public static void Shuffle<T>(T[] arr)
        {
            int n = arr.Length;
            while (n > 1)
            {
                int k = Rand.Next(0, n);
                n--;
                T value = arr[k];
                arr[k] = arr[n];
                arr[n] = value;
            }
        }

        private static void del_arg(List<string> argv, int index)
        {
            argv.RemoveAt(index);
        }

        public static bool find_arg(List<string> argv, string arg)
        {
            var index = argv.FindIndex(arg.Equals);
            if (index < 0)
            {
                return false;
            }
            del_arg(argv, index);
            return true;
        }

        public static T find_value_arg<T>(List<string> argv, string arg, T def)
        {
            var index = argv.FindIndex(arg.Equals);
            if (index < 0)
            {
                return def;
            }

            del_arg(argv, index);
            var value = (T)Convert.ChangeType(argv[index], typeof(T));
            del_arg(argv, index);
            return value;
        }

        public static string Basecfg(string cfgfile)
        {
            var filenameIndex = cfgfile.LastIndexOf("/");
            var filename = cfgfile.Substring(filenameIndex);
            var dotIndex = filename.IndexOf(".");
            return dotIndex < 0 ? filename : filename.Substring(0, dotIndex - 1);
        }

        public static void find_replace(string str, string orig, string rep, out string output)
        {
            output = str.Replace(orig, rep);
        }

        public static void top_k(byte[] a, int n, int k, int[] index)
        {
            var temp = new Dictionary<byte, int>();
            for (var l = 0; l < n; ++l)
            {
                temp[a[l]] = l;
            }

            Array.Sort(a);

            for (var i = 0; i < k; ++i)
            {
                index[i] = temp[a[i]];
            }
        }

        public static void Error(string s)
        {
            Console.Error.WriteLine(s);
            throw new Exception(s);
        }

        public static void file_error(string s)
        {
            Console.Error.WriteLine($"Couldn't open file: {s}");
            throw new Exception($"Couldn't open file: {s}");
        }

        public static string Strip(string s)
        {
            var len = s.Length;
            for (var i = 0; i < len; ++i)
            {
                var c = s[i];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                {
                    s = s.Remove(i, 1);
                    len--;
                    i--;
                }
            }

            return s;
        }

        public static int count_fields(string line)
        {
            return line.Count(x => x == ',');
        }

        public static byte[] parse_fields(string line)
        {
            var fields = new List<byte>();
            foreach (var s in line.Split(','))
            {
                byte t;
                t = byte.TryParse(s, out t)
                    ? t
                    : (byte)0;
                fields.Add(t);
            }

            return fields.ToArray();
        }

        public static byte mean_array(byte[] a, int n, int aStart = 0)
        {
            var sum = 0f;
            for (var i = 0; i < n; ++i)
            {
                sum += a[aStart + i];
            }
            return (byte)(sum / n);
        }

        public static void mean_arrays(byte[][] a, int n, int els, out byte[] avg)
        {
            int i;
            int j;
            var avgf = new float[els];
            avg = new byte[els];
            for (j = 0; j < n; ++j)
            {
                for (i = 0; i < els; ++i)
                {
                    avgf[i] += a[j][i];
                }
            }
            for (i = 0; i < els; ++i)
            {
                avg[i] = (byte)(avgf[i] / n);
            }
        }

        public static void print_statistics(byte[] a, int n, int aStart = 0)
        {
            byte m = mean_array(a, n, aStart);
            byte v = variance_array(a, n, aStart);
            Console.WriteLine($"MSE: {mse_array(a, n, aStart):F6}, Mean: {m:F6}f, Variance: {v}f");
        }

        public static byte variance_array(byte[] a, int n, int aStart = 0)
        {
            int i;
            float sum = 0;
            byte mean = mean_array(a, n, aStart);
            for (i = 0; i < n; ++i) sum += (a[aStart + i] - mean) * (a[aStart + i] - mean);
            byte variance = (byte)(sum / n);
            return variance;
        }

        public static int constrain_int(int a, int min, int max)
        {
            if (a < min) return min;
            if (a > max) return max;
            return a;
        }

        public static byte Constrain(byte min, byte max, byte a)
        {
            if (a < min) return min;
            if (a > max) return max;
            return a;
        }

        public static byte dist_array(byte[] a, byte[] b, int n, int sub)
        {
            int i;
            double sum = 0;
            for (i = 0; i < n; i += sub) sum += Math.Pow(a[i] - b[i], 2);
            return (byte)Math.Sqrt(sum);
        }

        public static byte mse_array(byte[] a, int n, int aStart = 0)
        {
            int i;
            float sum = 0;
            for (i = 0; i < n; ++i) sum += a[aStart + i] * a[aStart + i];
            return (byte)Math.Sqrt(sum / n);
        }

        public static void normalize_array(ref byte[] a, int n)
        {
            using (var array = new VectorOfByte(a))
            using (var outArray = new VectorOfByte())
            {
                CvInvoke.Normalize(array, outArray);
                a = outArray.ToArray();
            }

            // Not sure this is the right algorithm as it doesn't compute a normal array
            //int i;
            //byte mu = mean_array(a, n);
            //var sigma = Math.Sqrt(variance_array(a, n));
            //for (i = 0; i < a.Length; ++i)
            //{
            //    a[i] = (byte)((a[i] - mu) / sigma);
            //}
        }

        public static byte mag_array(byte[] a, int n)
        {
            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                sum += a[i] * a[i];
            }
            return (byte)Math.Sqrt(sum);
        }

        public static void scale_array(byte[] a, int n, byte s)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                a[i] *= s;
            }
        }

        public static int sample_array(byte[] a, int n)
        {
            float sum = sum_array(a);
            scale_array(a, n, (byte)(1.0f / sum));
            int r = rand_uniform(0, 1);
            int i;
            for (i = 0; i < n; ++i)
            {
                r = r - a[i];
                if (r <= 0) return i;
            }
            return n - 1;
        }

        public static float sum_array(byte[] a)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                sum += a[i];
            }

            return sum;
        }

        public static int max_index(byte[] a, int num)
        {
            if (!a.Any()) return -1;
            int maxI = 0;
            byte max = a[0];
            for (var i = 1; i < num; ++i)
            {
                if (a[i] > max)
                {
                    max = a[i];
                    maxI = i;
                }
            }
            return maxI;
        }

        public static Random Rand = new Random(DateTime.Now.Millisecond);
        private static int haveSpare;
        private static double rand1, rand2;

        public static int rand_int(int min, int max)
        {
            if (max < min)
            {
                int s = min;
                min = max;
                max = s;
            }

            return Rand.Next(min, max);
        }

        public static byte rand_normal()
        {
            if (haveSpare != 0)
            {
                haveSpare = 0;
                return (byte)(Math.Sqrt(rand1) * Math.Sin(rand2));
            }

            haveSpare = 1;

            rand1 = Rand.NextDouble() / double.MaxValue;
            if (rand1 < 1e-100) rand1 = 1e-100;
            rand1 = -2 * Math.Log(rand1);
            rand2 = (Rand.NextDouble() / double.MaxValue) * Math.PI * 2;

            return (byte)(Math.Sqrt(rand1) * Math.Cos(rand2));
        }

        public static byte rand_uniform(byte min, byte max)
        {
            if (max < min)
            {
                byte swap = min;
                min = max;
                max = swap;
            }
            return (byte)(((byte)Rand.Next() / byte.MaxValue * (byte)(max - min)) + min);
        }

        public static byte rand_scale(byte s)
        {
            byte scale = rand_uniform(1, s);
            if (Rand.Next() % 2 == 1) return scale;
            return (byte)(1.0f / scale);
        }
    }
}
