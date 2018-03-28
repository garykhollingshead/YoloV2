using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Yolo_V2.Data
{
    public static class Utils
    {
        public static readonly int SecretNum = -1234;

        public static int[] read_map(string filename)
        {
            if (!File.Exists(filename))
            {
                file_error(filename);
            }

            return File.ReadAllLines(filename).Select(int.Parse).ToArray();
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

        public static void del_arg(List<string> argv, int index)
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

        public static T find_int_arg<T>(List<string> argv, string arg, T def)
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

        public static int alphanum_to_int(char c)
        {
            return (c < 58) ? c - 48 : c - 87;
        }

        public static char int_to_alphanum(int i)
        {
            if (i == 36) return '.';
            return (i < 10) ? (char)(i + 48) : (char)(i + 87);
        }

        public static void Pm(int m, int n, float[] a)
        {
            int i, j;
            for (i = 0; i < m; ++i)
            {
                Console.WriteLine($"{i + 1} ");
                for (j = 0; j < n; ++j)
                {
                    Console.WriteLine($"{a[i * n + j]:2.4}, ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine("");
        }

        public static void find_replace(string str, string orig, string rep, out string output)
        {
            output = str.Replace(orig, rep);
        }

        public static void top_k(float[] a, int n, int k, int[] index)
        {
            var temp = new Dictionary<float, int>();
            a.Select((f, ind) => temp[f] = ind);
            var ata = temp.Keys.OrderByDescending(f => f).Select(v => temp[v]).ToArray();
            for (var i = 0; i < k; ++i)
            {
                index[i] = ata[i];
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

        public static List<string> split_str(string s, char delim)
        {
            return s.Split(delim).ToList();
        }

        public static string Strip(string s)
        {
            var len = s.Length;
            for (var i = 0; i < len; ++i)
            {
                var c = s[i];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                {
                    s = s.Remove(i, i + 1);
                    len--;
                    i--;
                }
            }

            return s;
        }

        public static string strip_char(string s, char bad)
        {
            var len = s.Length;
            for (var i = 0; i < len; ++i)
            {
                if (bad == s[i])
                {
                    s = s.Remove(i, i + 1);
                    len--;
                    i--;
                }
            }

            return s;
        }

        public static List<string> parse_csv_line(string line)
        {
            var newLines = new List<string>();
            foreach (var li in line.Split('\"'))
            {
                newLines.AddRange(li.Split(','));
            }

            return newLines;
        }

        public static int count_fields(string line)
        {
            return line.Count(x => x == ',');
        }

        public static float[] parse_fields(string line)
        {
            var fields = new List<float>();
            foreach (var s in line.Split(','))
            {
                float t;
                t = float.TryParse(s, out t)
                    ? t
                    : float.NaN;
                fields.Add(t);
            }

            return fields.ToArray();
        }

        public static float mean_array(float[] a, int n)
        {
            return a.Take(n).Sum() / n;
        }

        public static void mean_arrays(float[][] a, float[] avg)
        {
            foreach (var f in a)
            {
                for (var i = 0; i < f.Length; ++i)
                {
                    avg[i] += f[i];
                }
            }

            for (var i = 0; i < avg.Length; ++i)
            {
                avg[i] /= a.Length;
            }
        }

        public static void print_statistics(float[] a, int n)
        {
            float m = mean_array(a, n);
            float v = variance_array(a, n);
            Console.WriteLine($"MSE: {mse_array(a):.6}, Mean: {m:.6}f, Variance: {v}f");
        }

        public static float variance_array(float[] a, int n)
        {
            int i;
            float sum = 0;
            float mean = mean_array(a, n);
            for (i = 0; i < n; ++i) sum += (a[i] - mean) * (a[i] - mean);
            float variance = sum / n;
            return variance;
        }

        public static int constrain_int(int a, int min, int max)
        {
            if (a < min) return min;
            if (a > max) return max;
            return a;
        }

        public static float Constrain(float min, float max, float a)
        {
            if (a < min) return min;
            if (a > max) return max;
            return a;
        }

        public static float dist_array(float[] a, float[] b, int n, int sub)
        {
            int i;
            double sum = 0;
            for (i = 0; i < n; i += sub) sum += Math.Pow(a[i] - b[i], 2);
            return (float)Math.Sqrt(sum);
        }

        public static float mse_array(float[] a)
        {
            int i;
            float sum = 0;
            for (i = 0; i < a.Length; ++i) sum += a[i] * a[i];
            return (float)Math.Sqrt(sum / a.Length);
        }

        public static void normalize_array(float[] a, int n)
        {
            int i;
            float mu = mean_array(a, n);
            var sigma = Math.Sqrt(variance_array(a, n));
            for (i = 0; i < a.Length; ++i)
            {
                a[i] = (float)((a[i] - mu) / sigma);
            }
        }

        public static void translate_array(float[] a, int n, float s)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                a[i] += s;
            }
        }

        public static float mag_array(float[] a, int n)
        {
            int i;
            float sum = 0;
            for (i = 0; i < n; ++i)
            {
                sum += a[i] * a[i];
            }
            return (float)Math.Sqrt(sum);
        }

        public static void scale_array(float[] a, int n, float s)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                a[i] *= s;
            }
        }

        public static int sample_array(float[] a, int n)
        {
            float sum = a.Sum();
            scale_array(a, n, 1.0f / sum);
            float r = rand_uniform(0, 1);
            int i;
            for (i = 0; i < n; ++i)
            {
                r = r - a[i];
                if (r <= 0) return i;
            }
            return n - 1;
        }

        public static int max_index(float[] a, int num)
        {
            if (!a.Any()) return -1;
            int maxI = 0;
            float max = a[0];
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
        private static int _haveSpare;
        private static double _rand1, _rand2;

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

        public static float rand_normal()
        {
            if (_haveSpare != 0)
            {
                _haveSpare = 0;
                return (float)(Math.Sqrt(_rand1) * Math.Sin(_rand2));
            }

            _haveSpare = 1;

            _rand1 = Rand.NextDouble() / double.MaxValue;
            if (_rand1 < 1e-100) _rand1 = 1e-100;
            _rand1 = -2 * Math.Log(_rand1);
            _rand2 = (Rand.NextDouble() / double.MaxValue) * Math.PI * 2;

            return (float)(Math.Sqrt(_rand1) * Math.Cos(_rand2));
        }

        public static ulong rand_size_t()
        {
            return ((ulong)(Rand.Next() & 0xff) << 56) |
                    ((ulong)(Rand.Next() & 0xff) << 48) |
                    ((ulong)(Rand.Next() & 0xff) << 40) |
                    ((ulong)(Rand.Next() & 0xff) << 32) |
                    ((ulong)(Rand.Next() & 0xff) << 24) |
                    ((ulong)(Rand.Next() & 0xff) << 16) |
                    ((ulong)(Rand.Next() & 0xff) << 8) |
                    ((ulong)(Rand.Next() & 0xff) << 0);
        }

        public static float rand_uniform(float min, float max)
        {
            if (max < min)
            {
                float swap = min;
                min = max;
                max = swap;
            }
            return (float)(Rand.NextDouble() / double.MaxValue * (max - min)) + min;
        }

        public static float rand_scale(float s)
        {
            float scale = rand_uniform(1, s);
            if (Rand.Next() % 2 == 1) return scale;
            return 1.0f / scale;
        }

        public static float[][] one_hot_encode(float[] a, int n, int k)
        {
            var t = new float[n][];

            for (var i = 0; i < n; ++i)
            {
                t[i] = new float[k];
                int index = (int)a[i];
                t[i][index] = 1;
            }
            return t;
        }
    }
}
