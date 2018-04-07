using System;
using System.IO;
using System.Text;

namespace Yolo_V2.Data
{
    public class Matrix
    {
        public int Rows;
        public int Cols;
        public byte[][] Vals;

        public static float matrix_topk_accuracy(Matrix truth, Matrix guess, int k)
        {
            int[] indexes = new int[k];
            int n = truth.Cols;
            int i, j;
            int correct = 0;
            for (i = 0; i < truth.Rows; ++i)
            {
                Utils.top_k(guess.Vals[i], n, k, indexes);
                for (j = 0; j < k; ++j)
                {
                    int classIndex = indexes[j];
                    if (truth.Vals.Length > i && truth.Vals[i].Length > classIndex)
                    {
                        ++correct;
                        break;
                    }
                }
            }
            return (float)correct / truth.Rows;
        }

        public void scale_matrix(float scale)
        {
            int i, j;
            for (i = 0; i < Rows; ++i)
            {
                for (j = 0; j < Cols; ++j)
                {
                    Vals[i][j] = (byte)(Vals[i][j] * scale);
                }
            }
        }

        public static void matrix_add_matrix(Matrix from, Matrix to)
        {
            int i, j;
            for (i = 0; i < from.Rows; ++i)
            {
                for (j = 0; j < from.Cols; ++j)
                {
                    to.Vals[i][j] += from.Vals[i][j];
                }
            }
        }

        public Matrix(int rows = 0, int cols = 0)
        {
            Rows = rows;
            Cols = cols;
            Vals = new byte[Rows][];
            for (var i = 0; i < Rows; ++i)
            {
                Vals[i] = new byte[Cols];
            }
        }

        public Matrix(string filename)
        {
            if (!File.Exists(filename))
            {
                Utils.file_error(filename);
            }

            Cols = -1;
            
            int n = 0;
            int size = 1024;
            Vals = new byte[size][]; 
            foreach (var line in File.ReadAllLines(filename))
            {
                if (Cols == -1) Cols = Utils.count_fields(line);
                if (n == size)
                {
                    size *= 2;
                    Array.Resize(ref Vals, size);
                }
                Vals[n] = Utils.parse_fields(line);
                ++n;
            }
            Array.Resize(ref Vals, n);
            Rows = n;
        }

        public void to_csv()
        {
            int i, j;
            var lines = new StringBuilder();
            for (i = 0; i < Rows; ++i)
            {
                for (j = 0; j < Cols; ++j)
                {
                    if (j > 0) lines.Append(",");
                    lines.Append($"{Vals[i][j]:F17}");
                }

                lines.AppendLine();
            }
            Console.Write(lines);
        }
    }
}