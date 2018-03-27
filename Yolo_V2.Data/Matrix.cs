using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Yolo_V2.Data
{
    public class Matrix
    {
        public int Rows;
        public int Cols;
        public float[][] Vals;

        public float matrix_topk_accuracy(Matrix truth, Matrix guess, int k)
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

        public void scale_matrix(Matrix m, float scale)
        {
            int i, j;
            for (i = 0; i < m.Rows; ++i)
            {
                for (j = 0; j < m.Cols; ++j)
                {
                    m.Vals[i][j] *= scale;
                }
            }
        }

        public Matrix resize_matrix(Matrix m, int size)
        {
            int i;
            if (m.Rows == size) return m;

            Array.Resize(ref m.Vals, size);
            for (i = m.Rows; i < size; ++i)
            {
                Array.Resize(ref m.Vals[i], m.Cols);
            }
            m.Rows = size;
            return m;
        }

        public void matrix_add_matrix(Matrix from, Matrix to)
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
            Vals = new float[Rows][];
            for (var i = 0; i < Rows; ++i)
            {
                Vals[i] = new float[Cols];
            }
        }

        public Matrix hold_out_matrix(Matrix m, int n)
        {
            int i;
            Matrix h = new Matrix();
            h.Rows = n;
            h.Cols = m.Cols;
            m.Vals = new float[m.Rows][];
            for (i = 0; i < n; ++i)
            {
                int index = Utils.Rand.Next() % m.Rows;
                h.Vals[i] = m.Vals[index];
                m.Vals[index] = m.Vals[--(m.Rows)];
            }
            return h;
        }

        public float[] pop_column(Matrix m, int c)
        {
            float[] col = new float[m.Rows];
            int i, j;
            for (i = 0; i < m.Rows; ++i)
            {
                col[i] = m.Vals[i][c];
                for (j = c; j < m.Cols - 1; ++j)
                {
                    m.Vals[i][j] = m.Vals[i][j + 1];
                }
            }
            --m.Cols;
            return col;
        }

        public Matrix csv_to_matrix(string filename)
        {
            if (!File.Exists(filename))
            {
                Utils.file_error(filename);
            }


            Matrix m = new Matrix();
            m.Cols = -1;
            
            int n = 0;
            int size = 1024;
            m.Vals = new float[size][]; 
            foreach (var line in File.ReadAllLines(filename))
            {
                if (m.Cols == -1) m.Cols = Utils.count_fields(line);
                if (n == size)
                {
                    size *= 2;
                    Array.Resize(ref m.Vals, size);
                }
                m.Vals[n] = Utils.parse_fields(line);
                ++n;
            }
            Array.Resize(ref m.Vals, n);
            m.Rows = n;
            return m;
        }

        public void matrix_to_csv(Matrix m)
        {
            int i, j;
            var lines = new StringBuilder();
            for (i = 0; i < m.Rows; ++i)
            {
                for (j = 0; j < m.Cols; ++j)
                {
                    if (j > 0) lines.Append(",");
                    lines.Append($"{m.Vals[i][j]:17g}");
                }

                lines.AppendLine();
            }
            Console.Write(lines);
        }

        public void print_matrix(Matrix m)
        {
            int i, j;
            Console.Write($"{m.Rows} X {m.Cols} Matrix:\n");
            Console.Write(" __");
            for (j = 0; j < 16 * m.Cols - 1; ++j) Console.Write(" ");
            Console.Write("__ \n");

            Console.Write("|  ");
            for (j = 0; j < 16 * m.Cols - 1; ++j) Console.Write(" ");
            Console.Write("  |\n");

            for (i = 0; i < m.Rows; ++i)
            {
                Console.Write("|  ");
                for (j = 0; j < m.Cols; ++j)
                {
                    Console.Write($"{m.Vals[i][j]:15.7} ", m.Vals[i][j]);
                }
                Console.Write(" |\n");
            }
            Console.Write("|__");
            for (j = 0; j < 16 * m.Cols - 1; ++j) Console.Write(" ");
            Console.Write("__|\n");
        }
    }
}