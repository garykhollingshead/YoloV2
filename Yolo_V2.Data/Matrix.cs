﻿using System;
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
                    Vals[i][j] *= scale;
                }
            }
        }

        public void resize_matrix(int size)
        {
            int i;
            if (Rows == size) return;

            Array.Resize(ref Vals, size);
            for (i = Rows; i < size; ++i)
            {
                Array.Resize(ref Vals[i], Cols);
            }
            Rows = size;
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
            Vals = new float[Rows][];
            for (var i = 0; i < Rows; ++i)
            {
                Vals[i] = new float[Cols];
            }
        }

        public Matrix hold_out_matrix(int n)
        {
            int i;
            Matrix h = new Matrix();
            h.Rows = n;
            h.Cols = Cols;
            Vals = new float[Rows][];
            for (i = 0; i < n; ++i)
            {
                int index = Utils.Rand.Next() % Rows;
                h.Vals[i] = Vals[index];
                Vals[index] = Vals[--(Rows)];
            }
            return h;
        }

        public float[] pop_column(int c)
        {
            float[] col = new float[Rows];
            int i, j;
            for (i = 0; i < Rows; ++i)
            {
                col[i] = Vals[i][c];
                for (j = c; j < Cols - 1; ++j)
                {
                    Vals[i][j] = Vals[i][j + 1];
                }
            }
            --Cols;
            return col;
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
            Vals = new float[size][]; 
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
                    lines.Append($"{Vals[i][j]:17g}");
                }

                lines.AppendLine();
            }
            Console.Write(lines);
        }

        public void print_matrix()
        {
            int i, j;
            Console.Write($"{Rows} X {Cols} Matrix:\n");
            Console.Write(" __");
            for (j = 0; j < 16 * Cols - 1; ++j) Console.Write(" ");
            Console.Write("__ \n");

            Console.Write("|  ");
            for (j = 0; j < 16 * Cols - 1; ++j) Console.Write(" ");
            Console.Write("  |\n");

            for (i = 0; i < Rows; ++i)
            {
                Console.Write("|  ");
                for (j = 0; j < Cols; ++j)
                {
                    Console.Write($"{Vals[i][j]:15.7} ", Vals[i][j]);
                }
                Console.Write(" |\n");
            }
            Console.Write("|__");
            for (j = 0; j < 16 * Cols - 1; ++j) Console.Write(" ");
            Console.Write("__|\n");
        }
    }
}