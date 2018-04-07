using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Yolo_V2.Data
{
    public class Tree
    {
        private List<int> Leaf = new List<int>();
        private int N;
        public List<int> Parent = new List<int>();
        public List<int> Group = new List<int>();
        private List<string> Name = new List<string>();

        public int Groups;
        public List<int> GroupSize = new List<int>();
        public List<int> GroupOffset = new List<int>();

        public Tree(string filename)
        {
            if (!File.Exists(filename))
            {
                return;
            }
            var lastParent = -1;
            int groupSize = 0;
            int groups = 0;
            int n = 0;
            foreach (var line in File.ReadAllLines(filename))
            {
                var parts = line.Split(' ');
                var id = parts[0];
                var parent = int.Parse(parts[1]);
                Parent.Add(parent);
                Name.Add(id);

                if (parent != lastParent)
                {
                    ++groups;

                    GroupOffset.Add(n-groupSize);
                    GroupSize.Add(groupSize);

                    groupSize = 0;
                    lastParent = parent;
                }
                Group.Add(groups);
                ++n;
                ++groupSize;
            }
            ++groups;
            GroupOffset.Add(n - groupSize);
            GroupSize.Add(groupSize);
            N = n;
            Groups = groups;

            int i;
            for (i = 0; i < n; ++i) Leaf.Add(1);
            for (i = 0; i < n; ++i) if (Parent[i] >= 0) Leaf[Parent[i]] = 0;
        }

        public void Hierarchy_predictions(byte[] predictions, int predStart, int n, bool onlyLeaves)
        {
            int j;
            for (j = 0; j < n; ++j)
            {
                int parent = Parent[j];
                if (parent >= 0)
                {
                    predictions[predStart + j] *= predictions[predStart + parent];
                }
            }
            if (onlyLeaves)
            {
                for (j = 0; j < n; ++j)
                {
                    if (Leaf[j] != 0) predictions[predStart + j] = 0;
                }
            }
        }

        public void Change_leaves(string leafList)
        {
            var leaves = Data.GetPaths(leafList);
            var found = 0;
            for (var i = 0; i < N; ++i)
            {
                Leaf[i] = 0;
                if (leaves.Any(t => Name[i] == t))
                {
                    Leaf[i] = 1;
                    ++found;
                }
            }
            Console.WriteLine($"Found {found} leaves.");
    }

        public float Get_hierarchy_probability(float[] x, int xStart, int c)
        {
            float p = 1;
            while (c >= 0)
            {
                p = p * x[c + xStart];
                c = Parent[c];
            }
            return p;
        }
}
}