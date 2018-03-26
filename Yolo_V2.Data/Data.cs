using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Yolo_V2.Data
{
    public class Data
    {
        public int W;
        public int H;
        public Matrix X;
        public Matrix Y;
        public int Shallow;
        public List<int> NumBoxes;
        public Box[][] Boxes;

        public static List<string> GetPaths(string filename)
        {
            return !File.Exists(filename) 
                ? new List<string>() 
                : File.ReadAllLines(filename).ToList();
        }
    }
}