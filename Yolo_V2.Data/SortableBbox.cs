namespace Yolo_V2.Data
{
    public class SortableBbox
    {
        public int Index;
        public int Sclass;
        public float[][] Probs;
        public Network Net;
        public string Filename;
        public int Classes;
        public float Elo;
        public float[] Elos;
    }
}