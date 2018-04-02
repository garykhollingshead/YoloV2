namespace Yolo_V2.Data
{
    public class SortableBbox
    {
        public int index;
        public int sclass;
        public float[][] probs;
        public Network net;
        public string filename;
        public int classes;
        public float elo;
        public float[] elos;
    }
}