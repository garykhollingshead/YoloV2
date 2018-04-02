namespace Yolo_V2.Data
{
    public class DBox
    {
        public float DX;
        public float DY;
        public float DW;
        public float DH;

        public DBox() { }

        public DBox(float x, float y, float w, float h)
        {
            DX = x;
            DY = y;
            DW = w;
            DH = h;
        }
    }
}