namespace Yolo_V2.Data
{
    public class DBox
    {
        public float Dx;
        public float Dy;
        public float Dw;
        public float Dh;

        public DBox() { }

        public DBox(float x, float y, float w, float h)
        {
            Dx = x;
            Dy = y;
            Dw = w;
            Dh = h;
        }
    }
}