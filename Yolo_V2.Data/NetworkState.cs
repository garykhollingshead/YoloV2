namespace Yolo_V2.Data
{
    public class NetworkState
    {
        public byte[] Truth;
        public int InputIndex;
        public byte[] InputBackup;
        public byte[] Input;
        public int DeltaIndex;
        public byte[] DeltaBackup;
        public byte[] Delta;
        public float[] Workspace;
        public bool Train;
        public int Index;
        public Network Net;
    }
}