using System.Collections.Generic;

namespace Yolo_V2.Data
{
    public class NetworkState
    {
        public float[] Truth;
        public int InputIndex;
        public float[] InputBackup;
        public float[] Input;
        public int DeltaIndex;
        public float[] DeltaBackup;
        public float[] Delta;
        public float[] Workspace;
        public bool Train;
        public int Index;
        public Network Net;
    }
}