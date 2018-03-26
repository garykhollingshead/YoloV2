using System.Collections.Generic;

namespace Yolo_V2.Data
{
    public class NetworkState
    {
        public float[] Truth;
        public float[] Input;
        public float[] Delta;
        public float[] Workspace;
        public int Train;
        public int Index;
        public Network Net;
    }
}