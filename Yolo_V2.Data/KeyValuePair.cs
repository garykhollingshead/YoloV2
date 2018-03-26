namespace Yolo_V2.Data
{
    public class KeyValuePair
    {
        public string Key;
        public string Val;
        public bool Used;

        public KeyValuePair(string key = "", string val = "", bool used = true)
        {
            Key = key;
            Val = val;
            Used = used;
        }
    }
}