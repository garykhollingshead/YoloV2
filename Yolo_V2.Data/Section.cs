using System.Collections.Generic;

namespace Yolo_V2.Data
{
    public class Section
    {
        public string Type;
        public List<KeyValuePair> Options;

        public Section(string type = null)
        {
            Type = type;
            Options = new List<KeyValuePair>();
        }

        public Section(Section s)
        {
            Type = s.Type;
            Options = new List<KeyValuePair>(s.Options);
        }
    }
}