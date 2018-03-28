using System;
using System.Collections.Generic;
using System.Linq;

namespace Yolo_V2.Data
{
    public class Section
    {
        public string Type;
        public KeyValuePair[] Options;

        public Section(string type = null)
        {
            Type = type;
            Options = new KeyValuePair[0];
        }

        public Section(Section s)
        {
            Type = s.Type;
            Options = s.Options.ToArray();
        }
    }
}