using System;
using System.Collections.Generic;
using System.IO;

namespace Yolo_V2.Data
{
    public static class OptionList
    {
        public static List<KeyValuePair> read_data_cfg(string filename)
        {
            if (!File.Exists(filename))
            {
                Utils.file_error(filename);
            }

            var lines = File.ReadAllLines(filename);
            var options = new List<KeyValuePair>();
            for (var i = 0; i < lines.Length; ++i)
            {
                lines[i] = Utils.Strip(lines[i]);
                switch (lines[i][0])
                {
                    case '\0':
                    case '#':
                    case ';':
                        break;
                    default:
                        if (!read_option(lines[i], options))
                        {
                            Console.Error.WriteLine($"Config file error line {i}, could parse: {lines[i]}");
                        }
                        break;
                }
            }
            return options;
        }

        public static bool read_option(string s, List<KeyValuePair> options)
        {
            var parts = s.Split('=');
            if (parts.Length != 2)
            {
                return false;
            }

            options.Add(new KeyValuePair(parts[0], parts[1]));
            return true;
        }

        public static void option_unused(KeyValuePair[] l)
        {
            foreach (var kvp in l)
            {
                if (!kvp.Used)
                {
                    Console.Error.WriteLine($"Unused field: '{kvp.Key} = {kvp.Val}'");
                }
            }
        }

        public static string option_find(KeyValuePair[] l, string key)
        {
            foreach (var kvp in l)
            {
                if (kvp.Key == key)
                {
                    kvp.Used = true;
                    return kvp.Val;
                }
            }

            return null;
        }

        public static string option_find_str(KeyValuePair[] l, string key, string def)
        {
            string v = option_find(l, key);
            if (!string.IsNullOrEmpty(v)) return v;
            if (!string.IsNullOrEmpty(def)) Console.Error.WriteLine($"{key}: Using default '{def}'");
            return def;
        }

        public static int option_find_int(KeyValuePair[] l, string key, int def)
        {
            string v = option_find(l, key);
            try
            {
                if (!string.IsNullOrEmpty(v)) return int.Parse(v);
            }
            catch
            {
                Console.Error.WriteLine($"{key}: Using default '{def}'");
            }
            return def;
        }

        public static int option_find_int_quiet(KeyValuePair[] l, string key, int def)
        {
            string v = option_find(l, key);
            try
            {
                if (!string.IsNullOrEmpty(v)) return int.Parse(v);
            }
            catch
            {
            }
            return def;
        }

        public static float option_find_float_quiet(KeyValuePair[] l, string key, float def)
        {
            string v = option_find(l, key);
            try
            {
                if (!string.IsNullOrEmpty(v)) return float.Parse(v);
            }
            catch
            {
            }
            return def;
        }

        public static float option_find_float(KeyValuePair[] l, string key, float def)
        {
            string v = option_find(l, key);
            try
            {
                if (!string.IsNullOrEmpty(v)) return float.Parse(v);
            }
            catch
            {
                Console.Error.WriteLine($"{key}: Using default '{def}'");
            }
            return def;
        }
    }
}