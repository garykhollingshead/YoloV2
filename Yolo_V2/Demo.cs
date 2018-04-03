using System;
using System.Diagnostics;
using System.IO.Ports;
using System.Threading;
using Emgu.CV;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public class Demo
    {

        public static int FRAMES = 3;

        static string[] demo_names;
        static Image[][] demo_alphabet;
        static int demo_classes;

        static float[][] probs;
        static Box[] boxes;
        static Network net;
        static Image inputImage;
        static Image in_s;
        static Image det;
        static Image det_s;
        static Image disp = new Image();
        static VideoCapture vidCap = null;
        static float fps = 0;
        static float demo_thresh = 0;

        static float[][] predictions = new float[FRAMES][];
        static int demo_index = 0;
        static Image[] images = new Image[FRAMES];
        static float[] avg;

        private static void fetch_in_thread()
        {
            inputImage = LoadArgs.get_image_from_stream(vidCap);
            if (inputImage.Data.Length == 0)
            {
                Utils.Error("Stream closed.");
            }
            in_s = LoadArgs.resize_image(inputImage, net.W, net.H);
        }

        private static void detect_in_thread()
        {
            float nms = .4f;

            Layer l = net.Layers[net.N - 1];
            float[] X = det_s.Data;
            float[] prediction = Network.network_predict(net, X);

            Array.Copy(prediction, 0, prediction, demo_index, l.Outputs);
            Utils.mean_arrays(predictions, FRAMES, l.Outputs, avg);
            l.Output = avg;

            if (l.LayerType == LayerType.Detection)
            {
                l.get_detection_boxes(1, 1, demo_thresh, probs, boxes, false);
            }
            else if (l.LayerType == LayerType.Region)
            {
                Layer.get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, false, new int[0]);
            }
            else
            {
                Utils.Error("Last Layer must produce detections\n");
            }
            if (nms > 0) Box.do_nms(boxes, probs, l.W * l.H * l.N, l.Classes, nms);
            Console.Write($"\033[2J");
            Console.Write($"\033[1;1H");
            Console.Write($"\nFPS:%.1f\n", fps);
            Console.Write($"Objects:\n\n");

            images[demo_index] = det;
            det = images[(demo_index + FRAMES / 2 + 1) % FRAMES];
            demo_index = (demo_index + 1) % FRAMES;

            LoadArgs.draw_detections(det, l.W * l.H * l.N, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
        }

        public static void demo(string cfgfile, string weightfile, float thresh, int cam_index, string filename, string[] names, int classes, int frame_skip, string prefix)
        {
            //skip = frame_skip;
            Image[][] alphabet = LoadArgs.load_alphabet();
            int delay = frame_skip;
            demo_names = names;
            demo_alphabet = alphabet;
            demo_classes = classes;
            demo_thresh = thresh;
            Console.Write($"Demo\n");
            net = Parser.parse_network_cfg(cfgfile);
            if (!string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(net, 1);

            if (!string.IsNullOrEmpty(filename))
            {
                Console.Write($"video file: %s\n", filename);
            }

            using (var capture = !string.IsNullOrEmpty(filename)
                ? new VideoCapture(filename)
                : new VideoCapture(cam_index))
            {
                vidCap = capture;
                if (!vidCap.IsOpened) Utils.Error("Couldn't connect to webcam.\n");

                Layer l = net.Layers[net.N - 1];
                int j;

                avg = new float[l.Outputs];
                for (j = 0; j < FRAMES; ++j) predictions[j] = new float[l.Outputs];
                for (j = 0; j < FRAMES; ++j) images[j] = new Image(1, 1, 3);

                boxes = new Box[l.W * l.H * l.N];
                probs = new float[l.W * l.H * l.N][];
                for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = new float[l.Classes];

                Thread fetch_thread;
                Thread detect_thread;

                fetch_in_thread();
                det = inputImage;
                det_s = in_s;

                fetch_in_thread();
                detect_in_thread();
                disp = det;
                det = inputImage;
                det_s = in_s;

                for (j = 0; j < FRAMES / 2; ++j)
                {
                    fetch_in_thread();
                    detect_in_thread();
                    disp = det;
                    det = inputImage;
                    det_s = in_s;
                }

                int count = 0;
                var sw = new Stopwatch();
                sw.Stop();

                while (true)
                {
                    ++count;
                    fetch_thread = new Thread(fetch_in_thread);
                    detect_thread = new Thread(detect_in_thread);
                    fetch_thread.Start();
                    detect_thread.Start();

                    if (string.IsNullOrEmpty(prefix))
                    {
                        LoadArgs.show_image(disp, "Demo");
                        int c = CvInvoke.WaitKey(1);
                        if (c == 10)
                        {
                            if (frame_skip == 0) frame_skip = 60;
                            else if (frame_skip == 4) frame_skip = 0;
                            else if (frame_skip == 60) frame_skip = 4;
                            else frame_skip = 0;
                        }
                    }
                    else
                    {
                        var buff = $"{prefix}_{count:08}";
                        LoadArgs.save_image(disp, buff);
                    }

                    fetch_thread.Join();
                    detect_thread.Join();

                    if (delay == 0)
                    {
                        disp = det;
                    }

                    det = inputImage;
                    det_s = in_s;
                    --delay;
                    if (delay < 0)
                    {
                        delay = frame_skip;

                        sw.Stop();
                        float curr = 1f/ sw.Elapsed.Seconds;
                        fps = curr;
                        sw.Reset();
                        sw.Start();
                    }
                }
            }

            vidCap = null;
        }
    }
}