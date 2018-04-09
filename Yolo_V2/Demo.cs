using System;
using System.Diagnostics;
using System.Threading;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Demo
    {
        private static int frames = 3;

        private static string[] demoNames;
        private static int demoClasses;

        private static float[][] probs;
        private static Box[] boxes;
        private static Network net;
        private static Image nextImage;
        private static object resizedImageLock = new object();
        private static Image resizedImage;
        static AutoResetEvent capturedImageEvent = new AutoResetEvent(false);
        private static Image detectedImage;
        private static object displayImageLock = new object();
        private static Image displayImage = new Image();
        private static VideoCapture vidCap;
        private static float fps;
        private static float demoThresh;

        private static float[][] predictions = new float[frames][];
        private static int demoIndex;
        private static Image[] images = new Image[frames];
        private static float[] avg;

        private static void fetch_in_thread()
        {
            while (true)
            {
                nextImage = LoadArgs.get_image_from_stream(vidCap);
                if (nextImage.Data.Length == 0)
                {
                    Utils.Error("Stream closed.");
                }
                var tempImage = LoadArgs.resize_image(nextImage, net.W, net.H);
                lock (resizedImageLock)
                {
                    resizedImage = tempImage;
                }

                capturedImageEvent.Set();
            }
        }

        private static void detect_in_thread()
        {
            float nms = .4f;

            Layer l = net.Layers[net.N - 1];
            lock (resizedImageLock)
            {
                detectedImage = new Image(resizedImage);
            }

            var x = detectedImage.Data;
            float[] prediction = Network.network_predict(ref net, ref x);

            Array.Copy(prediction, 0, predictions[demoIndex], 0, l.Outputs);
            Utils.mean_arrays(predictions, frames, l.Outputs, avg);
            l.Output = avg;

            if (l.LayerType == Layers.Detection)
            {
                l.get_detection_boxes(1, 1, demoThresh, ref probs, ref boxes, false);
            }
            else if (l.LayerType == Layers.Region)
            {
                l.get_region_boxes(1, 1, demoThresh, ref probs, ref boxes, false, new int[0]);
            }
            else
            {
                Utils.Error("Last Layer must produce detections\n");
            }
            if (nms > 0) Box.do_nms(boxes, probs, l.Width * l.Height * l.N, l.Classes, nms);
            Console.Clear();
            Console.SetCursorPosition(1, 1);
            Console.Write($"\nFPS:{fps:F1}\n", fps);
            Console.Write($"Objects:\n\n");

            LoadArgs.draw_detections(ref detectedImage, l.Width * l.Height * l.N, demoThresh, boxes, probs, demoNames, demoClasses);
            lock (displayImageLock)
            {
                displayImage = detectedImage;
            }
        }

        public static void DemoRun(string cfgfile, string weightfile, float thresh, int camIndex, string filename, string[] names, int classes, int frameSkip, string prefix)
        {
            //skip = frame_skip;
            int delay = frameSkip;
            demoNames = names;
            demoClasses = classes;
            demoThresh = thresh;
            Console.Write($"Demo\n");
            net = Parser.parse_network_cfg(cfgfile);
            if (!string.IsNullOrEmpty(weightfile))
            {
                Parser.load_weights(net, weightfile);
            }
            Network.set_batch_network(ref net, 1);

            if (!string.IsNullOrEmpty(filename))
            {
                Console.Write($"video file: %s\n", filename);
            }

            using (var capture = !string.IsNullOrEmpty(filename)
                ? new VideoCapture(filename)
                : new VideoCapture(camIndex))
            {
                vidCap = capture;

                CvInvoke.NamedWindow("Demo", NamedWindowType.Normal);

                if (!vidCap.IsOpened) Utils.Error("Couldn't connect to webcam.\n");

                Layer l = net.Layers[net.N - 1];
                int j;

                avg = new float[l.Outputs];
                for (j = 0; j < frames; ++j) predictions[j] = new float[l.Outputs];
                for (j = 0; j < frames; ++j) images[j] = new Image(1, 1, 3);

                boxes = new Box[l.Width * l.Height * l.N];
                probs = new float[l.Width * l.Height * l.N][];
                for (j = 0; j < l.Width * l.Height * l.N; ++j) probs[j] = new float[l.Classes];

                var fetchThread = new Thread(fetch_in_thread);
                fetchThread.Start();
                capturedImageEvent.WaitOne();
                int count = 0;
                while (true)
                {
                    ++count;
                    var detectThread = new Thread(detect_in_thread);
                    detectThread.Start();

                    detectThread.Join();

                    Image disp;
                    lock (displayImageLock)
                    {
                        disp = new Image(displayImage);
                    }

                    if (string.IsNullOrEmpty(prefix))
                    {
                        LoadArgs.show_image(disp, "Demo");
                    }
                    else
                    {
                        var buff = $"{prefix}_{count:D8}";
                        LoadArgs.save_image(disp, buff);
                    }
                }
            }

            vidCap = null;
        }
    }
}