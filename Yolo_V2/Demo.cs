using System;
using System.Diagnostics;
using System.Threading;
using Emgu.CV;
using Yolo_V2.Data;
using Yolo_V2.Data.Enums;

namespace Yolo_V2
{
    public static class Demo
    {
        private static int frames = 3;

        private static string[] demoNames;
        private static Image[][] demoAlphabet;
        private static int demoClasses;
         
        private static float[][] probs;
        private static Box[] boxes;
        private static Network net;
        private static Image inputImage;
        private static Image inS;
        private static Image det;
        private static Image detS;
        private static Image disp = new Image();
        private static VideoCapture vidCap;
        private static float fps;
        private static float demoThresh;
         
        private static float[][] predictions = new float[frames][];
        private static int demoIndex;
        private static Image[] images = new Image[frames];
        private static float[] avg;

        private static void fetch_in_thread()
        {
            inputImage = LoadArgs.get_image_from_stream(vidCap);
            if (inputImage.Data.Length == 0)
            {
                Utils.Error("Stream closed.");
            }
            inS = LoadArgs.resize_image(inputImage, net.W, net.H);
        }

        private static void detect_in_thread()
        {
            float nms = .4f;

            Layer l = net.Layers[net.N - 1];
            float[] x = detS.Data;
            float[] prediction = Network.network_predict(net, x);

            Array.Copy(prediction, 0, prediction, demoIndex, l.Outputs);
            Utils.mean_arrays(predictions, frames, l.Outputs, avg);
            l.Output = avg;

            if (l.LayerType == LayerType.Detection)
            {
                l.get_detection_boxes(1, 1, demoThresh, probs, boxes, false);
            }
            else if (l.LayerType == LayerType.Region)
            {
                Layer.get_region_boxes(l, 1, 1, demoThresh, probs, boxes, false, new int[0]);
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

            images[demoIndex] = det;
            det = images[(demoIndex + frames / 2 + 1) % frames];
            demoIndex = (demoIndex + 1) % frames;

            LoadArgs.draw_detections(det, l.W * l.H * l.N, demoThresh, boxes, probs, demoNames, demoAlphabet, demoClasses);
        }

        public static void DemoRun(string cfgfile, string weightfile, float thresh, int camIndex, string filename, string[] names, int classes, int frameSkip, string prefix)
        {
            //skip = frame_skip;
            Image[][] alphabet = LoadArgs.load_alphabet();
            int delay = frameSkip;
            demoNames = names;
            demoAlphabet = alphabet;
            demoClasses = classes;
            demoThresh = thresh;
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
                : new VideoCapture(camIndex))
            {
                vidCap = capture;
                if (!vidCap.IsOpened) Utils.Error("Couldn't connect to webcam.\n");

                Layer l = net.Layers[net.N - 1];
                int j;

                avg = new float[l.Outputs];
                for (j = 0; j < frames; ++j) predictions[j] = new float[l.Outputs];
                for (j = 0; j < frames; ++j) images[j] = new Image(1, 1, 3);

                boxes = new Box[l.W * l.H * l.N];
                probs = new float[l.W * l.H * l.N][];
                for (j = 0; j < l.W * l.H * l.N; ++j) probs[j] = new float[l.Classes];

                Thread fetchThread;
                Thread detectThread;

                fetch_in_thread();
                det = inputImage;
                detS = inS;

                fetch_in_thread();
                detect_in_thread();
                disp = det;
                det = inputImage;
                detS = inS;

                for (j = 0; j < frames / 2; ++j)
                {
                    fetch_in_thread();
                    detect_in_thread();
                    disp = det;
                    det = inputImage;
                    detS = inS;
                }

                int count = 0;
                var sw = new Stopwatch();
                sw.Stop();

                while (true)
                {
                    ++count;
                    fetchThread = new Thread(fetch_in_thread);
                    detectThread = new Thread(detect_in_thread);
                    fetchThread.Start();
                    detectThread.Start();

                    if (string.IsNullOrEmpty(prefix))
                    {
                        LoadArgs.show_image(disp, "Demo");
                        int c = CvInvoke.WaitKey(1);
                        if (c == 10)
                        {
                            if (frameSkip == 0) frameSkip = 60;
                            else if (frameSkip == 4) frameSkip = 0;
                            else if (frameSkip == 60) frameSkip = 4;
                            else frameSkip = 0;
                        }
                    }
                    else
                    {
                        var buff = $"{prefix}_{count:08}";
                        LoadArgs.save_image(disp, buff);
                    }

                    fetchThread.Join();
                    detectThread.Join();

                    if (delay == 0)
                    {
                        disp = det;
                    }

                    det = inputImage;
                    detS = inS;
                    --delay;
                    if (delay < 0)
                    {
                        delay = frameSkip;

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