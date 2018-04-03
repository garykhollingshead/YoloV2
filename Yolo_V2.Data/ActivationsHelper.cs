using System;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public static class ActivationsHelper
    {
        public static string Get_activation_string(Activation a)
        {
            return a.ToString();
        }

        public static Activation Get_activation(string s)
        {
            return (Activation)Enum.Parse(typeof(Activation), s);
        }

        public static float Activate(float x, Activation a)
        {
            switch (a)
            {
                case Activation.Linear:
                    return Linear_activate(x);
                case Activation.Logistic:
                    return Logistic_activate(x);
                case Activation.Loggy:
                    return Loggy_activate(x);
                case Activation.Relu:
                    return Relu_activate(x);
                case Activation.Elu:
                    return Elu_activate(x);
                case Activation.Relie:
                    return Relie_activate(x);
                case Activation.Ramp:
                    return Ramp_activate(x);
                case Activation.Leaky:
                    return Leaky_activate(x);
                case Activation.Tanh:
                    return Tanh_activate(x);
                case Activation.Plse:
                    return Plse_activate(x);
                case Activation.Stair:
                    return Stair_activate(x);
                case Activation.Hardtan:
                    return Hardtan_activate(x);
                case Activation.Lhtan:
                    return Lhtan_activate(x);
            }
            return 0;
        }

        public static void Activate_array(float[] x, int n, Activation a)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                x[i] = Activate(x[i], a);
            }
        }

        public static float Gradient(float x, Activation a)
        {
            switch (a)
            {
                case Activation.Linear:
                    return Linear_gradient(x);
                case Activation.Logistic:
                    return Logistic_gradient(x);
                case Activation.Loggy:
                    return Loggy_gradient(x);
                case Activation.Relu:
                    return Relu_gradient(x);
                case Activation.Elu:
                    return Elu_gradient(x);
                case Activation.Relie:
                    return Relie_gradient(x);
                case Activation.Ramp:
                    return Ramp_gradient(x);
                case Activation.Leaky:
                    return Leaky_gradient(x);
                case Activation.Tanh:
                    return Tanh_gradient(x);
                case Activation.Plse:
                    return Plse_gradient(x);
                case Activation.Stair:
                    return Stair_gradient(x);
                case Activation.Hardtan:
                    return Hardtan_gradient(x);
                case Activation.Lhtan:
                    return Lhtan_gradient(x);
            }
            return 0;
        }

        public static void Gradient_array(float[] x, int n, Activation a, float[] delta)
        {
            int i;
            for (i = 0; i < n; ++i)
            {
                delta[i] *= Gradient(x[i], a);
            }
        }

        public static float Stair_activate(float x)
        {
            int n = (int)Math.Floor(x);
            if (n % 2 == 0)
            {
                return (float)Math.Floor(x / 2.0);
            }

            return (x - n) + (float)Math.Floor(x / 2.0);
        }

        public static float Hardtan_activate(float x)
        {
            if (x < -1) return -1;
            return x > 1 ? 1 : x;
        }

        public static float Linear_activate(float x)
        {
            return x;
        }

        public static float Logistic_activate(float x)
        {
            return 1.0f/ (1.0f + (float)Math.Exp(-x));
        }

        public static float Loggy_activate(float x)
        {
            return 2.0f/ (1.0f + (float)Math.Exp(-x)) - 1;
        }

        public static float Relu_activate(float x)
        {
            return (x > 0) ? x : 0;
        }

        public static float Elu_activate(float x)
        {
            return (x >= 0) ? x - 1 : (float)Math.Exp(x) - 1;
        }

        public static float Relie_activate(float x)
        {
            return (x > 0) ? x : 0.01f * x;
        }

        public static float Ramp_activate(float x)
        {
            return (x > 0) ? x + 0.1f * x :  0.1f * x;
        }

        public static float Leaky_activate(float x)
        {
            return (x > 0) ? x : 0.1f * x;
        }

        public static float Tanh_activate(float x)
        {
            return ((float)Math.Exp(2 * x) - 1) / ((float)Math.Exp(2 * x) + 1);
        }

        public static float Plse_activate(float x)
        {
            if (x < -4) return .01f * (x + 4);
            if (x > 4) return .01f * (x - 4) + 1;
            return 0.125f * x + 0.5f;
        }

        public static float Lhtan_activate(float x)
        {
            if (x < 0) return 0.001f * x;
            if (x > 1) return 0.001f * (x - 1) + 1;
            return x;
        }

        public static float Lhtan_gradient(float x)
        {
            if (x > 0 && x < 1) return 1;
            return 0.001f;
        }

        public static float Hardtan_gradient(float x)
        {
            if (x > -1 && x < 1) return 1;
            return 0;
        }

        public static float Linear_gradient(float x)
        {
            return 1;
        }

        public static float Logistic_gradient(float x)
        {
            return (1 - x) * x;
        }

        public static float Loggy_gradient(float x)
        {
            float y = (x + 1.0f) / 2.0f;
            return 2 * (1 - y) * y;
        }

        public static float Stair_gradient(float x)
        {
            if (Math.Floor(x) == x) return 0;
            return 1;
        }

        public static float Relu_gradient(float x)
        {
            return (x > 0) ? 1 : 0;
        }

        public static float Elu_gradient(float x)
        {
            return (x >= 0) ? 1 : (x + 1);
        }

        public static float Relie_gradient(float x)
        {
            return (x > 0) ? 1 : .01f;
        }

        public static float Ramp_gradient(float x)
        {
            return (x > 0) ? 1.1f : .1f;
        }

        public static float Leaky_gradient(float x)
        {
            return (x > 0) ? 1 : .1f;
        }

        public static float Tanh_gradient(float x)
        {
            return 1 - x * x;
        }

        public static float Plse_gradient(float x)
        {
            return (x < 0 || x > 1) ? .01f : .125f;
        }

        private static void activate_array_kernel(float[] x, int n, Activation a)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) x[i] = Activate(x[i], a);
        }

        private static void gradient_array_kernel(float[] x, int n, Activation a, float[] delta)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) delta[i] *= Gradient(x[i], a);
        }

        [GpuManaged]
        public static void activate_array_ongpu(float[] x, int n, Activation a)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(activate_array_kernel, lp, x, n, a);
        }

        [GpuManaged]
        public static void gradient_array_ongpu(float[] x, int n, Activation a, float[] delta)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(gradient_array_kernel, lp, x, n, a, delta);
        }
    }
}
