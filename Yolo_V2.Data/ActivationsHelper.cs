using System;
using Alea;
using Alea.CSharp;
using Yolo_V2.Data.Enums;

namespace Yolo_V2.Data
{
    public static class ActivationsHelper
    {
        public static Activation Get_activation(string s)
        {
            s = char.ToUpper(s[0]) + s.Substring(1);
            return (Activation)Enum.Parse(typeof(Activation), s);
        }

        private static float Activate(float x, Activation a)
        {
            if (a == Activation.Linear)
                return Linear_activate(x);
            else if (a == Activation.Logistic)
            {
                return Logistic_activate(x);
            }
            else if (a == Activation.Loggy)
            {
                return Loggy_activate(x);
            }
            else if (a == Activation.Relu)
            {
                return Relu_activate(x);
            }
            else if (a == Activation.Elu)
            {
                return Elu_activate(x);
            }
            else if (a == Activation.Relie)
            {
                return Relie_activate(x);
            }
            else if (a == Activation.Ramp)
            {
                return Ramp_activate(x);
            }
            else if (a == Activation.Leaky)
            {
                return Leaky_activate(x);
            }
            else if (a == Activation.Tanh)
            {
                return Tanh_activate(x);
            }
            else if (a == Activation.Plse)
            {
                return Plse_activate(x);
            }
            else if (a == Activation.Stair)
            {
                return Stair_activate(x);
            }
            else if (a == Activation.Hardtan)
            {
                return Hardtan_activate(x);
            }
            else if (a == Activation.Lhtan)
            {
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

        private static float Gradient(float x, Activation a)
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

        private static float Stair_activate(float x)
        {
            int n = (int)Math.Floor(x);
            if (n % 2 == 0)
            {
                return (float)Math.Floor(x / 2.0);
            }

            return (x - n) + (float)Math.Floor(x / 2.0);
        }

        private static float Hardtan_activate(float x)
        {
            if (x < -1) return -1;
            return x > 1 ? 1 : x;
        }

        private static float Linear_activate(float x)
        {
            return x;
        }

        public static float Logistic_activate(float x)
        {
            return 1.0f/ (1.0f + (float)Math.Exp(-x));
        }

        private static float Loggy_activate(float x)
        {
            return 2.0f/ (1.0f + (float)Math.Exp(-x)) - 1;
        }

        private static float Relu_activate(float x)
        {
            return (x > 0) ? x : 0;
        }

        private static float Elu_activate(float x)
        {
            return (x >= 0) ? x - 1 : (float)Math.Exp(x) - 1;
        }

        private static float Relie_activate(float x)
        {
            return (x > 0) ? x : 0.01f * x;
        }

        private static float Ramp_activate(float x)
        {
            return (x > 0) ? x + 0.1f * x :  0.1f * x;
        }

        private static float Leaky_activate(float x)
        {
            return (x >= 0) ? x : 0.1f * x;
        }

        private static float Tanh_activate(float x)
        {
            return ((float)Math.Exp(2 * x) - 1) / ((float)Math.Exp(2 * x) + 1);
        }

        private static float Plse_activate(float x)
        {
            if (x < -4) return .01f * (x + 4);
            if (x > 4) return .01f * (x - 4) + 1;
            return 0.125f * x + 0.5f;
        }

        private static float Lhtan_activate(float x)
        {
            if (x < 0) return 0.001f * x;
            if (x > 1) return 0.001f * (x - 1) + 1;
            return x;
        }

        private static float Lhtan_gradient(float x)
        {
            if (x > 0 && x < 1) return 1;
            return 0.001f;
        }

        private static float Hardtan_gradient(float x)
        {
            if (x > -1 && x < 1) return 1;
            return 0;
        }

        private static float Linear_gradient(float x)
        {
            return 1;
        }

        public static float Logistic_gradient(float x)
        {
            return (1 - x) * x;
        }

        private static float Loggy_gradient(float x)
        {
            float y = (x + 1.0f) / 2.0f;
            return 2 * (1 - y) * y;
        }

        private static float Stair_gradient(float x)
        {
            if (Math.Floor(x) == x) return 0;
            return 1;
        }

        private static float Relu_gradient(float x)
        {
            return (x > 0) ? 1 : 0;
        }

        private static float Elu_gradient(float x)
        {
            return (x >= 0) ? 1 : (x + 1);
        }

        private static float Relie_gradient(float x)
        {
            return (x > 0) ? 1 : .01f;
        }

        private static float Ramp_gradient(float x)
        {
            return (x > 0) ? 1.1f : .1f;
        }

        private static float Leaky_gradient(float x)
        {
            return (x > 0) ? 1 : .1f;
        }

        private static float Tanh_gradient(float x)
        {
            return 1 - x * x;
        }

        private static float Plse_gradient(float x)
        {
            return (x < 0 || x > 1) ? .01f : .125f;
        }

        private static void activate_array_kernel(float[] x, int n, Activation a)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) x[i] = Activate(x[i], a);
        }

        private static void gradient_array_kernel(float[] x, int n, Activation a, float[] delta, int xStart = 0, int deltaStart = 0)
        {
            int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
            if (i < n) delta[i + deltaStart] *= Gradient(x[i + xStart], a);
        }

        [GpuManaged]
        public static void activate_array_ongpu(float[] x, int n, Activation a)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(activate_array_kernel, lp, x, n, a);
        }

        [GpuManaged]
        public static void gradient_array_ongpu(float[] x, int n, Activation a, float[] delta, int xStart = 0, int deltaStart = 0)
        {
            var lp = new LaunchParam(CudaUtils.cuda_gridsize(n), new dim3(CudaUtils.BlockSize));
            Gpu.Default.Launch(gradient_array_kernel, lp, x, n, a, delta, xStart, deltaStart);
        }
    }
}
