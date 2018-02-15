using System.IO;
using TensorFlow;

namespace TensorFlowSharp.Service
{
    public static class ImageUtil
    {
        public static TFTensor CreateTensorFromImageFile(string file)
        {
            byte[] contents = File.ReadAllBytes(file);

            TFTensor tensor = TFTensor.CreateString(contents);

            return tensor;
        }
    }
}