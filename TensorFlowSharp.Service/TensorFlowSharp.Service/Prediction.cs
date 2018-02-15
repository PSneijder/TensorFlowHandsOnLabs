
namespace TensorFlowSharp.Service
{
    struct Prediction
    {
        public Prediction(string label, float score)
            : this()
        {
            Label = label;
            Score = score;
        }

        public string Label { get; set; }
        public float Score { get; set; }
    }
}