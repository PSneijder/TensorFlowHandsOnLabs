using System;
using System.Collections.Generic;

namespace TensorFlowSharp.Service
{
    class Program
    {
        static void Main()
        {
            IReadOnlyCollection<Prediction> predictions = new TensorFlowSharp().Run();

            foreach (Prediction prediction in predictions)
            {
                Console.WriteLine($"{prediction.Label} = {prediction.Score}");
            }

            Console.WriteLine("Press ANY key to quit.");
            Console.ReadKey();
        }
    }
}