using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Configuration;
using System.IO;
using System.Linq;
using TensorFlow;

namespace TensorFlowSharp.Service
{
    sealed class TensorFlowSharp
    {
        public ReadOnlyCollection<Prediction> Run()
        {
            string libsBath = ConfigurationManager.AppSettings["LibsPath"];
            string setsPath = ConfigurationManager.AppSettings["SetsPath"];

            string image = $@"{setsPath}\sample_flower.jpg";

            byte[] model = File.ReadAllBytes($@"{setsPath}\output_graph.pb");
            string[] labels = File.ReadAllLines($@"{setsPath}\output_labels.txt");

            using (var graph = new TFGraph())
            {
                graph.Import(new TFBuffer(model));

                using (var session = new TFSession(graph))
                {
                    TFTensor tensor = ImageUtil.CreateTensorFromImageFile(image);

                    TFSession.Runner runner = session.GetRunner();

                    if (runner == null || tensor == null)
                    {
                        Console.WriteLine("Runner or Tensor is null!?");
                        Environment.Exit(1);
                    }

                    runner.AddInput(graph["DecodeJpeg/contents"][0], tensor);
                    runner.Fetch(graph["final_result"][0]);

                    try
                    {
                        TFTensor[] output = runner.Run();

                        float[,] scores = (float[,]) output[0].GetValue();

                        var predictions = new List<Prediction>();

                        for (int i = 0; i < scores.Length; i++)
                        {
                            float score = scores[0, i];
                            string label = labels[i];

                            predictions.Add(new Prediction(label, score));
                        }

                        return predictions
                            .OrderByDescending(p => p.Score)
                                .ToList()
                                    .AsReadOnly();
                    }
                    catch (TFException e)
                    {
                        Console.WriteLine(e.ToString());
                    }
                }
            }

            return new List<Prediction>().AsReadOnly();
        }
    }
}