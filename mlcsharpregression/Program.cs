using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

// Define your data structure
public class InputData
{
    [LoadColumn(0)] public float X;
    [LoadColumn(1)] public float Y;
}

public class OutputData
{
    [ColumnName("Score")]
    public float Y;
}

class Program
{
    static void Main(string[] args)
    {
        // Create a new MLContext
        var context = new MLContext();

        // Create sample data
        var data = new[]
        {
            new InputData { X = 1, Y = 2 },
            new InputData { X = 2, Y = 4 },
            new InputData { X = 3, Y = 6 },
            new InputData { X = 4, Y = 8 },
        };

        // Load data into IDataView
        var dataView = context.Data.LoadFromEnumerable(data);

        // Define the pipeline
        var pipeline = context.Transforms.Concatenate("Features", nameof(InputData.X))
            .Append(context.Transforms.NormalizeMinMax("Features"))
            .Append(context.Transforms.CopyColumns("Label", nameof(InputData.Y)))
            .Append(context.Transforms.Concatenate("Features", "Features"))
            .Append(context.Regression.Trainers.Sdca())
            .Append(context.Transforms.CopyColumns("Score", nameof(OutputData.Y)));

        // Train the model
        var model = pipeline.Fit(dataView);

        // Make predictions
        var predictionEngine = context.Model.CreatePredictionEngine<InputData, OutputData>(model);
        var prediction = predictionEngine.Predict(new InputData { X = 2 });

        Console.WriteLine($"Prediction: {prediction.Y}");
    }
}
