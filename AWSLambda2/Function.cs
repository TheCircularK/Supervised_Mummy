using Amazon.Lambda.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Assembly attribute to enable the Lambda function's JSON input to be converted into a .NET class.
[assembly: LambdaSerializer(typeof(Amazon.Lambda.Serialization.SystemTextJson.DefaultLambdaJsonSerializer))]

namespace AWSLambda2;

public class Function
{
    private InferenceSession _session;

    public Function() { }

    public Function(InferenceSession session)
    {
        _session = session;
    }
    
    /// <summary>
    /// A simple function that takes a string and does a ToUpper
    /// </summary>
    /// <param name="input"></param>
    /// <param name="context"></param>
    /// <returns></returns>
    public Prediction FunctionHandler(MummyData data)
    {
        _session = new InferenceSession(Path.Combine(Directory.GetCurrentDirectory(), "supervised.onnx"));

        var result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())
            });

        Tensor<string> score = result.First().AsTensor<string>();
        var categories = new[] { "W", "E", "unknown" };
        int predictionIndex = Array.IndexOf(score.ToArray(), score.Max());
        var prediction = new Prediction { PredictedValue = categories[predictionIndex] };

        return prediction;
    }
}
