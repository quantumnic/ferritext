/// Example usage of ferritext C# bindings.

using Ferritext;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine($"ferritext v{OcrEngine.Version()}");

        using var engine = new OcrEngine("eng");
        string result = engine.OcrFile("test_image.png");
        Console.WriteLine($"Result: {result}");
    }
}
