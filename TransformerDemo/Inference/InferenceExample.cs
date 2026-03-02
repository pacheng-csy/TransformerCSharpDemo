using System;
using System.Collections.Generic;
using System.Linq;

namespace TransformerDemo;

/// <summary>
/// 调用已保存模型进行推理的示例（单独代码文件）。
/// 用法：训练后模型会保存到本地目录，运行程序时传入参数 "infer" 即可执行本示例，或直接调用 InferenceExample.Run("SavedModel")。
/// </summary>
public static class InferenceExample
{
    /// <summary>
    /// 从指定目录加载模型，并对若干输入序列进行“每个元素加 1”的推理，打印输入、期望输出与模型预测。
    /// </summary>
    /// <param name="modelDir">模型目录（需包含 config.txt 与 weights.bin），默认 "SavedModel"</param>
    public static void Run(int[][] input,string modelDir = "SavedModel")
    {
        if (!System.IO.Directory.Exists(modelDir))
        {
            Console.WriteLine($"模型目录不存在: {modelDir}。请先运行训练并保存模型。");
            return;
        }

        var vocab = new Vocabulary();
        var (model, maxLen) = TransformerModel.LoadFromDirectory(modelDir, out int vocabSize, out _, out _);
        Console.WriteLine($"已加载模型: {modelDir}, maxLen={maxLen}, vocabSize={vocabSize}");

        foreach (int[] inputDigits in input)
        {
            int[] targetDigits = inputDigits.Select(d => (d + 1) % 10).ToArray();
            var sample = new DataGenerator.Sample(inputDigits, targetDigits);
            var (inputIds, targetIds, validLengths) = DataGenerator.Preprocess(
                new List<DataGenerator.Sample> { sample }, maxLen, vocab);

            var logits = model.Forward(inputIds, targetIds, validLengths, validLengths);
            var predIds = new int[targetIds[0].Length];
            for (int s = 0; s < predIds.Length; s++)
            {
                int best = 0;
                for (int c = 1; c < logits[0][0].Length; c++)
                    if (logits[0][s][c] > logits[0][s][best]) best = c;
                predIds[s] = best;
            }

            // 只取有效部分并解码为数字（忽略 [PAD][SOS][EOS] 的纯数字部分）
            var predDigits = predIds
                .Where(id => id != Vocabulary.PadId && id != Vocabulary.SosId && id != Vocabulary.EosId)
                .Select(id => vocab.DecodeToDigit(id))
                .Where(d => d >= 0)
                .ToArray();

            Console.WriteLine($"  输入: [{string.Join(", ", inputDigits)}]");
            Console.WriteLine($"  期望: [{string.Join(", ", targetDigits)}]");
            Console.WriteLine($"  预测: [{string.Join(", ", predDigits)}]");
            Console.WriteLine();
        }
    }
}
