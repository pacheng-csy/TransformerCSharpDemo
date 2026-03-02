using System;
using System.Collections.Generic;
using System.IO;

namespace TransformerDemo;

/// <summary>
/// 三国演义问答推理：加载训练好的模型与词表，对用户问题做自回归解码生成答案。
/// </summary>
public static class SanguoInference
{
    /// <summary>
    /// 使用给定模型与词表，对问题字符串进行自回归解码，生成答案文本（遇到 [EOS] 或达到 maxLen 即停止）。
    /// </summary>
    public static string Ask(TransformerModel model, CharVocabulary vocab, string question, int maxLen)
    {
        question = question?.Trim() ?? "";
        if (string.IsNullOrEmpty(question))
            return "";

        int[] qIds = vocab.EncodeText(question);
        var encList = new List<int> { CharVocabulary.SosId };
        foreach (int id in qIds)
            encList.Add(id);
        encList.Add(CharVocabulary.EosId);
        while (encList.Count < maxLen)
            encList.Add(CharVocabulary.PadId);
        int[] encIds = encList.ToArray();
        int encValidLen = qIds.Length + 2;

        int[][] encBatch = new[] { encIds };
        int[] encValidLengths = new[] { encValidLen };

        var decIds = new int[maxLen];
        decIds[0] = CharVocabulary.SosId;
        for (int i = 1; i < maxLen; i++)
            decIds[i] = CharVocabulary.PadId;

        var outputIds = new List<int>();
        for (int step = 0; step < maxLen - 1; step++)
        {
            int decValidLen = step + 1;
            int[][] decBatch = new[] { decIds };
            int[] decValidLengths = new[] { decValidLen };

            float[][][] logits = model.Forward(encBatch, decBatch, encValidLengths, decValidLengths);
            float[] lastLogits = logits[0][step];
            int nextId = Argmax(lastLogits);

            if (nextId == CharVocabulary.EosId)
                break;
            outputIds.Add(nextId);
            decIds[step + 1] = nextId;
        }

        return vocab.DecodeIds(outputIds.ToArray());
    }

    private static int Argmax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++)
            if (a[i] > a[best]) best = i;
        return best;
    }

    /// <summary>
    /// 从目录加载模型与词表，返回 (model, maxLen, vocab)。目录需包含 config.txt、weights.bin、vocab.json。
    /// </summary>
    public static (TransformerModel model, int maxLen, CharVocabulary vocab) LoadFromDirectory(string path)
    {
        if (!Directory.Exists(path))
            throw new DirectoryNotFoundException($"模型目录不存在: {path}");

        var (model, maxLen) = TransformerModel.LoadFromDirectory(path, out _, out _, out _);
        var vocab = new CharVocabulary();
        vocab.LoadFromDirectory(path);
        return (model, maxLen, vocab);
    }

    /// <summary>
    /// 交互式问答：从控制台读入问题，输出答案；输入空行或 "exit" 退出。
    /// </summary>
    public static void RunInteractive(string modelDir)
    {
        var (model, maxLen, vocab) = LoadFromDirectory(modelDir);
        Console.WriteLine($"已加载三国问答模型: {modelDir}, maxLen={maxLen}");
        Console.WriteLine("输入问题后回车获取答案，空行或 exit 退出。\n");

        while (true)
        {
            Console.Write("问: ");
            string? q = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(q) || q.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;
            try
            {
                string a = Ask(model, vocab, q, maxLen);
                Console.WriteLine($"答: {a}\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"错误: {ex.Message}\n");
            }
        }
    }
}
