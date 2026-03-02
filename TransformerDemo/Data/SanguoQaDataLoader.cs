using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace TransformerDemo;

/// <summary>
/// 从 JSONL 文件加载三国演义问答数据，构建字符级词表，并输出与 DataGenerator.Sample 兼容的样本列表。
/// 样本中 Input / Target 为已编码的 token id 序列（仅内容，不含 [SOS]/[EOS]，由 PreprocessTokenized 统一添加）。
/// </summary>
public static class SanguoQaDataLoader
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    /// <summary>单条 JSON 记录</summary>
    public record QaRecord(string Question, string Answer);

    /// <summary>从 JSONL 文件读取所有问答对（每行一个 JSON 对象，含 question 与 answer 字段）。</summary>
    public static List<QaRecord> ReadJsonl(string path,int? take = null)
    {
        var list = new List<QaRecord>();
        if (!File.Exists(path))
            return list;
        foreach (string line in File.ReadLines(path))
        {
            string s = line.Trim();
            if (string.IsNullOrEmpty(s)) continue;
            try
            {
                var doc = JsonSerializer.Deserialize<JsonElement>(s);
                string q = doc.TryGetProperty("question", out var qp) ? qp.GetString() ?? "" : "";
                string a = doc.TryGetProperty("answer", out var ap) ? ap.GetString() ?? "" : "";
                q = q.Trim();
                a = a.Trim();
                if (string.IsNullOrEmpty(q) && string.IsNullOrEmpty(a))
                    continue;
                list.Add(new QaRecord(q, a));
            }
            catch
            {
                // 忽略无法解析的行
            }
        }
        if (take != null)
        {
            return list.Take(take.Value).ToList();
        }
        return list;
    }

    /// <summary>
    /// 加载训练集与验证集 JSONL，从全部问答文本构建字符词表，并将每条问答转为 Sample(Input=问题 token ids, Target=答案 token ids)。
    /// 返回 (trainSamples, validSamples, vocab)。若某文件不存在则对应列表为空。
    /// </summary>
    public static (List<DataGenerator.Sample> train, List<DataGenerator.Sample> valid, CharVocabulary vocab) Load(
        string trainPath,
        string validPath)
    {
        var trainRecords = ReadJsonl(trainPath,100);
        var validRecords = ReadJsonl(validPath,20);

        var allTexts = new List<string>();
        foreach (var r in trainRecords)
        {
            if (!string.IsNullOrEmpty(r.Question)) allTexts.Add(r.Question);
            if (!string.IsNullOrEmpty(r.Answer)) allTexts.Add(r.Answer);
        }
        foreach (var r in validRecords)
        {
            if (!string.IsNullOrEmpty(r.Question)) allTexts.Add(r.Question);
            if (!string.IsNullOrEmpty(r.Answer)) allTexts.Add(r.Answer);
        }

        var vocab = new CharVocabulary();
        vocab.BuildFromTexts(allTexts);

        var train = new List<DataGenerator.Sample>();
        foreach (var r in trainRecords)
        {
            int[] inputIds = string.IsNullOrEmpty(r.Question) ? Array.Empty<int>() : vocab.EncodeText(r.Question);
            int[] targetIds = string.IsNullOrEmpty(r.Answer) ? Array.Empty<int>() : vocab.EncodeText(r.Answer);
            train.Add(new DataGenerator.Sample(inputIds, targetIds));
        }

        var valid = new List<DataGenerator.Sample>();
        foreach (var r in validRecords)
        {
            int[] inputIds = string.IsNullOrEmpty(r.Question) ? Array.Empty<int>() : vocab.EncodeText(r.Question);
            int[] targetIds = string.IsNullOrEmpty(r.Answer) ? Array.Empty<int>() : vocab.EncodeText(r.Answer);
            valid.Add(new DataGenerator.Sample(inputIds, targetIds));
        }

        return (train, valid, vocab);
    }
}
