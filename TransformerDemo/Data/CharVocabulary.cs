using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TransformerDemo;

/// <summary>
/// 字符级词表：用于文本问答任务。特殊 token 与 Vocabulary 一致：[PAD]=0, [SOS]=1, [EOS]=2；
/// 其余为字符从 3 开始顺序编码。可从语料构建，并支持保存/加载以便推理时复用。
/// </summary>
public class CharVocabulary
{
    public const int PadId = 0;
    public const int SosId = 1;
    public const int EosId = 2;
    /// <summary>第一个字符 token 的 ID</summary>
    public const int FirstCharTokenId = 3;

    private readonly Dictionary<char, int> _charToId;
    private readonly List<char?> _idToChar; // [0]=PAD, [1]=SOS, [2]=EOS, 之后为字符

    /// <summary>词表总大小（包含 PAD/SOS/EOS 与所有普通字符）。</summary>
    public int Size => _idToChar.Count;

    /// <summary>
    /// 创建一个空的字符级词表实例。
    /// 调用方通常先 new，再通过 <see cref="BuildFromTexts(IEnumerable{string})"/> 从语料中构建映射。
    /// </summary>
    public CharVocabulary()
    {
        _charToId = new Dictionary<char, int>();
        _idToChar = new List<char?> { null, null, null }; // 0=PAD, 1=SOS, 2=EOS 不映射到字符
    }

    /// <summary>从若干文本中收集所有字符并构建词表（会清空当前映射并重建）。</summary>
    public void BuildFromTexts(IEnumerable<string> texts)
    {
        _charToId.Clear();
        _idToChar.Clear();
        _idToChar.Add(null);
        _idToChar.Add(null);
        _idToChar.Add(null);

        var chars = new HashSet<char>();
        foreach (string t in texts)
        {
            if (string.IsNullOrEmpty(t)) continue;
            foreach (char c in t)
                chars.Add(c);
        }

        var ordered = chars.OrderBy(c => c).ToList();
        foreach (char c in ordered)
        {
            int id = _idToChar.Count;
            _charToId[c] = id;
            _idToChar.Add(c);
        }
    }

    /// <summary>将单个字符编码为 token id；若未在词表中则抛出。</summary>
    public int EncodeChar(char c)
    {
        if (_charToId.TryGetValue(c, out int id))
            return id;
        throw new ArgumentOutOfRangeException(nameof(c), $"字符 '{c}' 不在词表中。请确保推理使用与训练相同的词表。");
    }

    /// <summary>将 token id 解码为字符；特殊 token 或未知 id 返回 null。</summary>
    public char? DecodeChar(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _idToChar.Count)
            return null;
        return _idToChar[tokenId];
    }

    /// <summary>将整段文本编码为 token id 数组（仅内容，不含 [SOS]/[EOS]）。</summary>
    public int[] EncodeText(string text)
    {
        if (string.IsNullOrEmpty(text))
            return Array.Empty<int>();
        var ids = new int[text.Length];
        for (int i = 0; i < text.Length; i++)
            ids[i] = EncodeChar(text[i]);
        return ids;
    }

    /// <summary>将 token id 数组解码为字符串，跳过 [PAD]/[SOS]/[EOS]。</summary>
    public string DecodeIds(int[] ids)
    {
        if (ids == null || ids.Length == 0)
            return string.Empty;
        var sb = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            if (id == PadId || id == SosId || id == EosId)
                continue;
            var c = DecodeChar(id);
            if (c.HasValue)
                sb.Append(c.Value);
        }
        return sb.ToString();
    }

    /// <summary>判断给定 token id 是否为 [PAD]。</summary>
    public bool IsPad(int tokenId) => tokenId == PadId;

    /// <summary>判断给定 token id 是否为特殊标记（[SOS] 或 [EOS]）。</summary>
    public bool IsSpecial(int tokenId) => tokenId == SosId || tokenId == EosId;

    /// <summary>保存词表到目录：vocab.json 存字符列表（与训练时模型同目录，便于推理加载）。</summary>
    public void SaveToDirectory(string path)
    {
        Directory.CreateDirectory(path);
        var chars = new List<string>();
        for (int i = FirstCharTokenId; i < _idToChar.Count; i++)
        {
            char? c = _idToChar[i];
            if (c.HasValue)
                chars.Add(c.Value.ToString());
            else
                chars.Add("");
        }
        var jsonPath = Path.Combine(path, "vocab.json");
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(chars));
    }

    /// <summary>从目录加载词表（读取 vocab.json）。</summary>
    public void LoadFromDirectory(string path)
    {
        var jsonPath = Path.Combine(path, "vocab.json");
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException($"词表文件不存在: {jsonPath}");
        var json = File.ReadAllText(jsonPath);
        var chars = JsonSerializer.Deserialize<List<string>>(json);
        if (chars == null)
            throw new InvalidOperationException("vocab.json 格式无效。");
        _charToId.Clear();
        _idToChar.Clear();
        _idToChar.Add(null);
        _idToChar.Add(null);
        _idToChar.Add(null);
        foreach (string s in chars)
        {
            if (string.IsNullOrEmpty(s))
                continue;
            char c = s[0];
            int id = _idToChar.Count;
            _charToId[c] = id;
            _idToChar.Add(c);
        }
    }
}
