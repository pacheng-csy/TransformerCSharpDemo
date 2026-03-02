using System;
using System.Collections.Generic;

namespace TransformerDemo;

/// <summary>
/// 词汇表（Vocabulary）：把“我们世界里的整数”转换成“模型世界里的 token id”的桥梁。
/// - 特殊标记：
///   - [PAD]=0：用来把不同长度的序列补齐到同一长度（填充位不参与真实计算）；
///   - [SOS]=1：Start Of Sequence，序列开始；
///   - [EOS]=2：End Of Sequence，序列结束。
/// - 普通数字：
///   - 数字 0~9 映射为 token 3~12（共 10 个），也就是给每个数字一个稳定的“单词编号”。
/// 初学者可以把它类比成“字典”：左边是我们认识的数字，右边是神经网络认识的索引。
/// </summary>
public class Vocabulary
{
    public const int PadId = 0;
    public const int SosId = 1;
    public const int EosId = 2;
    /// <summary>数字 0 对应 token 3，数字 9 对应 token 12</summary>
    public const int FirstDigitTokenId = 3;
    public const int DigitCount = 10;

    public int Size => FirstDigitTokenId + DigitCount; // 13

    private readonly int[] _tokenToDigit; // tokenId -> 数字值（仅对 3..12 有效，其余 -1）
    private readonly int _digitToTokenOffset;

    public Vocabulary()
    {
        _tokenToDigit = new int[Size];
        for (int i = 0; i < Size; i++)
            _tokenToDigit[i] = -1;
        for (int d = 0; d < DigitCount; d++)
            _tokenToDigit[FirstDigitTokenId + d] = d;
        _digitToTokenOffset = FirstDigitTokenId;
    }

    /// <summary>将数字（0~9）编码为 token id（3~12）</summary>
    public int EncodeDigit(int digit)
    {
        if (digit < 0 || digit >= DigitCount)
            throw new ArgumentOutOfRangeException(nameof(digit));
        return _digitToTokenOffset + digit;
    }

    /// <summary>将 token id 解码为数字；非数字 token 返回 -1</summary>
    public int DecodeToDigit(int tokenId)
    {
        if (tokenId < FirstDigitTokenId || tokenId >= FirstDigitTokenId + DigitCount)
            return -1;
        return _tokenToDigit[tokenId];
    }

    /// <summary>判断是否为 [PAD]</summary>
    public bool IsPad(int tokenId) => tokenId == PadId;

    /// <summary>判断是否为 [SOS] 或 [EOS]</summary>
    public bool IsSpecial(int tokenId) => tokenId == SosId || tokenId == EosId;
}
