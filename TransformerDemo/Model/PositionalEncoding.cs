using System;

namespace TransformerDemo;

/// <summary>
/// 位置编码（Positional Encoding）
///
/// 【为什么需要位置编码】
/// Self-Attention 在数学上是对所有位置“对称”的：它只看 token 之间的相似度，不看谁在前谁在后。
/// 如果不显式告诉模型“这是第几个位置”，它就无法区分 "A 在 B 前" 和 "B 在 A 前" 这样的顺序关系。
/// 所以我们需要给每个位置加上一个“位置向量”，再和词向量相加，一起送入后续层。
///
/// 【正弦/余弦公式】
/// 论文采用固定（不可学习）的编码：
///   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
///   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// 也就是说：
///   - 偶数维用 sin，奇数维用 cos；
///   - i 越大，分母 10000^(2i/d_model) 越大，对应的波长越长（低频分量）。
/// 可以把它理解为：我们在每个位置上叠加了一组不同频率的正弦波，模型可以从中“读出”相对位置信息。
/// 好处：(1) 相对位置可以通过三角恒等式表示为 PE 的线性组合；(2) 位置编码是固定公式，新位置也能直接算，不需要重新训练。
/// </summary>
public class PositionalEncoding
{
    /// <summary>预计算的位置编码表 [maxLen, dModel]，前向时按 seqLen 截取并加到输入</summary>
    private readonly float[][] _pe;

    public PositionalEncoding(int maxLen, int dModel)
    {
        _pe = MatrixHelper.Zeros(maxLen, dModel);
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < dModel; i++)
            {
                double angle = pos / Math.Pow(10000.0, (double)(2 * i) / dModel);
                if (i % 2 == 0)
                    _pe[pos][i] = (float)Math.Sin(angle);
                else
                    _pe[pos][i] = (float)Math.Cos(angle);
            }
        }
    }

    /// <summary>
    /// 前向：x 形状 (batch, seqLen, dModel)，将 PE[0:seqLen] 加到 x 上，输出同形状。
    /// </summary>
    public float[][][] Forward(float[][][] x)
    {
        int batch = x.Length, seqLen = x[0].Length, dModel = x[0][0].Length;
        var out_ = MatrixHelper.Zeros3(batch, seqLen, dModel);
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int d = 0; d < dModel; d++)
                    out_[b][s][d] = x[b][s][d] + _pe[s][d];
        return out_;
    }
}
