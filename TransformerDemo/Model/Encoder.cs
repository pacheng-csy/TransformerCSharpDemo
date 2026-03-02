using System;

namespace TransformerDemo;

/// <summary>
/// 编码器：堆叠 N 个 EncoderLayer。在首层前加一次位置编码。
/// 输入 (batch, seqLen, d_model)，输出同形状。
/// </summary>
public class Encoder
{
    private readonly PositionalEncoding _pe;
    private readonly EncoderLayer[] _layers;

    /// <summary>
    /// 构造编码器，由多个 <see cref="EncoderLayer"/> 堆叠而成，并共享一套位置编码。
    /// </summary>
    /// <param name="dModel">隐藏维度 d_model</param>
    /// <param name="numHeads">自注意力头数</param>
    /// <param name="dFf">前馈网络隐藏层维度</param>
    /// <param name="numLayers">编码器层数</param>
    /// <param name="maxLen">支持的最大序列长度</param>
    public Encoder(int dModel, int numHeads, int dFf, int numLayers, int maxLen)
    {
        _pe = new PositionalEncoding(maxLen, dModel);
        _layers = new EncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
            _layers[i] = new EncoderLayer(dModel, numHeads, dFf);
    }

    /// <summary>
    /// 前向传播：先加位置编码，再依次通过每一层 <see cref="EncoderLayer"/>。
    /// </summary>
    /// <param name="x">输入序列表示 (batch, seqLen, d_model)</param>
    /// <param name="paddingMask">Padding 掩码 (batch, seqLen, seqLen)，可为 null</param>
    /// <returns>编码器堆叠后的输出</returns>
    public float[][][] Forward(float[][][] x, bool[][][]? paddingMask)
    {
        x = _pe.Forward(x);
        foreach (var layer in _layers)
            x = layer.Forward(x, paddingMask);
        return x;
    }

    /// <summary>反向：dLdOut 与输出同形状，返回 dLdX（输入梯度，位置编码为加法，梯度直接传回）。</summary>
    public float[][][] Backward(float[][][] dLdOut)
    {
        var dLd = dLdOut;
        for (int i = _layers.Length - 1; i >= 0; i--)
            dLd = _layers[i].Backward(dLd);
        return dLd;
    }

    public void ZeroGrad()
    {
        foreach (var layer in _layers)
            layer.ZeroGrad();
    }

    public EncoderLayer[] Layers => _layers;
}
