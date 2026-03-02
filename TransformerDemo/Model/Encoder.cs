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

    public Encoder(int dModel, int numHeads, int dFf, int numLayers, int maxLen)
    {
        _pe = new PositionalEncoding(maxLen, dModel);
        _layers = new EncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
            _layers[i] = new EncoderLayer(dModel, numHeads, dFf);
    }

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
