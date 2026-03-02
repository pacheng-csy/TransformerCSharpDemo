using System;

namespace TransformerDemo;

/// <summary>
/// 解码器：堆叠 N 个 DecoderLayer。在首层前加一次位置编码。
/// 输入 (batch, decLen, d_model)，encOutput (batch, encLen, d_model)，输出 (batch, decLen, d_model)。
/// </summary>
public class Decoder
{
    private readonly PositionalEncoding _pe;
    private readonly DecoderLayer[] _layers;

    public Decoder(int dModel, int numHeads, int dFf, int numLayers, int maxLen)
    {
        _pe = new PositionalEncoding(maxLen, dModel);
        _layers = new DecoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
            _layers[i] = new DecoderLayer(dModel, numHeads, dFf);
    }

    public float[][][] Forward(float[][][] decInput, float[][][] encOutput, bool[][][]? decPaddingMask, bool[][][]? encPaddingMask)
    {
        decInput = _pe.Forward(decInput);
        foreach (var layer in _layers)
            decInput = layer.Forward(decInput, encOutput, decPaddingMask, encPaddingMask);
        return decInput;
    }

    /// <summary>反向：dLdOut 与输出同形状。返回 (dLdDecInput, dLdEncOutput)。</summary>
    public (float[][][] dLdDecInput, float[][][] dLdEncOutput) Backward(float[][][] dLdOut)
    {
        var dLdDec = dLdOut;
        var dLdEnc = (float[][][]?)null;
        for (int i = _layers.Length - 1; i >= 0; i--)
        {
            var (dDec, dEnc) = _layers[i].Backward(dLdDec);
            dLdDec = dDec;
            if (dLdEnc == null) dLdEnc = dEnc;
            else for (int b = 0; b < dLdEnc.Length; b++) MatrixHelper.AddInPlace(dLdEnc[b], dEnc[b]);
        }
        return (dLdDec, dLdEnc!);
    }

    public void ZeroGrad()
    {
        foreach (var layer in _layers)
            layer.ZeroGrad();
    }

    public DecoderLayer[] Layers => _layers;
}
