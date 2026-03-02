using System;
using System.Collections.Generic;
using System.IO;

namespace TransformerDemo;

/// <summary>
/// Transformer 模型：Embedding + 位置编码（在 Encoder/Decoder 内）+ Encoder + Decoder + 输出线性层映射到词表大小。
/// 用于序列到序列：输入编码器序列、解码器输入序列，输出解码器每个位置的 logits (vocabSize)。
/// </summary>
public class TransformerModel
{
    private readonly Embedding _embedding;
    private readonly Encoder _encoder;
    private readonly Decoder _decoder;
    private readonly float[][] _outputProj; // (d_model, vocabSize)
    private readonly float[][] _gradOutputProj;
    private float[][][]? _lastDecOut;  // 解码器输出（输出投影前），Backward 用
    private readonly int _dModel;
    private readonly int _vocabSize;
    private readonly int _numHeads;
    private readonly int _dFf;

    /// <summary>
    /// 构造一个完整的 Transformer 编码器-解码器模型。
    /// </summary>
    /// <param name="vocabSize">词表大小（输出类别数）</param>
    /// <param name="dModel">模型隐藏维度 d_model</param>
    /// <param name="numHeads">多头注意力头数</param>
    /// <param name="dFf">前馈网络隐藏层维度</param>
    /// <param name="numEncoderLayers">Encoder 堆叠层数</param>
    /// <param name="numDecoderLayers">Decoder 堆叠层数</param>
    /// <param name="maxLen">支持的最大序列长度（用于位置编码与 mask）</param>
    public TransformerModel(int vocabSize, int dModel, int numHeads, int dFf, int numEncoderLayers, int numDecoderLayers, int maxLen)
    {
        _vocabSize = vocabSize;
        _dModel = dModel;
        _numHeads = numHeads;
        _dFf = dFf;
        MaxLen = maxLen;
        _embedding = new Embedding(vocabSize, dModel);
        _encoder = new Encoder(dModel, numHeads, dFf, numEncoderLayers, maxLen);
        _decoder = new Decoder(dModel, numHeads, dFf, numDecoderLayers, maxLen);
        _outputProj = MatrixHelper.Zeros(dModel, vocabSize);
        _gradOutputProj = MatrixHelper.Zeros(dModel, vocabSize);
        MatrixHelper.XavierUniform(_outputProj);
    }

    /// <summary>
    /// 前向：encIds (batch, encLen), decIds (batch, decLen)。
    /// encValidLengths / decValidLengths 用于构造 padding mask。
    /// 返回 logits (batch, decLen, vocabSize)。
    /// </summary>
    public float[][][] Forward(int[][] encIds, int[][] decIds, int[] encValidLengths, int[] decValidLengths)
    {
        int batch = encIds.Length, encLen = encIds[0].Length, decLen = decIds[0].Length;
        bool[][][] encMask = MatrixHelper.CreatePaddingMask(batch, encLen, encValidLengths);
        bool[][][] decMask = MatrixHelper.CreatePaddingMask(batch, decLen, decValidLengths);

        var encEmb = _embedding.Forward(encIds);
        var encOut = _encoder.Forward(encEmb, encMask);

        var decEmb = _embedding.Forward(decIds);
        var decOut = _decoder.Forward(decEmb, encOut, decMask, encMask);
        _lastDecOut = MatrixHelper.Copy3(decOut);

        // decOut (batch, decLen, d_model) * outputProj (d_model, vocabSize) => (batch, decLen, vocabSize)
        var logits = MatrixHelper.MultiplyBatch3D(decOut, _outputProj);
        return logits;
    }

    /// <summary>所有参数梯度清零，每次训练步前调用。</summary>
    public void ZeroGrad()
    {
        _embedding.ZeroGrad();
        for (int i = 0; i < _gradOutputProj.Length; i++)
            for (int j = 0; j < _gradOutputProj[i].Length; j++)
                _gradOutputProj[i][j] = 0;
        _encoder.ZeroGrad();
        _decoder.ZeroGrad();
    }

    /// <summary>反向传播：dLdLogits (batch, decLen, vocabSize)；encIds/decIds 用于 embedding 回传。</summary>
    public void Backward(float[][][] dLdLogits, int[][] encIds, int[][] decIds)
    {
        if (_lastDecOut == null) throw new InvalidOperationException("Forward must be called before Backward.");
        var dLdDecOut = MatrixHelper.MultiplyBatch3DBackward(dLdLogits, _lastDecOut, _outputProj, _gradOutputProj);
        var (dLdDecInput, dLdEncOutput) = _decoder.Backward(dLdDecOut);
        var dLdEncInput = _encoder.Backward(dLdEncOutput);
        _embedding.Backward(dLdEncInput, encIds);
        _embedding.Backward(dLdDecInput, decIds);
    }

    /// <summary>收集所有梯度（顺序与 GetAllParameters 一致），用于 SGD 更新。</summary>
    public List<float[][]> GetAllGradients()
    {
        var list = new List<float[][]>();
        list.Add(_embedding.GetGradTable());
        list.Add(_gradOutputProj);
        foreach (var layer in _encoder.Layers)
        {
            var (gQ, gK, gV, gO) = layer.SelfAttn.GetGradients();
            list.Add(gQ); list.Add(gK); list.Add(gV); list.Add(gO);
            var (g1, g2) = layer.Ffn.GetGradients();
            list.Add(g1); list.Add(g2);
        }
        foreach (var layer in _decoder.Layers)
        {
            var (gQ, gK, gV, gO) = layer.SelfAttn.GetGradients();
            list.Add(gQ); list.Add(gK); list.Add(gV); list.Add(gO);
            var (gQ2, gK2, gV2, gO2) = layer.CrossAttn.GetGradients();
            list.Add(gQ2); list.Add(gK2); list.Add(gV2); list.Add(gO2);
            var (g1, g2) = layer.Ffn.GetGradients();
            list.Add(g1); list.Add(g2);
        }
        return list;
    }

    /// <summary>收集所有可训练参数（用于基于数值梯度的优化器）</summary>
    public List<float[][]> GetAllParameters()
    {
        var list = new List<float[][]>();
        list.Add(_embedding.GetTable());
        list.Add(_outputProj);
        foreach (var layer in _encoder.Layers)
        {
            var (wQ, wK, wV, wO) = layer.SelfAttn.GetParameters();
            list.Add(wQ); list.Add(wK); list.Add(wV); list.Add(wO);
            var (w1, w2) = layer.Ffn.GetParameters();
            list.Add(w1); list.Add(w2);
        }
        foreach (var layer in _decoder.Layers)
        {
            var (wQ, wK, wV, wO) = layer.SelfAttn.GetParameters();
            list.Add(wQ); list.Add(wK); list.Add(wV); list.Add(wO);
            var (wQ2, wK2, wV2, wO2) = layer.CrossAttn.GetParameters();
            list.Add(wQ2); list.Add(wK2); list.Add(wV2); list.Add(wO2);
            var (w1, w2) = layer.Ffn.GetParameters();
            list.Add(w1); list.Add(w2);
        }
        return list;
    }

    /// <summary>将扁平参数写回模型（顺序需与 GetAllParameters 一致）</summary>
    public void SetParametersFromFlat(float[] flat)
    {
        var list = GetAllParameters();
        int idx = 0;
        foreach (var m in list)
        {
            for (int i = 0; i < m.Length; i++)
                for (int j = 0; j < m[i].Length; j++)
                {
                    if (idx < flat.Length)
                        m[i][j] = flat[idx++];
                }
        }
    }

    /// <summary>将当前参数展平为一维数组（顺序与 SetParametersFromFlat 一致）</summary>
    public float[] GetParametersFlat()
    {
        var list = GetAllParameters();
        int count = 0;
        foreach (var m in list)
            count += m.Length * m[0].Length;
        var flat = new float[count];
        int idx = 0;
        foreach (var m in list)
            for (int i = 0; i < m.Length; i++)
                for (int j = 0; j < m[i].Length; j++)
                    flat[idx++] = m[i][j];
        return flat;
    }

    /// <summary>模型使用的词表大小（输出类别数）。</summary>
    public int VocabSize => _vocabSize;
    /// <summary>当前模型支持的最大序列长度（由构造函数传入）。</summary>
    public int MaxLen { get; private set; }

    /// <summary>将模型保存到指定目录：config.txt 存超参，weights.bin 存所有权重（与 GetAllParameters 顺序一致）。</summary>
    public void SaveToDirectory(string path)
    {
        Directory.CreateDirectory(path);
        var configPath = Path.Combine(path, "config.txt");
        using (var sw = new StreamWriter(configPath))
        {
            sw.WriteLine($"{_vocabSize} {_dModel} {_encoder.Layers.Length} {_decoder.Layers.Length} {MaxLen}");
            sw.WriteLine($"{_numHeads} {_dFf}");
        }

        var weightsPath = Path.Combine(path, "weights.bin");
        using (var fs = new FileStream(weightsPath, FileMode.Create, FileAccess.Write, FileShare.None))
        using (var bw = new BinaryWriter(fs))
        {
            foreach (var m in GetAllParameters())
            {
                int rows = m.Length;
                int cols = m[0].Length;
                bw.Write(rows);
                bw.Write(cols);
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        bw.Write(m[i][j]);
            }
        }
    }

    /// <summary>从目录加载模型权重（需先构造同结构的模型，再调用此方法）。</summary>
    public void LoadFromDirectory(string path)
    {
        var list = GetAllParameters();
        var weightsPath = Path.Combine(path, "weights.bin");
        using (var fs = new FileStream(weightsPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (var br = new BinaryReader(fs))
        {
            foreach (var m in list)
            {
                int rows = br.ReadInt32();
                int cols = br.ReadInt32();
                if (rows != m.Length || cols != m[0].Length)
                    throw new InvalidOperationException($"权重形状不匹配: 期望 ({m.Length},{m[0].Length}), 文件 ({rows},{cols})");
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        m[i][j] = br.ReadSingle();
            }
        }
    }

    /// <summary>从目录加载模型（静态工厂）：读取 config.txt 创建模型并加载 weights.bin。</summary>
    public static (TransformerModel model, int maxLen) LoadFromDirectory(string path, out int vocabSize, out int numHeads, out int dFf)
    {
        var configPath = Path.Combine(path, "config.txt");
        var lines = File.ReadAllLines(configPath);
        var line0 = lines[0].Split(' ');
        int v = int.Parse(line0[0]);
        int dModel = int.Parse(line0[1]);
        int numEncLayers = int.Parse(line0[2]);
        int numDecLayers = int.Parse(line0[3]);
        int maxLen = int.Parse(line0[4]);
        var line1 = lines[1].Split(' ');
        int heads = int.Parse(line1[0]);
        int ff = int.Parse(line1[1]);

        vocabSize = v;
        numHeads = heads;
        dFf = ff;

        var model = new TransformerModel(v, dModel, heads, ff, numEncLayers, numDecLayers, maxLen);
        model.LoadFromDirectory(path);
        return (model, maxLen);
    }
}
