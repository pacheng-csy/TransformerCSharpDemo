using System;

namespace TransformerDemo;

/// <summary>
/// 多头注意力（Multi-Head Attention）
///
/// 【直观理解】
/// 可以把“一个注意力头”看成是“用某一种视角去看整句话（或整段序列）”：比如专门学语法关系、专门学实体指代等。
/// 多头注意力就是同时开好几个“视角”（多个头），每个头各算一次 Scaled Dot-Product Attention，然后把结果拼在一起。
///
/// 【多个头如何并行】
/// - 首先把总维度 d_model 按照头数 h 平均切分：d_k = d_model / h；
/// - 对 Q、K、V 分别做线性投影得到 h 组 (Q_i, K_i, V_i)，对应 h 个头的输入；
/// - 每个头独立调用一次 ScaledDotProductAttention，得到 h 份 (seqLen x d_k) 的输出。
///
/// 【Concat 与线性变换】
/// - 把 h 个头的输出在特征维上拼接，得到 seqLen x (h*d_k) = seqLen x d_model；
/// - 再通过一个线性层 W^O 把拼接结果“混合”回 d_model 维度，
///   这样模型可以自动学习“如何综合不同头的视角”。
/// </summary>
public class MultiHeadAttention
{
    private readonly int _dModel, _numHeads, _dK, _dV;
    private readonly float[][] _wQ, _wK, _wV, _wO;
    private readonly float[][] _gradWQ, _gradWK, _gradWV, _gradWO;
    private readonly float _scale;

    // 前向缓存，供 Backward 使用
    private float[][][]? _lastQ, _lastK, _lastV, _lastQProj, _lastKProj, _lastVProj, _lastConcat;
    private float[][][][]? _lastHeadAttn;

    /// <summary>
    /// 构造多头注意力层，内部创建 4 个线性变换矩阵 W_Q / W_K / W_V / W_O 并做 Xavier 初始化。
    /// </summary>
    /// <param name="dModel">总特征维度 d_model</param>
    /// <param name="numHeads">注意力头数，要求能整除 d_model</param>
    public MultiHeadAttention(int dModel, int numHeads)
    {
        _dModel = dModel;
        _numHeads = numHeads;
        _dK = dModel / numHeads;
        _dV = dModel / numHeads;
        if (_dK * numHeads != dModel)
            throw new ArgumentException("d_model must be divisible by numHeads.");
        _scale = (float)(1.0 / Math.Sqrt(_dK));

        _wQ = MatrixHelper.Zeros(dModel, dModel);
        _wK = MatrixHelper.Zeros(dModel, dModel);
        _wV = MatrixHelper.Zeros(dModel, dModel);
        _wO = MatrixHelper.Zeros(dModel, dModel);
        _gradWQ = MatrixHelper.Zeros(dModel, dModel);
        _gradWK = MatrixHelper.Zeros(dModel, dModel);
        _gradWV = MatrixHelper.Zeros(dModel, dModel);
        _gradWO = MatrixHelper.Zeros(dModel, dModel);
        MatrixHelper.XavierUniform(_wQ);
        MatrixHelper.XavierUniform(_wK);
        MatrixHelper.XavierUniform(_wV);
        MatrixHelper.XavierUniform(_wO);
    }

    /// <summary>
    /// 将所有权重矩阵的梯度清零，在每次反向传播更新前调用。
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var g in new[] { _gradWQ, _gradWK, _gradWV, _gradWO })
            for (int i = 0; i < g.Length; i++)
                for (int j = 0; j < g[i].Length; j++)
                    g[i][j] = 0;
    }

    /// <summary>
    /// 前向：Q、K、V 形状 (batch, seqLen, d_model)，mask (batch, seqQ, seqK) 或 null。
    /// 输出 (batch, seqLen, d_model)。
    /// </summary>
    public float[][][] Forward(float[][][] q, float[][][] k, float[][][] v, bool[][][]? mask = null)
    {
        int batch = q.Length, seqQ = q[0].Length, seqK = k[0].Length;
        var qProj = MatrixHelper.MultiplyBatch3D(q, _wQ);
        var kProj = MatrixHelper.MultiplyBatch3D(k, _wK);
        var vProj = MatrixHelper.MultiplyBatch3D(v, _wV);

        var headOutputs = new float[_numHeads][][][];
        var headAttn = new float[_numHeads][][][];
        for (int h = 0; h < _numHeads; h++)
        {
            var qh = SliceHead(qProj, h);
            var kh = SliceHead(kProj, h);
            var vh = SliceHead(vProj, h);
            var (outH, attnH) = ScaledDotProductAttention.Forward(qh, kh, vh, mask);
            headOutputs[h] = outH;
            headAttn[h] = attnH;
        }
        var concat = ConcatHeads(headOutputs);
        _lastQ = q; _lastK = k; _lastV = v;
        _lastQProj = qProj; _lastKProj = kProj; _lastVProj = vProj;
        _lastConcat = concat;
        _lastHeadAttn = headAttn;
        return MatrixHelper.MultiplyBatch3D(concat, _wO);
    }

    /// <summary>反向：dLdOut 与输出同形状。返回 (dLdQ, dLdK, dLdV)。</summary>
    public (float[][][] dLdQ, float[][][] dLdK, float[][][] dLdV) Backward(float[][][] dLdOut)
    {
        if (_lastConcat == null || _lastHeadAttn == null) throw new InvalidOperationException("Forward must be called before Backward.");
        int batch = _lastQ!.Length, seqQ = _lastQ[0].Length, seqK = _lastK![0].Length;
        var dLdConcat = MatrixHelper.MultiplyBatch3DBackward(dLdOut, _lastConcat!, _wO, _gradWO);

        var dLdQProj = MatrixHelper.Zeros3(batch, seqQ, _dModel);
        var dLdKProj = MatrixHelper.Zeros3(batch, seqK, _dModel);
        var dLdVProj = MatrixHelper.Zeros3(batch, seqK, _dModel);
        for (int h = 0; h < _numHeads; h++)
        {
            var dLdOutH = SliceHead(dLdConcat, h);
            var qh = SliceHead(_lastQProj!, h);
            var kh = SliceHead(_lastKProj!, h);
            var vh = SliceHead(_lastVProj!, h);
            var (dLdQh, dLdKh, dLdVh) = ScaledDotProductAttention.Backward(
                dLdOutH, qh, kh, vh, _lastHeadAttn[h], _scale);
            CopyHeadInto(dLdQProj, dLdQh, h);
            CopyHeadInto(dLdKProj, dLdKh, h);
            CopyHeadInto(dLdVProj, dLdVh, h);
        }
        var dLdQ = MatrixHelper.MultiplyBatch3DBackward(dLdQProj, _lastQ, _wQ, _gradWQ);
        var dLdK = MatrixHelper.MultiplyBatch3DBackward(dLdKProj, _lastK, _wK, _gradWK);
        var dLdV = MatrixHelper.MultiplyBatch3DBackward(dLdVProj, _lastV, _wV, _gradWV);
        return (dLdQ, dLdK, dLdV);
    }

    /// <summary>
    /// 从完整张量中切出第 head 个注意力头对应的子向量区域。
    /// </summary>
    /// <param name="x">输入张量 (batch, seq, d_model)</param>
    /// <param name="head">头索引 [0, _numHeads)</param>
    /// <returns>形状为 (batch, seq, d_k) 的子张量</returns>
    private float[][][] SliceHead(float[][][] x, int head)
    {
        int batch = x.Length, seq = x[0].Length;
        var out_ = MatrixHelper.Zeros3(batch, seq, _dK);
        int start = head * _dK;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seq; s++)
                for (int d = 0; d < _dK; d++)
                    out_[b][s][d] = x[b][s][start + d];
        return out_;
    }

    /// <summary>
    /// 将所有头在最后一维上拼接回 d_model 维度。
    /// </summary>
    /// <param name="headOutputs">每个头的输出，长度为 numHeads，单个为 (batch, seq, d_k)</param>
    /// <returns>拼接后的张量 (batch, seq, d_model)</returns>
    private float[][][] ConcatHeads(float[][][][] headOutputs)
    {
        int batch = headOutputs[0].Length, seq = headOutputs[0][0].Length;
        var out_ = MatrixHelper.Zeros3(batch, seq, _dModel);
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seq; s++)
            {
                int col = 0;
                for (int h = 0; h < _numHeads; h++)
                    for (int d = 0; d < _dK; d++)
                        out_[b][s][col++] = headOutputs[h][b][s][d];
            }
        return out_;
    }

    /// <summary>
    /// 将单个头的梯度累加回完整梯度张量对应的切片位置。
    /// </summary>
    private void CopyHeadInto(float[][][] full, float[][][] headGrad, int head)
    {
        int start = head * _dK;
        for (int b = 0; b < full.Length; b++)
            for (int s = 0; s < full[0].Length; s++)
                for (int d = 0; d < _dK; d++)
                    full[b][s][start + d] += headGrad[b][s][d];
    }

    /// <summary>返回可训练参数（用于优化器更新）</summary>
    public (float[][] wQ, float[][] wK, float[][] wV, float[][] wO) GetParameters() =>
        (_wQ, _wK, _wV, _wO);
    /// <summary>返回参数梯度（Backward 后累加）</summary>
    public (float[][] gradWQ, float[][] gradWK, float[][] gradWV, float[][] gradWO) GetGradients() =>
        (_gradWQ, _gradWK, _gradWV, _gradWO);
}
