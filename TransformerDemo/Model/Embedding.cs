using System;

namespace TransformerDemo;

/// <summary>
/// 词嵌入层：将 token id 映射为 d_model 维向量。查表实现。
/// 支持反向传播：Backward 将 dLdOut 按 ids 累加到 _gradTable。
/// </summary>
public class Embedding
{
    private readonly float[][] _table;   // [vocabSize, dModel]
    private readonly float[][] _gradTable;

    /// <summary>
    /// 构造词嵌入层，内部创建大小为 (vocabSize, dModel) 的查表矩阵并做 Xavier 初始化。
    /// </summary>
    /// <param name="vocabSize">词表大小</param>
    /// <param name="dModel">嵌入向量维度 d_model</param>
    public Embedding(int vocabSize, int dModel)
    {
        _table = MatrixHelper.Zeros(vocabSize, dModel);
        _gradTable = MatrixHelper.Zeros(vocabSize, dModel);
        MatrixHelper.XavierUniform(_table);
    }

    /// <summary>ids: (batch, seqLen)，返回 (batch, seqLen, d_model)</summary>
    public float[][][] Forward(int[][] ids)
    {
        int batch = ids.Length, seqLen = ids[0].Length, dModel = _table[0].Length;
        var out_ = MatrixHelper.Zeros3(batch, seqLen, dModel);
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int id = ids[b][s];
                if (id < 0 || id >= _table.Length) continue;
                for (int d = 0; d < dModel; d++)
                    out_[b][s][d] = _table[id][d];
            }
        return out_;
    }

    /// <summary>将梯度清零，每次训练步前调用。</summary>
    public void ZeroGrad()
    {
        for (int i = 0; i < _gradTable.Length; i++)
            for (int j = 0; j < _gradTable[i].Length; j++)
                _gradTable[i][j] = 0;
    }

    /// <summary>反向传播：dLdOut (batch, seqLen, dModel)，按 ids 累加到 _gradTable。</summary>
    public void Backward(float[][][] dLdOut, int[][] ids)
    {
        MatrixHelper.ScatterAddIntoRows(_gradTable, ids, dLdOut);
    }

    /// <summary>获取当前嵌入权重矩阵（参数）。</summary>
    public float[][] GetTable() => _table;

    /// <summary>获取嵌入权重的梯度矩阵（在 Backward 后累加得到）。</summary>
    public float[][] GetGradTable() => _gradTable;
}
