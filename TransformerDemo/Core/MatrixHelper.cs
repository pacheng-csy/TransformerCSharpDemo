using System;

namespace TransformerDemo;

/// <summary>
/// 纯 C# 实现的矩阵/张量工具类，用于教学 Transformer。
/// 提供创建、乘法、转置、逐元素运算、Softmax（含 mask）、LayerNorm、Mask 生成等。
/// </summary>
public static class MatrixHelper
{
    private static readonly Random Rng = new(42);

    // ----- 创建与形状 -----

    /// <summary>
    /// 创建一个二维零矩阵，所有元素初始化为 0。
    /// </summary>
    /// <param name="rows">行数（第一维长度）</param>
    /// <param name="cols">列数（第二维长度）</param>
    /// <returns>形状为 (rows, cols) 的二维数组</returns>
    public static float[][] Zeros(int rows, int cols)
    {
        var m = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            m[i] = new float[cols];
        }
        return m;
    }

    /// <summary>
    /// 创建一个三维零张量，通常用于批量 (batch, rows, cols) 的中间结果缓冲。
    /// </summary>
    /// <param name="batch">批大小</param>
    /// <param name="rows">每个样本的行数</param>
    /// <param name="cols">每个样本的列数</param>
    /// <returns>形状为 (batch, rows, cols) 的三维数组</returns>
    public static float[][][] Zeros3(int batch, int rows, int cols)
    {
        var t = new float[batch][][];
        for (int b = 0; b < batch; b++)
            t[b] = Zeros(rows, cols);
        return t;
    }

    /// <summary>
    /// 创建一个二维矩阵并用指定常数填充。
    /// </summary>
    /// <param name="rows">行数</param>
    /// <param name="cols">列数</param>
    /// <param name="value">初始填充值，默认 0</param>
    /// <returns>元素全部为 value 的矩阵</returns>
    public static float[][] Create(int rows, int cols, float value = 0f)
    {
        var m = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            m[i] = new float[cols];
            for (int j = 0; j < cols; j++) m[i][j] = value;
        }
        return m;
    }

    /// <summary>
    /// 深拷贝二维矩阵，返回新的数组实例。
    /// </summary>
    /// <param name="a">源矩阵</param>
    /// <returns>内容相同但引用独立的矩阵</returns>
    public static float[][] Copy(float[][] a)
    {
        int rows = a.Length, cols = a[0].Length;
        var m = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            m[i] = new float[cols];
            Array.Copy(a[i], m[i], cols);
        }
        return m;
    }

    /// <summary>
    /// 深拷贝三维张量，逐 batch 调用 <see cref="Copy(float[][])"/>。
    /// </summary>
    /// <param name="a">源三维张量</param>
    /// <returns>内容相同但引用独立的新张量</returns>
    public static float[][][] Copy3(float[][][] a)
    {
        var t = new float[a.Length][][];
        for (int b = 0; b < a.Length; b++)
            t[b] = Copy(a[b]);
        return t;
    }

    /// <summary>Xavier 均匀初始化：在 [-bound, bound] 上均匀，bound = sqrt(6/(fanIn+fanOut))</summary>
    public static void XavierUniform(float[][] w, int fanIn, int fanOut)
    {
        float bound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < w.Length; i++)
            for (int j = 0; j < w[i].Length; j++)
                w[i][j] = (float)(Rng.NextDouble() * 2 * bound - bound);
    }

    /// <summary>
    /// 使用 Xavier 均匀分布初始化参数矩阵，自动根据行列数推导 fanIn/fanOut。
    /// </summary>
    /// <param name="w">待初始化的权重矩阵 (outDim, inDim)</param>
    public static void XavierUniform(float[][] w)
    {
        int fanIn = w[0].Length, fanOut = w.Length;
        XavierUniform(w, fanIn, fanOut);
    }

    // ----- 矩阵乘法与转置 -----

    /// <summary>矩阵乘法 C = A * B，A (M,K), B (K,N) => C (M,N)</summary>
    public static float[][] Multiply(float[][] a, float[][] b)
    {
        int M = a.Length, K = a[0].Length, N = b[0].Length;
        var c = Zeros(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        return c;
    }

    /// <summary>批量矩阵乘：X (B,M,K) * W (K,N) => (B,M,N)</summary>
    public static float[][][] MultiplyBatch3D(float[][][] x, float[][] w)
    {
        int B = x.Length, M = x[0].Length, K = x[0][0].Length, N = w[0].Length;
        var out_ = Zeros3(B, M, N);
        for (int b = 0; b < B; b++)
            out_[b] = Multiply(x[b], w);
        return out_;
    }

    /// <summary>
    /// 计算二维矩阵转置，行列互换。
    /// </summary>
    /// <param name="a">输入矩阵 (rows, cols)</param>
    /// <returns>转置后的矩阵 (cols, rows)</returns>
    public static float[][] Transpose(float[][] a)
    {
        int rows = a.Length, cols = a[0].Length;
        var t = new float[cols][];
        for (int j = 0; j < cols; j++)
        {
            t[j] = new float[rows];
            for (int i = 0; i < rows; i++)
                t[j][i] = a[i][j];
        }
        return t;
    }

    // ----- 逐元素运算 -----

    /// <summary>
    /// 按元素逐位相加：c = a + b。
    /// </summary>
    /// <param name="a">左操作数矩阵</param>
    /// <param name="b">右操作数矩阵，形状需与 a 一致</param>
    /// <returns>逐元素相加后的新矩阵</returns>
    public static float[][] Add(float[][] a, float[][] b)
    {
        int rows = a.Length, cols = a[0].Length;
        var c = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            c[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                c[i][j] = a[i][j] + b[i][j];
        }
        return c;
    }

    /// <summary>
    /// 就地加法：target[i][j] += b[i][j]，直接修改 target 引用的数据。
    /// </summary>
    /// <param name="target">被累加的目标矩阵</param>
    /// <param name="b">要加上的矩阵，形状需与 target 一致</param>
    public static void AddInPlace(float[][] target, float[][] b)
    {
        for (int i = 0; i < target.Length; i++)
            for (int j = 0; j < target[i].Length; j++)
                target[i][j] += b[i][j];
    }

    /// <summary>
    /// 按标量缩放矩阵：r = a * c。
    /// </summary>
    /// <param name="a">输入矩阵</param>
    /// <param name="c">缩放系数</param>
    /// <returns>每个元素都乘以 c 的新矩阵</returns>
    public static float[][] Scale(float[][] a, float c)
    {
        int rows = a.Length, cols = a[0].Length;
        var r = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            r[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                r[i][j] = a[i][j] * c;
        }
        return r;
    }

    /// <summary>
    /// 逐元素乘法：r[i][j] = a[i][j] * b[i][j]。
    /// </summary>
    /// <param name="a">左操作数矩阵</param>
    /// <param name="b">右操作数矩阵，形状需与 a 一致</param>
    /// <returns>逐元素乘积矩阵</returns>
    public static float[][] ElementMul(float[][] a, float[][] b)
    {
        int rows = a.Length, cols = a[0].Length;
        var r = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            r[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                r[i][j] = a[i][j] * b[i][j];
        }
        return r;
    }

    /// <summary>
    /// 对矩阵的每个元素取平方根，常用于方差/方差加 eps 后的标准差计算。
    /// </summary>
    /// <param name="a">输入矩阵</param>
    /// <returns>逐元素开方后的新矩阵</returns>
    public static float[][] Sqrt(float[][] a)
    {
        int rows = a.Length, cols = a[0].Length;
        var r = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            r[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                r[i][j] = (float)Math.Sqrt(a[i][j]);
        }
        return r;
    }

    /// <summary>沿最后一维做 Softmax。mask 为 null 表示全部有效；否则 mask[i][j]==false 的位置在 softmax 前设为负无穷。</summary>
    /// <param name="x">形状 (rows, cols)，对每行在 cols 维上做 softmax</param>
    /// <param name="mask">形状 (rows, cols)，false 表示屏蔽（不参与 softmax）</param>
    /// <returns>形状与 x 相同，每行和为 1</returns>
    public static float[][] SoftmaxLastDim(float[][] x, bool[][]? mask = null)
    {
        int rows = x.Length, cols = x[0].Length;
        var out_ = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            float[] row = new float[cols];
            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                float v = x[i][j];
                if (mask != null && !mask[i][j])
                    v = float.NegativeInfinity;
                else if (v > max)
                    max = v;
                row[j] = v;
            }
            // 减最大值保证数值稳定
            float sum = 0;
            for (int j = 0; j < cols; j++)
            {
                if (mask != null && !mask[i][j])
                {
                    row[j] = 0;
                    continue;
                }
                row[j] = (float)Math.Exp(row[j] - max);
                sum += row[j];
            }
            if (sum > 0)
                for (int j = 0; j < cols; j++)
                    row[j] /= sum;
            out_[i] = row;
        }
        return out_;
    }

    /// <summary>3D Softmax：对最后一维做 softmax。x[b][i][j] 对 j 做 softmax。mask 形状 (B, rows, cols)，false 处屏蔽。</summary>
    public static float[][][] SoftmaxLastDim3(float[][][] x, bool[][][]? mask = null)
    {
        var out_ = new float[x.Length][][];
        for (int b = 0; b < x.Length; b++)
            out_[b] = SoftmaxLastDim(x[b], mask?[b]);
        return out_;
    }

    /// <summary>生成 Padding Mask：validLengths[b] 表示第 b 个序列的有效长度，其余为 pad。
    /// 返回 (batch, seqLen, seqLen)：attnMask[b][i][j] = (i &lt; validLen &amp;&amp; j &lt; validLen)。</summary>
    /// <summary>
    /// 构造 Encoder/Decoder 的 padding 掩码：根据每个样本的有效长度，标记哪些位置是实际 token，哪些位置是 PAD。
    /// </summary>
    /// <param name="batch">批大小</param>
    /// <param name="seqLen">序列统一的最大长度</param>
    /// <param name="validLengths">每个样本的真实有效长度（不含 PAD）</param>
    /// <returns>
    /// 三维布尔张量 (batch, seqLen, seqLen)，
    /// 其中 mask[b][i][j] 表示第 b 个样本中，第 i 个查询位置是否可以看到第 j 个键位置。
    /// </returns>
    public static bool[][][] CreatePaddingMask(int batch, int seqLen, int[] validLengths)
    {
        var mask = new bool[batch][][];
        for (int b = 0; b < batch; b++)
        {
            mask[b] = new bool[seqLen][];
            int len = validLengths[b];
            for (int i = 0; i < seqLen; i++)
            {
                mask[b][i] = new bool[seqLen];
                for (int j = 0; j < seqLen; j++)
                    mask[b][i][j] = i < len && j < len;
            }
        }
        return mask;
    }

    /// <summary>因果 Mask（解码器自注意力）：下三角为 true，上三角为 false，防止看到未来位置。</summary>
    /// <summary>
    /// 为解码器自注意力构造标准的“下三角”因果 Mask，保证当前位置只能看到自己和过去的位置。
    /// </summary>
    /// <param name="seqLen">序列最大长度</param>
    /// <returns>二维布尔矩阵 (seqLen, seqLen)，下三角含对角线为 true，上三角为 false</returns>
    public static bool[][] CreateCausalMask(int seqLen)
    {
        var mask = new bool[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            mask[i] = new bool[seqLen];
            for (int j = 0; j < seqLen; j++)
                mask[i][j] = j <= i;
        }
        return mask;
    }

    /// <summary>合并 padding 与 causal：padding 为 (batch, Q, K)，causal 为 (Q, K)。返回 (batch, Q, K)，true 表示可关注。</summary>
    public static bool[][][] CombinePaddingAndCausal(bool[][][] paddingMask, bool[][] causalMask)
    {
        int B = paddingMask.Length, Q = paddingMask[0].Length, K = paddingMask[0][0].Length;
        var out_ = new bool[B][][];
        for (int b = 0; b < B; b++)
        {
            out_[b] = new bool[Q][];
            for (int i = 0; i < Q; i++)
            {
                out_[b][i] = new bool[K];
                for (int j = 0; j < K; j++)
                    out_[b][i][j] = paddingMask[b][i][j] && causalMask[i][j];
            }
        }
        return out_;
    }

    /// <summary>LayerNorm：对最后一维归一化。x (rows, cols)，对每行求 mean/var，再 (x-mean)/sqrt(var+eps)。</summary>
    /// <param name="x">输入 (rows, cols)</param>
    /// <param name="eps">数值稳定项</param>
    /// <param name="mean">输出：每行均值 (rows)</param>
    /// <param name="var">输出：每行方差 (rows)</param>
    /// <returns>归一化后的 (rows, cols)</returns>
    public static float[][] LayerNormForward(float[][] x, float eps, out float[] mean, out float[] var)
    {
        int rows = x.Length, cols = x[0].Length;
        mean = new float[rows];
        var = new float[rows];
        var out_ = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            float m = 0;
            for (int j = 0; j < cols; j++)
                m += x[i][j];
            m /= cols;
            mean[i] = m;
            float v = 0;
            for (int j = 0; j < cols; j++)
            {
                float d = x[i][j] - m;
                v += d * d;
            }
            v = v / cols + eps;
            var[i] = v;
            float std = (float)Math.Sqrt(v);
            out_[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                out_[i][j] = (x[i][j] - m) / std;
        }
        return out_;
    }

    /// <summary>ReLU：max(0, x)</summary>
    public static float[][] ReLU(float[][] x)
    {
        int rows = x.Length, cols = x[0].Length;
        var r = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            r[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                r[i][j] = x[i][j] > 0 ? x[i][j] : 0;
        }
        return r;
    }

    /// <summary>用于损失计算：对每行做 log-sum-exp 与取目标列的负对数（交叉熵的一步）。logits (N, C), target (N) 为类别索引。</summary>
    public static float CrossEntropyLoss(float[][] logits, int[] target, bool[][]? paddingMask = null)
    {
        int N = logits.Length, C = logits[0].Length;
        float sumLoss = 0;
        int count = 0;
        for (int n = 0; n < N; n++)
        {
            if (paddingMask != null && n < paddingMask.Length)
            {
                bool valid = false;
                for (int j = 0; j < paddingMask[n].Length; j++)
                    if (paddingMask[n][j]) { valid = true; break; }
                if (!valid) continue;
            }
            float max = logits[n][0];
            for (int c = 1; c < C; c++)
                if (logits[n][c] > max) max = logits[n][c];
            float logSumExp = 0;
            for (int c = 0; c < C; c++)
                logSumExp += (float)Math.Exp(logits[n][c] - max);
            logSumExp = (float)Math.Log(logSumExp) + max;
            int t = target[n];
            if (t >= 0 && t < C)
            {
                sumLoss += logSumExp - logits[n][t];
                count++;
            }
        }
        return count > 0 ? sumLoss / count : 0;
    }

    /// <summary>Argmax 沿最后一维，返回每行最大值的列索引。</summary>
    public static int[] ArgmaxLastDim(float[][] x)
    {
        var idx = new int[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            int best = 0;
            for (int j = 1; j < x[i].Length; j++)
                if (x[i][j] > x[i][best]) best = j;
            idx[i] = best;
        }
        return idx;
    }

    // ========== 反向传播（Backward）==========

    /// <summary>C = A*B。给定 dLdC，返回 dLdA = dLdC*B^T，dLdB = A^T*dLdC。</summary>
    public static (float[][] dLdA, float[][] dLdB) MultiplyBackward(float[][] dLdC, float[][] a, float[][] b)
    {
        float[][] bt = Transpose(b);
        float[][] at = Transpose(a);
        float[][] dLdA = Multiply(dLdC, bt);
        float[][] dLdB = Multiply(at, dLdC);
        return (dLdA, dLdB);
    }

    /// <summary>批量乘 out[b]=x[b]*w。dLdOut(B,M,N), x(B,M,K), w(K,N)。返回 dLdX(B,M,K)，dLdW 累加到 gradW(K,N)。</summary>
    public static float[][][] MultiplyBatch3DBackward(float[][][] dLdOut, float[][][] x, float[][] w, float[][] gradW)
    {
        int B = dLdOut.Length, M = dLdOut[0].Length, N = dLdOut[0][0].Length, K = w.Length;
        float[][] wt = Transpose(w);
        var dLdX = Zeros3(B, M, K);
        for (int b = 0; b < B; b++)
        {
            var (dxb, dwb) = MultiplyBackward(dLdOut[b], x[b], w);
            dLdX[b] = dxb;
            // gradW += x[b]^T * dLdOut[b]
            var xbt = Transpose(x[b]);
            var dwAdd = Multiply(xbt, dLdOut[b]);
            for (int i = 0; i < gradW.Length; i++)
                for (int j = 0; j < gradW[i].Length; j++)
                    gradW[i][j] += dwAdd[i][j];
        }
        return dLdX;
    }

    /// <summary>ReLU 反向：dLdx = dLdOut where x&gt;0 else 0。x 为前向时的输入（ReLU 前）。</summary>
    public static float[][] ReLUBackward(float[][] dLdOut, float[][] x)
    {
        int rows = x.Length, cols = x[0].Length;
        var dLdX = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            dLdX[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                dLdX[i][j] = x[i][j] > 0 ? dLdOut[i][j] : 0;
        }
        return dLdX;
    }

    /// <summary>Softmax 沿最后一维的反向。p 为前向 softmax 输出，dLdp 为上游梯度。dL/dx = p * (dLdp - sum(dLdp.*p))。</summary>
    public static float[][] SoftmaxLastDimBackward(float[][] dLdp, float[][] p)
    {
        int rows = p.Length, cols = p[0].Length;
        var dLdx = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            float sum = 0;
            for (int j = 0; j < cols; j++)
                sum += p[i][j] * dLdp[i][j];
            dLdx[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                dLdx[i][j] = p[i][j] * (dLdp[i][j] - sum);
        }
        return dLdx;
    }

    /// <summary>3D Softmax 反向：对每个 b 调用 SoftmaxLastDimBackward。</summary>
    public static float[][][] SoftmaxLastDim3Backward(float[][][] dLdp, float[][][] p)
    {
        var dLdx = new float[p.Length][][];
        for (int b = 0; b < p.Length; b++)
            dLdx[b] = SoftmaxLastDimBackward(dLdp[b], p[b]);
        return dLdx;
    }

    /// <summary>LayerNorm 反向。x, mean, var 为前向时的输入与统计量，eps 与前向一致。</summary>
    public static float[][] LayerNormBackward(float[][] dLdOut, float[][] x, float[] mean, float[] var, float eps)
    {
        int rows = x.Length, cols = x[0].Length;
        var dLdx = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            float std = (float)Math.Sqrt(var[i]);
            float invN = 1f / cols;
            float sumDldOut = 0, sumDldOutNorm = 0;
            for (int j = 0; j < cols; j++)
            {
                sumDldOut += dLdOut[i][j];
                sumDldOutNorm += dLdOut[i][j] * (x[i][j] - mean[i]) / var[i];
            }
            dLdx[i] = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                float norm = (x[i][j] - mean[i]) / std;
                dLdx[i][j] = (dLdOut[i][j] - invN * sumDldOut - norm * invN * sumDldOutNorm) / std;
            }
        }
        return dLdx;
    }

    /// <summary>交叉熵损失对 logits 的梯度。logits(N,C), target(N), paddingMask 与 CrossEntropyLoss 一致。返回 dLdLogits(N,C)，仅对有效位置填充，无效位置为 0。有效数 count 用于平均。</summary>
    public static float[][] CrossEntropyLossBackward(float[][] logits, int[] target, bool[][]? paddingMask, out int count)
    {
        int N = logits.Length, C = logits[0].Length;
        var dLdLogits = Zeros(N, C);
        count = 0;
        for (int n = 0; n < N; n++)
        {
            if (paddingMask != null && n < paddingMask.Length)
            {
                bool valid = false;
                for (int j = 0; j < paddingMask[n].Length; j++)
                    if (paddingMask[n][j]) { valid = true; break; }
                if (!valid) continue;
            }
            int t = target[n];
            if (t < 0 || t >= C) continue;
            count++;
            float max = logits[n][0];
            for (int c = 1; c < C; c++)
                if (logits[n][c] > max) max = logits[n][c];
            float sumExp = 0;
            for (int c = 0; c < C; c++)
                sumExp += (float)Math.Exp(logits[n][c] - max);
            for (int c = 0; c < C; c++)
            {
                float p = (float)Math.Exp(logits[n][c] - max) / sumExp;
                dLdLogits[n][c] = p - (c == t ? 1f : 0f);
            }
        }
        if (count > 0)
            for (int n = 0; n < N; n++)
                for (int c = 0; c < C; c++)
                    dLdLogits[n][c] /= count;
        return dLdLogits;
    }

    /// <summary>Embedding 反向：将 dLdOut(batch,seqLen,dModel) 按 ids(batch,seqLen) 累加到 gradTable(vocabSize,dModel)。</summary>
    public static void ScatterAddIntoRows(float[][] gradTable, int[][] ids, float[][][] dLdOut)
    {
        int batch = dLdOut.Length, seqLen = dLdOut[0].Length, dModel = dLdOut[0][0].Length;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int id = ids[b][s];
                if (id < 0 || id >= gradTable.Length) continue;
                for (int d = 0; d < dModel; d++)
                    gradTable[id][d] += dLdOut[b][s][d];
            }
    }
}
