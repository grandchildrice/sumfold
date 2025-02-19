use ff::PrimeField;
use crate::spartan::polys::multilinear::MultilinearPolynomial;

/// j番目の多項式群 g_{0,j}(x), g_{1,j}(x), ..., g_{(2^ν - 1), j}(x)
/// が与えられたとき、
/// f_j(b,x) = Σ_{i in {0,1}^ν} eq(b, i) * g_{i,j}(x)
/// を (ν + m) 変数の MLE として構築し、返す。
///
/// - gs_for_j: 長さ 2^ν のスライスで、各要素は m 変数の MultilinearPolynomial。
/// - 戻り値: (ν + m) 変数の MultilinearPolynomial (dense representation)。
pub fn build_fj_polynomial<Scalar: PrimeField>(
    gs_for_j: &[MultilinearPolynomial<Scalar>],
) -> MultilinearPolynomial<Scalar> {
    let num_b = gs_for_j.len();
    let log_num_b = (num_b as f64).log2();
    assert!((1 << (log_num_b as usize)) == num_b, "gs_for_j.len() must be 2^ν");
    let nu = log_num_b as usize;

    let m = gs_for_j[0].get_num_vars();
    for b in 1..num_b {
        assert_eq!(gs_for_j[b].get_num_vars(), m, "all g_{{b,j}}(x) must have the same number of variables");
    }

    let new_num_vars = nu + m;
    let new_len = 1 << new_num_vars;
    let mut f_j_evals = vec![Scalar::ZERO; new_len];

    // 修正：インデックスを (x << nu) + b とする
    for b in 0..num_b {
        let g_bj_evals = &gs_for_j[b].Z;
        for x in 0..(1 << m) {
            let big_index = (b << m) + x;
            f_j_evals[big_index] = g_bj_evals[x];
        }
    }

    MultilinearPolynomial::new(f_j_evals)
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use pasta_curves::Fp;
    use crate::spartan::polys::multilinear::MultilinearPolynomial;
    use super::*;

    #[test]
    fn test_multi_build_fj_polynomial() {
        test_build_fj_polynomial(2, 2);
        test_build_fj_polynomial(4, 4);
        test_build_fj_polynomial(4, 8);
    }

    fn test_build_fj_polynomial(nu: usize, m: usize) {
        let num_b = 1 << nu; // 2^ν
        let num_x = 1 << m;  // 2^m

        // ランダムな g_{b,j}(x) を用意する。
        // 今回は「j 番目固定」相当なので、b=0..3 に対して1本ずつMLEを作るだけ。
        // ここでは "j番目" という概念上、スライス gs_for_j のみを構築すればよい。
        let mut rng = rand::thread_rng();
        let mut gs_for_j = Vec::with_capacity(num_b);
        for _b in 0..num_b {
            // m=2変数 => 評価点数は4
            let evals: Vec<Fp> = (0..num_x).map(|_| Fp::random(&mut rng)).collect();
            let mlp = MultilinearPolynomial::new(evals);
            gs_for_j.push(mlp);
        }

        // build_fj_polynomial で f_j(b,x) を構築
        let f_j = build_fj_polynomial(&gs_for_j);

        // f_j は (ν + m) = 4 変数 => 評価点数は 2^4=16
        assert_eq!(f_j.get_num_vars(), nu + m);
        assert_eq!(f_j.len(), 1 << (nu + m));

        // 各 b, x について f_j(b, x) が g_{b,j}(x) と一致するかチェック
        for b in 0..num_b {
            for x in 0..num_x {
                let expected = gs_for_j[b].Z[x];

                // 入力 (b_bits, x_bits) を作り、f_j.evaluate(...) で確認
                // b_bits: νビット, x_bits: mビット
                let mut point = Vec::with_capacity(nu + m);
                for bit_index in (0..nu).rev() {
                    let bit_val = if ((b >> bit_index) & 1) == 1 { Fp::ONE } else { Fp::ZERO };
                    point.push(bit_val);
                }
                for bit_index in (0..m).rev() {
                    let bit_val = if ((x >> bit_index) & 1) == 1 { Fp::ONE } else { Fp::ZERO };
                    point.push(bit_val);
                }

                let actual = f_j.evaluate(&point);
                assert_eq!(actual, expected,
                    "Mismatch at b={}, x={:?}, expected={:?}, got={:?}",
                    b, x, expected, actual);
            }
        }
    }

}