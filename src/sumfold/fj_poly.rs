//! This library implements fj(x) polynomia for SumFold.
use ff::PrimeField;
use rayon::prelude::*;
use crate::spartan::polys::multilinear::MultilinearPolynomial;

/// Given a set of polynomials for the j-th index:
/// g_{0,j}(x), g_{1,j}(x), ..., g_{(2^ν - 1), j}(x),
/// this function constructs and returns the multilinear extension (MLE)
/// defined by:
///   f_j(b,x) = Σ_{i in {0,1}^ν} eq(b, i) * g_{i,j}(x)
/// as an MLE in (ν + m) variables.
///
/// - gs_for_j: A slice of length 2^ν, where each element is a MultilinearPolynomial in m variables.
/// - Returns: A MultilinearPolynomial in (ν + m) variables (dense representation).
pub fn build_fj_polynomial<Scalar: PrimeField>(
    gs_for_j: &[MultilinearPolynomial<Scalar>],
) -> MultilinearPolynomial<Scalar> {
    let num_b = gs_for_j.len();
    let log_num_b = (num_b as f64).log2();
    assert!((1 << (log_num_b as usize)) == num_b, "gs_for_j.len() must be 2^ν");
    let nu = log_num_b as usize;

    let l = gs_for_j[0].get_num_vars();
    for b in 1..num_b {
        assert_eq!(
            gs_for_j[b].get_num_vars(),
            l,
            "all g_{{b,j}}(x) must have the same number of variables"
        );
    }

    let new_num_vars = nu + l;
    let new_len = 1 << new_num_vars;
    let mut f_j_evals = vec![Scalar::ZERO; new_len];

    let block_size = 1 << l; // Size of each block
    // Using parallelization (with rayon)
    f_j_evals
        .par_chunks_mut(block_size)
        .enumerate()
        .for_each(|(b, chunk)| {
            // Copy gs_for_j[b].Z into the chunk
            chunk.copy_from_slice(&gs_for_j[b].Z);
        });

    MultilinearPolynomial::new(f_j_evals)
}

/// Given decimal representations of b and x,
/// this function converts them internally into their bit representation (B1,...,Bν, X1,...,Xm)
/// and evaluates f_j(b,x).
///
/// - f: A MultilinearPolynomial in (ν + m) variables (constructed via build_fj_polynomial)
/// - b: Decimal representation of b. An integer with ν bits (0 <= b < 2^ν)
/// - x: Decimal representation of x. An integer with m bits (0 <= x < 2^m)
/// - nu: The number of bits required to represent b.
/// - m: The number of bits required to represent x.
pub fn evaluate_fj_at_decimals<Scalar: PrimeField>(
    f: &MultilinearPolynomial<Scalar>,
    b: usize,
    x: usize,
    nu: usize,
    l: usize,
) -> Scalar {
    // Concatenate b_bits and x_bits to form the evaluation input vector.
    // Since the vectors have already been produced in parallel, a sequential extend is sufficient here.
    let point = build_bx_point(b, x, nu, l);
    f.evaluate(&point)
}

/// Builds the assignment vector (b,x) in (nu + l) bits, also from MSB to LSB.
/// The first `nu` bits correspond to `b`, and the next `l` bits to `x`.
pub fn build_bx_point<Scalar: PrimeField>(b_val: usize, x_val: usize, nu: usize, l: usize) -> Vec<Scalar> {
    let mut point = Vec::with_capacity(nu + l);

    // b in MSB-first order
    for i in (0..nu).rev() {
        let bit_b = if ((b_val >> i) & 1) == 1 {
            Scalar::ONE
        } else {
            Scalar::ZERO
        };
        point.push(bit_b);
    }

    // x in MSB-first order
    for i in (0..l).rev() {
        let bit_x = if ((x_val >> i) & 1) == 1 {
            Scalar::ONE
        } else {
            Scalar::ZERO
        };
        point.push(bit_x);
    }

    point
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use pasta_curves::Fp;
    use rayon::prelude::*;
    use crate::spartan::polys::multilinear::MultilinearPolynomial;
    use super::*;

    #[test]
    fn test_multi_build_fj_polynomial() {
        // Test with different numbers of b and x values.
        // The arguments represent the number of b and x values respectively (must be powers of 2).
        // For example, (b, x) = (2,2) => nu = log₂(2) = 1, m = log₂(2) = 1.
        // (b, x) = (4,4) => nu = 2, m = 2, etc.
        test_build_fj_polynomial(2, 2);
        test_build_fj_polynomial(4, 2);
        test_build_fj_polynomial(4, 4);
        test_build_fj_polynomial(8, 2);
        test_build_fj_polynomial(8, 4);
        test_build_fj_polynomial(16, 16);
        test_build_fj_polynomial(64, 64);
        test_build_fj_polynomial(128, 128);
    }

    /// b: the number of b values (must be 2^ν)
    /// x: the number of x values (must be 2^m)
    /// This function calculates nu and m from b and x.
    fn test_build_fj_polynomial(n: usize, x: usize) {
        let nu = (n as f64).log2() as usize; // log₂(b)
        let l = (x as f64).log2() as usize;  // log₂(x)

        // Prepare random g_{b,j}(x) polynomials.
        // In this test, we assume the j-th index is fixed,
        // so we only need to construct one MLE for each b.
        // For the purpose of the "j-th index", we only need to build the slice gs_for_j.
        println!("building gs_for_j...");
        let gs_for_j: Vec<MultilinearPolynomial<Fp>> = (0..n)
            .into_par_iter()
            .map(|_| {
                // Generate an independent RNG for each thread.
                let mut rng = rand::thread_rng();
                // Generate num_x evaluation values.
                let evals: Vec<Fp> = (0..x)
                    .map(|_| Fp::random(&mut rng))
                    .collect();
                MultilinearPolynomial::new(evals)
            })
            .collect();

        println!("building f_j(b,x)...");
        // Construct f_j(b,x) using build_fj_polynomial.
        let f_j = build_fj_polynomial(&gs_for_j);
        println!("...done");
        // f_j is a polynomial in (ν + m) variables, so the number of evaluation points is 2^(ν + m).
        assert_eq!(f_j.get_num_vars(), nu + l);
        assert_eq!(f_j.len(), 1 << (nu + l));

        // For each b and x, use evaluate_fj_at_decimals to check that the evaluation of f_j(b,x)
        // matches gs_for_j[b].Z[x].
        (0..(n * x)).into_par_iter().for_each(|idx| {
            let b_val = idx / x;
            let x_val = idx % x;
            println!("running evaluation at b={}, x={}...", b_val, x_val);
            let expected = gs_for_j[b_val].Z[x_val];
            let actual = evaluate_fj_at_decimals(&f_j, b_val, x_val, nu, l);
            assert_eq!(actual, expected,
                "Mismatch at b={}, x={}, expected={:?}, got={:?}",
                b_val, x_val, expected, actual);
            println!("...done");
        });

    }
}
