//! This library implements Q(b) polynomia for SumFold.
use ff::PrimeField;
use rayon::prelude::*;
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::sumfold::fj_poly::build_bx_point;

/// Converts a decimal integer `val` into a bit vector of length `num_bits`,
/// from the most significant bit to the least significant bit.
/// E.g., if val=6 (0b110) and num_bits=3, this returns [1, 1, 0].
fn decimal_to_bits_msb_first<Scalar: PrimeField>(val: usize, num_bits: usize) -> Vec<Scalar> {
    let mut bits = Vec::with_capacity(num_bits);
    // Iterate from the highest bit down to 0
    for i in (0..num_bits).rev() {
        let bit_val = if ((val >> i) & 1) == 1 {
            Scalar::ONE
        } else {
            Scalar::ZERO
        };
        bits.push(bit_val);
    }
    bits
}

/// Constructs Q(b) = eq(rho, b) * ( sum_{x in {0,1}^l} F( f_1(b,x), ..., f_t(b,x) ) ).
///
/// - `f_js`: [f_1, ..., f_t], each an MLE in (nu + l) variables (bits of b + bits of x).
/// - `F`: A function that takes t field elements and returns a single field element (e.g., product).
/// - `rho`: decimal integer in [0..2^nu).
/// - `nu`: number of bits for b.
/// - `l`: number of bits for x.
///
/// Returns a MultilinearPolynomial in `nu` variables (the variable b).
pub fn build_q_polynomial<Scalar: PrimeField>(
    f_js: &[MultilinearPolynomial<Scalar>],
    F: &(dyn Fn(&[Scalar]) -> Scalar + Sync),
    rho: usize,
    nu: usize,
    l: usize,
) -> MultilinearPolynomial<Scalar> {
    // 1) Build eq(rho, ·) as an EqPolynomial using MSB-first bits of rho.
    let rho_bits = decimal_to_bits_msb_first::<Scalar>(rho, nu);
    let eq_poly_rho = EqPolynomial::new(rho_bits);

    // 2) We'll build Q in dense form for b in [0..2^nu].
    let len_n = 1 << nu;
    let len_x = 1 << l;
    let t = f_js.len();
    let mut q_evals = vec![Scalar::ZERO; len_n];

    // 3) For each b in [0..2^nu), compute eq(rho,b) and sum over x.
    //    Q(b) = eq(rho,b) * Σ_x F( f_1(b,x), ..., f_t(b,x) ).
    q_evals
        .par_iter_mut()
        .enumerate()
        .for_each(|(b_val, q_out)| {
            // Evaluate eq(rho, b_val)
            let b_bits = decimal_to_bits_msb_first::<Scalar>(b_val, nu);
            let eq_val = eq_poly_rho.evaluate(&b_bits);

            if eq_val.is_zero().into() {
                // Q(b_val) = 0 if eq(rho,b_val)=0
                *q_out = Scalar::ZERO;
            } else {
                // sum_x F(...)
                let mut sum_val = Scalar::ZERO;
                for x_val in 0..len_x {
                    // Evaluate each f_j(b_val,x_val)
                    let bx_point = build_bx_point::<Scalar>(b_val, x_val, nu, l);

                    let mut f_inputs = Vec::with_capacity(t);
                    for f_j in f_js {
                        let val_j = f_j.evaluate(&bx_point);
                        f_inputs.push(val_j);
                    }
                    // Apply F to these t values
                    let f_val = F(&f_inputs);
                    sum_val += f_val;
                }
                // Multiply by eq(rho,b_val)
                *q_out = eq_val * sum_val;
            }
        });

    // 4) Return Q(b) as a multilinear polynomial in nu variables.
    MultilinearPolynomial::new(q_evals)
}

/// Example: a simple product function F(g1, g2, ..., g_t) = g1 * g2 * ... * g_t
pub fn product_F<Scalar: PrimeField>(vals: &[Scalar]) -> Scalar {
    let mut acc = Scalar::ONE;
    for v in vals {
        acc *= *v;
    }
    acc
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use rand::Rng;

    // Suppose we use pasta_curves::Fp as our field Scalar.
    use pasta_curves::Fp as Scalar;

    // Adjust these imports to match your project structure.
    use crate::spartan::polys::multilinear::MultilinearPolynomial;
    use crate::sumfold::q_poly::{build_q_polynomial, product_F};

    #[test]
    fn test_build_q_polynomials() {
      test_build_q_polynomial_simple(2, 2, 2);
      test_build_q_polynomial_simple(8, 4, 2);
      test_build_q_polynomial_simple(16, 4, 2);
      test_build_q_polynomial_simple(32, 4, 2);
      test_build_q_polynomial_simple(64, 4, 2);
      test_build_q_polynomial_simple(1024, 128, 2);
    }

    fn test_build_q_polynomial_simple(n: usize, x: usize, t: usize) {
        // 1) Set parameters:
        //    n: number of b values = 2^nu,
        //    x: number of x values = 2^l,
        //    t: number of polynomials per b.
        // Compute nu and l from n and x.
        let nu = (n as f64).log2() as usize; // nu = log₂(n)
        let l  = (x as f64).log2() as usize;  // l = log₂(x)

        // 2) For each b in [0..n), create t random 1-variable polynomials g_b^j.
        //    Each such polynomial has 2 evaluations (since x=2 => 1 variable => 2^1=2).
        println!("building g_bj...");
        let mut rng = rand::thread_rng();
        let mut g_bj: Vec<Vec<MultilinearPolynomial<Scalar>>> = Vec::with_capacity(n);
        for _b in 0..n {
            let mut polys_for_this_b = Vec::with_capacity(t);
            for _j in 0..t {
                let evals: Vec<Scalar> = (0..x).map(|_| Scalar::random(&mut rng)).collect();
                polys_for_this_b.push(MultilinearPolynomial::new(evals));
            }
            g_bj.push(polys_for_this_b);
        }

        // 3) Build f_j(b,x) in (nu+l) variables.
        //    Since nu + l = (log₂(n) + log₂(x)), the total number of evaluation points is 2^(nu+l).
        println!("building f_j(b,x)...");
        let size = 1 << (nu + l);
        let mut f_js = Vec::with_capacity(t);
        for j in 0..t {
            let mut f_eval = vec![Scalar::ZERO; size]; // size = 2^(nu+l)
            // The index is computed as (b_val << l) + x_val,
            // because the b-bits are the top bits and x-bits are the bottom bits.
            for b_val in 0..n {
                for x_val in 0..x {
                    let index = (b_val << l) + x_val;
                    f_eval[index] = g_bj[b_val][j].Z[x_val];
                }
            }
            f_js.push(MultilinearPolynomial::new(f_eval));
        }

        // 4) Choose a random rho in [0..n)
        let rho = rng.gen_range(0..n);

        // 5) Build Q(b) = eq(rho,b) * sum_x( product_{j=1..t} f_j(b,x) )
        println!("building Q...");
        let Q = build_q_polynomial(&f_js, &product_F, rho, nu, l);

        // 6) Let r_b = rho, evaluate Q(r_b)
        let r_b = rho;
        let Q_r_b = Q.Z[r_b]; // Q is in nu variables => length = 2^nu

        // 7) Check that Q(r_b) == sum_{x in [0..x)} of product( g_{r_b}^j(x) for j=1..t ).
        println!("checking Q(r_b)...");
        let mut sum_val = Scalar::ZERO;
        for x_val in 0..x {
            let mut prod_val = Scalar::ONE;
            for j in 0..t {
                prod_val *= g_bj[r_b][j].Z[x_val];
            }
            sum_val += prod_val;
        }

        assert_eq!(
            Q_r_b, sum_val,
            "Mismatch: Q(r_b) = {:?}, sum_x(product(g_r_b^j(x))) = {:?}",
            Q_r_b, sum_val
        );
    }
}
