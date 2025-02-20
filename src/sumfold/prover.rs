//! This library implements SumFold prover.
// prover.rs

use ff::PrimeField;
use crate::traits::Group;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::sumfold::q_poly::build_q_polynomial; // or wherever build_q_polynomial is
use crate::spartan::sumcheck::SumcheckProof;

/// A sumcheck instance containing a function F, a vector of polynomials, and a proof.
#[derive(Clone)]
pub struct SumcheckInstance<G: Group, Scalar: PrimeField> {
    /// A function that takes a slice of Scalars and returns a Scalar.
    pub F: std::sync::Arc<dyn Fn(&[Scalar]) -> Scalar + Send + Sync>,
    /// A vector of multilinear polynomials.
    pub g_vec: Vec<MultilinearPolynomial<Scalar>>,
    /// A proof for the sumcheck protocol.
    pub proof: SumcheckProof<G>,
}

/// Implements sumfold() following the requested steps:
/// 1. F = instances[0].F
/// 2. Ensure all g_vec have the same length.
/// 3. Define n, t, b, x, etc.
/// 4. Prepare g_bj from instances.
/// 5. Prepare f_js from g_bj.
/// 6. Pick a random rho in [0..n).
/// 7. Call build_q_polynomial.
/// 8. Return (instances[rho], q_b, rho).
///
/// Output type: (SumcheckInstance<Scalar>, Scalar, MultilinearPolynomial<Scalar>)
pub fn sumfold<G: Group, Scalar: PrimeField>(
    instances: Vec<SumcheckInstance<G, Scalar>>,
    rho_int: usize,
) -> (SumcheckInstance<G, Scalar>, Scalar, MultilinearPolynomial<Scalar>) {
    // Step 1: F = instances[0].F (not strictly used below, but we store it)
    let F = instances[0].F.clone();

    // Step 2: ensure all g_vec have the same length
    let first_len = instances[0].g_vec.len();
    for inst in &instances {
        assert_eq!(
            inst.g_vec.len(),
            first_len,
            "All instances must have the same number of polynomials in g_vec"
        );
    }

    // Step 3: define n, t, b, x, etc.
    // Here we only define n = instances.len(), t = g_vec.len() for demonstration.
    // b,x typically come from the dimension of the polynomials, but we skip that detail.
    let n = instances.len();
    let t = instances[0].g_vec.len();
    let x = instances[0].g_vec[0].Z.len();
    let nu = (n as f64).log2() as usize; // nu = log₂(n)
    let l  = (x as f64).log2() as usize;  // l = log₂(x)

    // Step 4: Prepare g_bj from instances
    let g_bj: Vec<Vec<MultilinearPolynomial<Scalar>>> = instances
      .iter()
      .map(|inst| inst.g_vec.clone())
      .collect();

    // Step 5: Prepare f_js from g_bj
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

    // Step 6: pick random rho in [0..n)
    // let rho_int = n % 2;
    let rho_field = Scalar::from(rho_int as u64);

    // Step 7: call build_q_polynomial
    // For demonstration, we hardcode nu=1, l=1 or something suitable.
    // In practice, you should derive (nu, l) from the polynomial dimension.
    let q_b = build_q_polynomial(
        &f_js.clone(),
        &*F,
        rho_int, // integer in [0..n)
        nu,
        l
    );

    // Step 8: return (instances[rho], q_b, rho)
    // But your requested type is (SumcheckInstance<Scalar>, Scalar, MultilinearPolynomial<Scalar>).
    // So we put (instances[rho_int], rho_field, q_b).
    (
        instances[rho_int].clone(),
        rho_field,
        q_b
    )
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use rand::Rng;
    use std::sync::Arc;
    use pasta_curves::pallas::{Point as PallasPoint, Scalar as PallasScalar};
    use crate::traits::Group;
    use crate::spartan::polys::multilinear::MultilinearPolynomial;
    use crate::spartan::sumcheck::SumcheckProof;
    use crate::sumfold::prover::{sumfold, SumcheckInstance};
    use crate::traits::TranscriptEngineTrait;

    #[test]
    fn test_sumfold() {
        // Parameters:
        // n = 2 instances, t = 2 polynomials per instance, x = 4 evaluations per polynomial.
        let n = 2;
        let t = 2;
        let x = 4;
        let l = (x as f64).log2() as usize;    // for x=4, l=2

        // Define F as the product function.
        let F_arc: Arc<dyn Fn(&[PallasScalar]) -> PallasScalar + Send + Sync> =
            Arc::new(|vals: &[PallasScalar]| vals.iter().product());

        let mut rng = rand::thread_rng();
        let mut instances: Vec<SumcheckInstance<PallasPoint, PallasScalar>> = Vec::with_capacity(n);

        // For each instance, create a g_vec with t polynomials (each with x evaluations).
        for _ in 0..n {
            let mut g_vec = Vec::with_capacity(t);
            for _ in 0..t {
                let evals: Vec<PallasScalar> =
                    (0..x).map(|_| PallasScalar::random(&mut rng)).collect();
                g_vec.push(MultilinearPolynomial::new(evals));
            }

            // For the sumcheck proof, use poly_A = g_vec[0] and poly_B = g_vec[1].
            let mut poly_A = g_vec[0].clone();
            let mut poly_B = g_vec[1].clone();

            // Compute claim = sum_{x in [0, x)} (g_vec[0](x) * g_vec[1](x))
            let mut claim = PallasScalar::zero();
            for i in 0..x {
                claim += poly_A.Z[i] * poly_B.Z[i];
            }

            // Create a transcript using the group-associated transcript.
            let mut transcript = <PallasPoint as Group>::TE::new(b"SumFold");
            // Run prove_quad to obtain a sumcheck proof.
            // We use num_rounds = l (here, l = 2).
            let (proof, _r_vec, _final_vals) = SumcheckProof::<PallasPoint>::prove_quad(
                &claim,
                l,
                &mut poly_A,
                &mut poly_B,
                |a, b| *a * *b,
                &mut transcript,
            ).expect("prove_quad failed");

            let instance = SumcheckInstance {
                F: F_arc.clone(),
                g_vec,
                proof,
            };
            instances.push(instance);
        }

        // Call sumfold with these instances.
        // Choose a random integer rho_int in [0, n). For n=2, this is either 0 or 1.
        let rho_int = rng.gen_range(0..n);
        let (chosen_inst, _, q_b) = sumfold(instances, rho_int);

        // Compute T = sum_{x in [0, x)} F( [g_vec[0](x), g_vec[1](x)] )
        let mut T = PallasScalar::zero();
        for x_idx in 0..x {
            let val = (chosen_inst.F)(&[chosen_inst.g_vec[0].Z[x_idx],
                                          chosen_inst.g_vec[1].Z[x_idx]]);
            T += val;
        }

        // In our sumfold, build_q_polynomial was called on chosen_inst.g_vec.
        // The folded polynomial q_b is in nu variables (nu = 1 for n=2),
        // so its evaluation vector has length 2. We evaluate it at b = rho.
        // Since our construction in sumfold uses the bit assignment where
        // the b bits occupy the top bits, we simply index by rho_int.
        let q_val = q_b.Z[rho_int];

        assert_eq!(
            q_val, T,
            "Mismatch: q_b(rho) = {:?}, T = {:?}",
            q_val, T
        );
    }
}
