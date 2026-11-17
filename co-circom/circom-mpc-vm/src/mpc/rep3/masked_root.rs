//! Masked-root evaluation of inverse power maps in the REP3 MPC-VM.
//!
//! An inverse power map is `x ↦ x^e` with `e·d ≡ 1 (mod p − 1)` for a small public `d`, such as
//! the inverse S-box `x^(1/5)` of Rescue. Square-and-multiply needs about 1.5·log2(p) sequential
//! multiplications for it. The masked root of eprint 2019/426 (Appendix C) needs one opening:
//!
//! 1. offline, in batches: random shares `[u]`, `[t]`, `[r]`; open `u·t` (uniform), set
//!    `[u⁻¹] = [t]·(u·t)⁻¹`, compute `[u^d]` with square-and-multiply on the whole batch, and
//!    convert `[r]` to a binary sharing `[r]_B` with the upstream conversion;
//! 2. online, a zero test that opens only the uniform `s = x + r`: `x = 0` exactly when the
//!    bits of `r` equal the public bits of `s`, so `[z] = [x = 0]` is the upstream binary zero
//!    test of `[r]_B ⊕ s`, injected back into the field;
//! 3. online: open `c = (x + z)·u^d`; then `x^e = c^e·[u⁻¹] − [z]`.
//!
//! `x + z` is never zero, and `u^d` is uniform on the nonzero field elements because `d` is
//! coprime to `p − 1`, so every opened `c` is a fresh uniform nonzero value whatever `x` is.
//! The zero test keeps the case `x = 0` from opening as `c = 0`: for `x = 0` the parties compute
//! `1^e − 1 = 0` without learning that `x` was zero.

use std::{any::Any, cell::RefCell};

use ark_ff::PrimeField;
use itertools::izip;
use mpc_core::{
    protocols::rep3::{
        Rep3PrimeFieldShare, Rep3State, Rep3UintShare, arithmetic, binary, conversion,
    },
    uint::FieldUint,
};
use mpc_net::Network;

type Share<F> = Rep3PrimeFieldShare<F>;

/// The largest `d` of an inverse power map evaluated with a masked root.
const MAX_DEGREE: u64 = 15;
/// Masks prepared by the first batch of a witness extension; later batches double, up to `MAX_BATCH`.
const FIRST_BATCH: usize = 128;
const MAX_BATCH: usize = 1024;

/// Returns `d` with `e·d ≡ 1 (mod p − 1)` and `2 ≤ d ≤ MAX_DEGREE`, if there is one.
pub(super) fn inverse_degree<F: PrimeField>(e: F) -> Option<u64> {
    if e.is_zero() {
        return None;
    }
    let e = e.into_bigint();
    let mut p_minus_1 = F::MODULUS.as_ref().to_vec();
    // p is odd, so subtracting one never borrows
    p_minus_1[0] -= 1;
    (2..=MAX_DEGREE).find(|&d| {
        // e < p, so e·d = 1 + k·(p − 1) for some k < d exactly when e·d ≡ 1 (mod p − 1)
        let ed = mul_small(e.as_ref(), d);
        (0..d).any(|k| {
            let mut rhs = mul_small(&p_minus_1, k);
            add_one(&mut rhs);
            rhs == ed
        })
    })
}

/// `x^e` for a shared `x` when `e` is an inverse power exponent, otherwise `None`.
pub(super) fn inverse_power<F: PrimeField + FieldUint, N: Network>(
    x: Share<F>,
    e: F,
    net: &N,
    state: &mut Rep3State,
) -> eyre::Result<Option<Share<F>>> {
    let Some(degree) = inverse_degree(e) else {
        return Ok(None);
    };
    let mask = take_mask::<F, N>(degree, net, state)?;
    let shifted = arithmetic::open(x + mask.offset, net)?;
    let difference = binary::xor_public(&mask.offset_bits, &shifted.to_uint(), state.id);
    let is_zero = binary::is_zero(&difference, net, state)?;
    let is_zero = conversion::bit_inject(&is_zero, net, state)?;
    let c = arithmetic::mul_open(x + is_zero, mask.power, net, state)?;
    if c.is_zero() {
        eyre::bail!("masked root opened zero: the masks are inconsistent between the parties");
    }
    let root = arithmetic::mul_public(mask.inverse, c.pow(e.into_bigint()));
    Ok(Some(root - is_zero))
}

/// Preprocessed randomness for one inverse power: `[u^d]`, `[u⁻¹]`, `[r]` and `[r]_B`.
struct Mask<F: FieldUint> {
    power: Share<F>,
    inverse: Share<F>,
    offset: Share<F>,
    offset_bits: Rep3UintShare<F>,
}

/// Prepared masks of one REP3 state.
struct MaskPool<F: FieldUint> {
    owner: usize,
    degree: u64,
    next_batch: usize,
    masks: Vec<Mask<F>>,
}

thread_local! {
    static POOL: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
}

fn take_mask<F: FieldUint, N: Network>(
    degree: u64,
    net: &N,
    state: &mut Rep3State,
) -> eyre::Result<Mask<F>> {
    let owner = std::ptr::from_ref::<Rep3State>(state) as usize;
    let mut pool = POOL
        .with(|cell| cell.borrow_mut().take())
        .and_then(|pool| pool.downcast::<MaskPool<F>>().ok())
        .filter(|pool| pool.owner == owner && pool.degree == degree)
        .unwrap_or_else(|| {
            Box::new(MaskPool {
                owner,
                degree,
                next_batch: FIRST_BATCH,
                masks: Vec::new(),
            })
        });
    while pool.masks.is_empty() {
        let size = pool.next_batch;
        pool.next_batch = (size * 2).min(MAX_BATCH);
        pool.masks = prepare_masks(size, degree, net, state)?;
    }
    let mask = pool.masks.pop().expect("the pool was refilled");
    POOL.with(|cell| *cell.borrow_mut() = Some(pool as Box<dyn Any>));
    Ok(mask)
}

/// `size` masks for fresh random `u ≠ 0` and `r`, in one opening, the multiplication rounds of
/// `u^d` and one batched binary conversion.
fn prepare_masks<F: FieldUint, N: Network>(
    size: usize,
    degree: u64,
    net: &N,
    state: &mut Rep3State,
) -> eyre::Result<Vec<Mask<F>>> {
    let u = (0..size)
        .map(|_| arithmetic::rand(state))
        .collect::<Vec<_>>();
    let t = (0..size)
        .map(|_| arithmetic::rand(state))
        .collect::<Vec<Share<F>>>();
    let r = (0..size)
        .map(|_| arithmetic::rand(state))
        .collect::<Vec<Share<F>>>();
    let r_bits = conversion::a2b_selector_many(&r, net, state)?;
    let ut = arithmetic::mul_open_vec(&u, &t, net, state)?;
    let mut power = u.clone();
    for bit in (0..63 - degree.leading_zeros()).rev() {
        power = arithmetic::mul_vec(&power, &power, net, state)?;
        if (degree >> bit) & 1 == 1 {
            power = arithmetic::mul_vec(&power, &u, net, state)?;
        }
    }
    // u·t = 0 means u = 0 or t = 0; every party sees the same opening and drops the same pairs
    Ok(izip!(power, t, ut, r, r_bits)
        .filter_map(|(power, t, ut, offset, offset_bits)| {
            ut.inverse().map(|inv| Mask {
                power,
                inverse: arithmetic::mul_public(t, inv),
                offset,
                offset_bits,
            })
        })
        .collect())
}

fn mul_small(limbs: &[u64], factor: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(limbs.len() + 1);
    let mut carry = 0u128;
    for &limb in limbs {
        let product = u128::from(limb) * u128::from(factor) + carry;
        out.push(product as u64);
        carry = product >> 64;
    }
    out.push(carry as u64);
    out
}

fn add_one(limbs: &mut [u64]) {
    for limb in limbs {
        let (sum, overflow) = limb.overflowing_add(1);
        *limb = sum;
        if !overflow {
            return;
        }
    }
}
