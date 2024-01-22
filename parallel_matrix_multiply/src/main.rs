use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul};

fn multiply_single_threaded<T: Default + Copy + AddAssign + Mul<Output = T>>(
    a: Vec<Vec<T>>,
    b: Vec<Vec<T>>,
) -> Vec<Vec<T>> {
    let mut result = vec![vec![T::default(); b[0].len()]; a.len()];

    for i in 0..a.len() {
        for j in 0..b[0].len() {
            for k in 0..b.len() {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn compute_inner_prods<T: Default + Copy + Add<Output = T> + Mul<Output = T>>(
    row: &[T],
    b: &[Vec<T>],
) -> Vec<T> {
    (0..b[0].len())
        .map(|j| (0..b.len()).fold(T::default(), |acc, i| acc + row[i] * b[i][j]))
        .collect()
}

fn multiply_multi_threaded<
    T: Default + Copy + AddAssign + Add<Output = T> + Mul<Output = T> + Sync + Send,
>(
    a: Vec<Vec<T>>,
    b: Vec<Vec<T>>,
) -> Vec<Vec<T>> {
    let mut unordered_rows = (0..a.len())
        .into_par_iter()
        .map(move |i| {
            let a_row = &a[i];

            (i, compute_inner_prods(a_row, &b))
        })
        .collect::<Vec<_>>();

    unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));

    unordered_rows.into_iter().map(|(_, row)| row).collect()
}

fn gen_random_matrix<T>(n: usize, m: usize) -> Vec<Vec<T>>
where
    Standard: Distribution<T>,
    T: Default + Copy,
{
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| (0..m).map(|_| rng.gen::<T>()).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

fn main() {
    for pw2 in 0..9 {
        let n = 1 << pw2;
        let a: Vec<Vec<i64>> = gen_random_matrix::<i16>(n, n)
            .into_iter()
            .map(|v| v.into_iter().map(|i| i as i64).collect())
            .collect();

        let multi_start = std::time::Instant::now();
        let multi_res = multiply_multi_threaded(a.clone(), a.clone());
        let multi_time = multi_start.elapsed();

        let single_start = std::time::Instant::now();
        let single_res = multiply_single_threaded(a.clone(), a.clone());
        let single_time = single_start.elapsed();

        assert_eq!(multi_res, single_res);

        println!(
            "n: {}, single_time: {:?}, multi_time: {:?}",
            n, single_time, multi_time
        );
    }
}
