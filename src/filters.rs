//! Moving mean, median and percentile with the edge rule and window placement of
//! scipy.ndimage (mode "reflect", origin 0), which the defaults were tuned with.

use crate::grid::Grid;

/// Index into 0..n for a position that may fall outside it, mirrored as d c b a | a b c d | d c b a.
fn reflect(i: isize, n: usize) -> usize {
    // most calls are inside the range; the modulo is only needed at the edges
    if i >= 0 && (i as usize) < n {
        return i as usize;
    }
    let period = 2 * n as isize;
    let m = i.rem_euclid(period);
    if m < n as isize {
        m as usize
    } else {
        (period - 1 - m) as usize
    }
}

/// Mean over a window of `size` values; the window of output i starts at i - size / 2.
pub fn mean_1d(v: &[f64], size: usize) -> Vec<f64> {
    let n = v.len();
    if n == 0 || size <= 1 {
        return v.to_vec();
    }
    let start = size as isize / 2;
    let mut sums = Vec::with_capacity(n + size);
    sums.push(0.0);
    let mut total = 0.0;
    for j in 0..(n + size - 1) {
        total += v[reflect(j as isize - start, n)];
        sums.push(total);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((sums[i + size] - sums[i]) / size as f64);
    }
    out
}

/// Ordering for floats, so a window can hold f32 and f64 values without converting either.
trait Ordered: Copy {
    fn order(self, other: Self) -> std::cmp::Ordering;
}

impl Ordered for f32 {
    fn order(self, other: Self) -> std::cmp::Ordering {
        self.total_cmp(&other)
    }
}

impl Ordered for f64 {
    fn order(self, other: Self) -> std::cmp::Ordering {
        self.total_cmp(&other)
    }
}

/// The values of a moving window kept in order, so a step costs a search and a short shift.
struct Sorted<T>(Vec<T>);

impl<T: Ordered> Sorted<T> {
    fn new(values: &[T]) -> Self {
        let mut v = values.to_vec();
        v.sort_unstable_by(|a, b| a.order(*b));
        Sorted(v)
    }

    /// Remove `old` and insert `new` with one shift of the values between them.
    /// total_cmp calls two values equal only when their bits are, so the removed one is exact.
    fn replace(&mut self, old: T, new: T) {
        let v = &mut self.0;
        let from = v.partition_point(|x| x.order(old).is_lt());
        let to = v.partition_point(|x| x.order(new).is_lt());
        if to > from {
            v.copy_within(from + 1..to, from);
            v[to - 1] = new;
        } else {
            v.copy_within(to..from, to + 1);
            v[to] = new;
        }
    }
}

/// The value of rank `rank` (0 = smallest) in a window of `size` values.
pub fn rank_1d(v: &[f64], size: usize, rank: usize) -> Vec<f64> {
    let n = v.len();
    if n == 0 || size <= 1 {
        return v.to_vec();
    }
    let start = size as isize / 2;
    let first: Vec<f64> = (0..size).map(|j| v[reflect(j as isize - start, n)]).collect();
    let mut window = Sorted::new(&first);

    let mut out = Vec::with_capacity(n);
    out.push(window.0[rank]);
    for i in 1..n {
        window.replace(
            v[reflect(i as isize - 1 - start, n)],
            v[reflect(i as isize + size as isize - 1 - start, n)],
        );
        out.push(window.0[rank]);
    }
    out
}

pub fn median_1d(v: &[f64], size: usize) -> Vec<f64> {
    rank_1d(v, size, size / 2)
}

/// Rank filter over `size_b` bins by `size_t` frames of a grid; frames are independent, so long
/// grids are split over the cores.
pub fn rank_2d(g: &Grid<f32>, size_b: usize, size_t: usize, rank: usize) -> Grid<f32> {
    let mut out = Grid::new(g.bins, g.frames, 0.0f32);
    if g.bins == 0 || g.frames == 0 {
        return out;
    }
    let workers = if cfg!(miri) || g.frames < 256 {
        1
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    };
    if workers == 1 {
        rank_2d_frames(g, size_b, size_t, rank, 0, &mut out.data);
        return out;
    }
    let chunk = g.frames.div_ceil(workers);
    std::thread::scope(|scope| {
        for (w, part) in out.data.chunks_mut(chunk * g.bins).enumerate() {
            scope.spawn(move || rank_2d_frames(g, size_b, size_t, rank, w * chunk, part));
        }
    });
    out
}

/// Frames `start..start + out.len() / bins` of a rank filter, written into `out` frame by frame.
fn rank_2d_frames(g: &Grid<f32>, size_b: usize, size_t: usize, rank: usize, start: usize, out: &mut [f32]) {
    let (sb, st) = (size_b as isize / 2, size_t as isize / 2);
    for local in 0..out.len() / g.bins {
        let t = start + local;
        let frames: Vec<&[f32]> =
            (0..size_t).map(|dt| g.frame(reflect(t as isize + dt as isize - st, g.frames))).collect();
        let bin = |j: isize| reflect(j - sb, g.bins);

        let mut first = Vec::with_capacity(size_b * size_t);
        for f in &frames {
            for db in 0..size_b {
                first.push(f[bin(db as isize)]);
            }
        }
        let mut window = Sorted::new(&first);
        let frame = &mut out[local * g.bins..(local + 1) * g.bins];
        frame[0] = window.0[rank];

        for (b, slot) in frame.iter_mut().enumerate().skip(1) {
            let (old, new) = (bin(b as isize - 1), bin(b as isize + size_b as isize - 1));
            for f in &frames {
                window.replace(f[old], f[new]);
            }
            *slot = window.0[rank];
        }
    }
}

pub fn median_2d(g: &Grid<f32>, size_b: usize, size_t: usize) -> Grid<f32> {
    rank_2d(g, size_b, size_t, size_b * size_t / 2)
}

/// The rank scipy.ndimage.percentile_filter picks for a percentile of `size` values.
pub fn percentile_rank(size: usize, percentile: f64) -> usize {
    let rank = (size as f64 * percentile / 100.0) as usize;
    rank.min(size - 1)
}

/// Mean over `size` frames of every bin.
/// The sums are those of mean_1d, added in the same order, but frame by frame through memory.
pub fn mean_time(g: &Grid<f32>, size: usize) -> Grid<f32> {
    let n = g.frames;
    if n == 0 || size <= 1 {
        return g.clone();
    }
    let start = size as isize / 2;
    let mut out = Grid::new(g.bins, n, 0.0f32);

    // running sums of every bin; only the last size + 1 of them are needed
    let mut ring = vec![vec![0.0f64; g.bins]; size + 1];
    let mut total = vec![0.0f64; g.bins];
    for j in 0..(n + size - 1) {
        let frame = g.frame(reflect(j as isize - start, n));
        for (t, v) in total.iter_mut().zip(frame) {
            *t += *v as f64;
        }
        ring[(j + 1) % (size + 1)].copy_from_slice(&total);

        if j + 1 >= size {
            let i = j + 1 - size;
            let (hi, lo) = (&ring[(i + size) % (size + 1)], &ring[i % (size + 1)]);
            for (b, o) in out.frame_mut(i).iter_mut().enumerate() {
                *o = ((hi[b] - lo[b]) / size as f64) as f32;
            }
        }
    }
    out
}

/// Mean over `size` bins of every frame.
pub fn mean_freq(g: &Grid<f32>, size: usize) -> Grid<f32> {
    let mut out = g.clone();
    for t in 0..g.frames {
        let col: Vec<f64> = g.frame(t).iter().map(|v| *v as f64).collect();
        for (o, v) in out.frame_mut(t).iter_mut().zip(mean_1d(&col, size)) {
            *o = v as f32;
        }
    }
    out
}

/// Median of a slice as numpy computes it (mean of the two middle values for an even count).
pub fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    let hi = nth(&mut v, n / 2);
    if n % 2 == 1 {
        return hi;
    }
    let lo = v[..n / 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (lo + hi) / 2.0
}

fn nth(v: &mut [f64], rank: usize) -> f64 {
    let (_, value, _) = v.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
    *value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_matches_scipy_reflect() {
        // scipy.ndimage.uniform_filter1d([1, 2, 3, 4, 5], 3) and size 4
        assert_eq!(mean_1d(&[1.0, 2.0, 3.0, 4.0, 5.0], 3), vec![4.0 / 3.0, 2.0, 3.0, 4.0, 14.0 / 3.0]);
        assert_eq!(mean_1d(&[1.0, 2.0, 3.0, 4.0, 5.0], 4), vec![1.5, 1.75, 2.5, 3.5, 4.25]);
    }

    #[test]
    fn median_matches_scipy_reflect() {
        // scipy.ndimage.median_filter([5, 1, 4, 2, 3], 3)
        assert_eq!(median_1d(&[5.0, 1.0, 4.0, 2.0, 3.0], 3), vec![5.0, 4.0, 2.0, 3.0, 3.0]);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    // the plain way: gather every window and pick the rank
    fn rank_2d_direct(g: &Grid<f32>, size_b: usize, size_t: usize, rank: usize) -> Grid<f32> {
        let mut out = Grid::new(g.bins, g.frames, 0.0f32);
        let (sb, st) = (size_b as isize / 2, size_t as isize / 2);
        for t in 0..g.frames {
            for b in 0..g.bins {
                let mut window = Vec::new();
                for dt in 0..size_t {
                    for db in 0..size_b {
                        let tt = reflect(t as isize + dt as isize - st, g.frames);
                        let bb = reflect(b as isize + db as isize - sb, g.bins);
                        window.push(g.at(bb, tt) as f64);
                    }
                }
                out.set(b, t, nth(&mut window, rank) as f32);
            }
        }
        out
    }

    #[test]
    fn sliding_rank_matches_direct() {
        let mut rng = crate::rng::Rng::new(5);
        let mut g = Grid::new(37, 11, 0.0f32);
        for v in g.data.iter_mut() {
            // few distinct values, so ties are common; both infinities as well
            *v = (rng.next_f32() * 6.0).floor();
        }
        g.data[40] = f32::INFINITY;
        g.data[41] = f32::NEG_INFINITY;

        // windows wider than the grid take hours under Miri; cargo test runs them
        let mut shapes = vec![(9, 5, 22), (9, 5, 33), (3, 1, 0)];
        let mut sizes = vec![1, 2, 5, 41];
        if !cfg!(miri) {
            shapes.push((45, 13, 300));
            sizes.push(861);
        }

        for (sb, st, rank) in shapes {
            let a = rank_2d(&g, sb, st, rank);
            let b = rank_2d_direct(&g, sb, st, rank);
            assert!(a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()));
        }

        let row: Vec<f64> = g.data.iter().map(|v| *v as f64).collect();
        for size in sizes {
            let direct: Vec<f64> = (0..row.len())
                .map(|i| {
                    let mut w: Vec<f64> = (0..size)
                        .map(|j| row[reflect(i as isize + j as isize - size as isize / 2, row.len())])
                        .collect();
                    nth(&mut w, size / 2)
                })
                .collect();
            let sliding = median_1d(&row, size);
            assert!(sliding.iter().zip(&direct).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    #[test]
    fn mean_time_matches_mean_1d() {
        let mut rng = crate::rng::Rng::new(9);
        let mut g = Grid::new(13, 40, 0.0f32);
        for v in g.data.iter_mut() {
            *v = rng.next_f32() * 1e3;
        }
        for size in [1, 2, 3, 86] {
            let m = mean_time(&g, size);
            for b in 0..g.bins {
                let row: Vec<f64> = g.row(b).iter().map(|v| *v as f64).collect();
                let want: Vec<f32> = mean_1d(&row, size).iter().map(|v| *v as f32).collect();
                assert!(m.row(b).iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits()));
            }
        }
    }

    #[test]
    fn percentile_rank_matches_scipy() {
        assert_eq!(percentile_rank(45, 75.0), 33);
        assert_eq!(percentile_rank(5, 100.0), 4);
    }
}
