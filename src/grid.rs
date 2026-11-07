//! A spectrogram-shaped array: `bins` rows by `frames` columns, one frame stored contiguously.

#[derive(Clone)]
pub struct Grid<T> {
    pub bins: usize,
    pub frames: usize,
    pub data: Vec<T>,
}

impl<T: Copy> Grid<T> {
    pub fn new(bins: usize, frames: usize, value: T) -> Self {
        Grid { bins, frames, data: vec![value; bins * frames] }
    }

    pub fn at(&self, bin: usize, frame: usize) -> T {
        self.data[frame * self.bins + bin]
    }

    pub fn set(&mut self, bin: usize, frame: usize, value: T) {
        self.data[frame * self.bins + bin] = value;
    }

    pub fn frame(&self, frame: usize) -> &[T] {
        &self.data[frame * self.bins..(frame + 1) * self.bins]
    }

    pub fn frame_mut(&mut self, frame: usize) -> &mut [T] {
        &mut self.data[frame * self.bins..(frame + 1) * self.bins]
    }

    /// The values of one bin over all frames.
    pub fn row(&self, bin: usize) -> Vec<T> {
        (0..self.frames).map(|t| self.at(bin, t)).collect()
    }

    pub fn set_row(&mut self, bin: usize, values: &[T]) {
        for (t, v) in values.iter().enumerate() {
            self.set(bin, t, *v);
        }
    }

    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Grid<U> {
        Grid { bins: self.bins, frames: self.frames, data: self.data.iter().map(|v| f(*v)).collect() }
    }
}
