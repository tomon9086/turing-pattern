use image::{GrayImage, Luma};
use minifb::{Key, Window, WindowOptions};
use ndarray::{s, Array2, ArrayView2, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

const SIZE: usize = 256;
const DIFF_A: f64 = 0.16;
const DIFF_B: f64 = 0.04;
const FEED: f64 = 0.070;
const KILL: f64 = 0.062;

fn laplacian(arr: ArrayView2<f64>) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((SIZE, SIZE));
    Zip::from(result.slice_mut(s![1..-1, 1..-1]))
        .and(arr.slice(s![1..-1, 1..-1]))
        .and(arr.slice(s![0..-2, 1..-1]))
        .and(arr.slice(s![2.., 1..-1]))
        .and(arr.slice(s![1..-1, 0..-2]))
        .and(arr.slice(s![1..-1, 2..]))
        .for_each(|r, &center, &up, &down, &left, &right| {
            *r = up + down + left + right - 4.0 * center;
        });

    result
}

fn update(conc_a: &mut Array2<f64>, conc_b: &mut Array2<f64>) {
    let lap_a = laplacian(conc_a.view());
    let lap_b = laplacian(conc_b.view());

    Zip::from(conc_a)
        .and(conc_b)
        .and(&lap_a)
        .and(&lap_b)
        .for_each(|a, b, &lap_a, &lap_b| {
            let reaction = *a * *b * *b;
            *a += DIFF_A * lap_a - reaction + FEED * (1.0 - *a);
            *b += DIFF_B * lap_b + reaction - (KILL + FEED) * *b;
        });
}

fn main() {
    let mut conc_a = Array2::random((SIZE, SIZE), Uniform::new(0.5, 1.0));
    let mut conc_b = Array2::random((SIZE, SIZE), Uniform::new(0.0, 0.5));

    let mut window = Window::new("Turing Pattern", SIZE, SIZE, WindowOptions::default())
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

    while window.is_open() && !window.is_key_down(Key::Escape) {
        update(&mut conc_a, &mut conc_b);

        let mut img = GrayImage::new(SIZE as u32, SIZE as u32);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let value = (conc_b[[x as usize, y as usize]] * 255.0) as u8;
            *pixel = Luma([value]);
        }

        let buffer: Vec<u32> = img
            .enumerate_pixels()
            .map(|(_, _, pixel)| {
                let value = pixel.0[0] as u32;
                (value << 16) | (value << 8) | value
            })
            .collect();

        window.update_with_buffer(&buffer, SIZE, SIZE).unwrap();
    }
}
