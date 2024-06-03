use image::{GrayImage, Luma};
use nannou::color::rgb;
use nannou::{color::*, event::Update, App, Frame};
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

fn main() {
    nannou::app(model).update(update).simple_window(view).run();
}

struct Model {
    conc_a: Array2<f64>,
    conc_b: Array2<f64>,
}

fn model(_app: &App) -> Model {
    Model {
        conc_a: Array2::random((SIZE, SIZE), Uniform::new(0.5, 1.0)),
        conc_b: Array2::random((SIZE, SIZE), Uniform::new(0.0, 0.5)),
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    let conc_a = &mut model.conc_a;
    let conc_b = &mut model.conc_b;

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

fn view(app: &App, model: &Model, frame: Frame) {
    let window_rect = app.window_rect();
    let draw = app.draw();

    let cell_w = window_rect.w() / SIZE as f32;
    let cell_h = window_rect.h() / SIZE as f32;

    // let conc_a = &model.conc_a;
    let conc_b = &model.conc_b;

    let mut img = GrayImage::new(SIZE as u32, SIZE as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let value = (conc_b[[x as usize, y as usize]] * 255.0) as u8;
        *pixel = Luma([value]);

        draw.rect()
            .w_h(cell_w, cell_h)
            .color(rgb(value, value, value))
            .x_y(x as f32 * cell_w, y as f32 * cell_h);
    }

    draw.text(&app.elapsed_frames().to_string())
        .font_size(24)
        .x(window_rect.left() + 50.0)
        .y(window_rect.top() - 10.0)
        .color(WHITE);
    draw.background().color(BLACK);
    draw.to_frame(app, &frame).unwrap();
}
