use ndarray::prelude::*;
use rand::Rng;
use std::fs::File;
#[allow(unused_imports)]
use std::io::{BufRead, BufReader, Error, Write};

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();

    // 乱数で初期の重みを設定
    let w_mid_to_out: Vec<f64> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let w_in_to_mid: Vec<f64> = (0..6).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let mut mat_w_mid_to_out = Array::from_shape_vec((1, 3), w_mid_to_out).unwrap();
    let mut mat_w_in_to_mid = Array::from_shape_vec((3, 2), w_in_to_mid).unwrap();

    // 訓練用データを生成してファイルに格納する
    // sqrt(x1^2 + x2^2)
    let filename = "data.txt";
    let mut file = File::create(filename)?;
    let data_num = 1000;

    for _ in 0..data_num {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let z = func(x1, x2);

        write!(file, "{}\n", format!("{} {} {}", x1, x2, z))?;
    }

    let input = File::open(filename)?;
    let buffered = BufReader::new(input);
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut train: Vec<f64> = Vec::new();

    // 訓練データと正解データ
    for line in buffered.lines() {
        let line_string: String = line.unwrap().to_string();
        let split = line_string.split_whitespace();
        let vec_line: Vec<&str> = split.collect::<Vec<&str>>();
        let vec_f64: Vec<f64> = vec_line
            .iter()
            .enumerate()
            .filter(|&(i, _v)| i != 2 as usize)
            .map(|(_i, v)| v.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();

        data.push(vec_f64);
        train.push(vec_line[2].parse::<f64>().unwrap());
    }

    let train_mat = Array::from_shape_vec((1, data_num), train).unwrap();
    let data_arr2: Array2<f64> = data.iter().map(|v| [v[0], v[1]]).collect::<Vec<_>>().into();
    let data_rev = data_arr2.reversed_axes();

    let eta = 0.1;

    let times = 1000;

    for _ in 0..times {
        // println!("{}", mat_w_in_to_mid);
        let u = mat_w_in_to_mid.dot(&data_rev); // (3, 2) * (2, n)
        // println!("{}", u);
        let y = u.map(|i| sigmod(*i)); // (3, n)
        // println!("{}", y);
        let v = mat_w_mid_to_out.dot(&y); // (1, 3) * (3, n)
        let z = v.map(|i| sigmod(*i)); // (1, n)

        let data_rev2 = data_rev.to_owned().reversed_axes();

        let j = &train_mat - z; // ∂J/∂z (1, n)

        mat_w_mid_to_out = // (1, 3)
            mat_w_mid_to_out
            + eta * (&j * v.map(|i| sigmod_diff(*i)))
            .dot(&y.reversed_axes()); // (1, n) * (n, 3)

        let mat_w_mid_to_out2 = mat_w_mid_to_out.to_owned().reversed_axes();

        mat_w_in_to_mid = mat_w_in_to_mid // (3, 2)
            + eta
                * (&mat_w_mid_to_out2.dot( // (3, 1)
                    &(&j * v.map(|i| sigmod_diff(*i))) // (1, n)
                )
                * u.map(|i| sigmod_diff(*i)))  // (3, n)
                .dot(&data_rev2); // (n, 2)
    }

    let filename2 = "result.txt";
    let mut file = File::create(filename2)?;

    for _ in 0..data_num {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let x = Array::from_shape_vec((2, 1), vec![x1, x2]).unwrap();

        let u = mat_w_in_to_mid.dot(&x); // (3, 2) * (2, 1)
        let y = u.map(|i| sigmod(*i)); // (3, 1)
        let v = mat_w_mid_to_out.dot(&y); // (1, 3) * (3, 1)
        let z = v.map(|i| sigmod(*i)); // (1, 1)

        write!(file, "{}\n", format!("{} {} {}", x1, x2, z[[0, 0]]))?;
    }

    Ok(())
}

fn func(x1: f64, x2: f64) -> f64 {
    (x1.powf(2.0) + x2.powf(2.0)).sqrt()
}

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn sigmod(u: f64) -> f64 {
    1. / (1. + (-u).exp())
}

fn sigmod_diff(u: f64) -> f64 {
    sigmod(u) * (1. - sigmod(u))
}
