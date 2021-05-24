use ndarray::prelude::*;
use rand::Rng;
use std::fs::File;
#[allow(unused_imports)]
use std::io::{BufRead, BufReader, Error, Write};

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();

    // 乱数で初期の重みを設定
    let w_mid_to_out: Vec<f64> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let w_in_to_mid: Vec<f64> = (0..9).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let mut mat_w_mid_to_out = Array::from_shape_vec((1, 3), w_mid_to_out).unwrap();
    let mut mat_w_in_to_mid = Array::from_shape_vec((3, 3), w_in_to_mid).unwrap();

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

        data.push(vec_f64); // x1, x2 訓練データ
        train.push(vec_line[2].parse::<f64>().unwrap()); // y 教師データ
    }

    // 教師データ
    let train_mat = Array::from_shape_vec((1, data_num), train).unwrap(); // 1xN

    // 訓練データ
    let data_arr2: Array2<f64> = data.iter().map(|v| [v[0], v[1], 1.0]).collect::<Vec<_>>().into(); // [1000, 2]
    let data_rev = data_arr2.reversed_axes(); // [3, 1000]

    let eta = 0.05; // 学習率

    let times = 100;

    /*
    w_kj(t + 1) = w_kj(t) - η ∂J(w)/∂w_kj

    ∂J(w)/∂w_kj = ∂J/∂z_k・∂z_k/∂v_k・∂v_k/∂w_k j
    ∂J/∂z_k = -(t_k - z_k)
    ∂z_k/∂v_k = f'(v_k)
    */

    let filename = "loss.txt";
    let mut file2 = File::create(filename)?;

    for i in 0..times {
        // 順伝播（forward-propagation）
        // println!("{}", mat_w_in_to_mid);
        let u = mat_w_in_to_mid.dot(&data_rev); // (3, 3) * (3, 1000)
        let y = u.map(|i| sigmod(*i)); // (3, 1000)
        let v = mat_w_mid_to_out.dot(&y); // (1, 3) * (3, 1000)
        // let z = v.map(|i| sigmod(*i)); // (1, 1000)
        let z = &v; // (1, 1000)

        let data_rev2 = data_rev
            .to_owned()
            .reversed_axes(); // (n, 2)

        let j = &train_mat - z; // ∂J/∂z (1, 1000)

        // イテレーションごとのロスを合計してファイルに書き込む
        let loss = j
            .to_owned()
            .reversed_axes();
        write!(file2, "{}\n", format!("{} {}", i,
            loss.map(|x| *x).
            fold(0 as f64, |acc, i| acc + (*i).powf(2f64)) / 2f64))?;
        // println!("{:?}", loss.shape()); // [n, 1]

        // 誤差逆伝播法（BackPropagation）
        // 中間→出力層結合w_kjの更新
        mat_w_mid_to_out = mat_w_mid_to_out +
            eta * (&j)
            .dot(&y.reversed_axes()); // (1, n) * (n, 3)

        // mat_w_mid_to_out = // (1, 3)
        //     mat_w_mid_to_out
        //     + eta * (&j * v.map(|i| sigmod_diff(*i)))
        //     .dot(&y.reversed_axes()); // (1, n) * (n, 3)

        let mat_w_mid_to_out2 = mat_w_mid_to_out
            .to_owned()
            .reversed_axes(); // (3, 1)

        // 入力→中間層結合w_jiの更新
        mat_w_in_to_mid = mat_w_in_to_mid // (3, 3)
            + eta *
                (&mat_w_mid_to_out2 // (3, 1)
                .dot(&j) // (1, 1000)
                * u.map(|i| sigmod_diff(*i))) // (3, 1000)
                .dot(&data_rev2); // (1000, 3)

        // mat_w_in_to_mid = mat_w_in_to_mid // (3, 2)
        //     + eta
        //         * (&mat_w_mid_to_out2.dot( // (3, 1)
        //             &(&j * v.map(|i| sigmod_diff(*i))) // (1, n)
        //         )
        //         * u.map(|i| sigmod_diff(*i)))  // (3, n)
        //         .dot(&data_rev2); // (n, 2)
    }

    let filename2 = "result.txt";
    let mut file = File::create(filename2)?;

    // 学習した重みをもとにデータを出力
    for _ in 0..data_num {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let x = Array::from_shape_vec((3, 1), vec![x1, x2, 1.0]).unwrap();

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
    // 1. / (1. + f64::exp(-u))
}

fn sigmod_diff(u: f64) -> f64 {
    sigmod(u) * (1. - sigmod(u))
}
