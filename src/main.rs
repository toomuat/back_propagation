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
    let times = 1000;

    let filename = "loss.txt";
    let mut file2 = File::create(filename)?;

    let input: Array2<f64> = data // [1000, 3]
        .iter()
        .map(|v| [v[0], v[1], 1.0])
        .collect::<Vec<_>>()
        .into();
    let mut m1 = mat_w_in_to_mid; // [3, 3]
    let mut m2 = mat_w_mid_to_out; // [3, 1]

    // println!("{}", input.get((0, 0)).unwrap());

    let mut flag = true;

    for t in 0..times {
        let mut j: f64 = 0.0;
        let mut j_sum: f64 = 0.0;

        for i in 0..data_num {
            let x: ArrayBase<ndarray::OwnedRepr<f64>, _> = input
                .slice(s![i, ..])
                .to_owned();
            let u: ArrayBase<ndarray::OwnedRepr<f64>, _> = m1.dot(&x); // [3, 3] * [3, 1]
            let y = u.map(|i| sigmod(*i)); // [3, 1]
            let v = m2.dot(&y); // [1, 3] * [3, 1]
            let z = &v; // [1, 1]

            // println!("{:?}", *u.get((0, 0)).unwrap());
            // println!("{:?}", u[0]);
            // println!("{:?}", u.shape());
            // loop{}

            j = train_mat[[0, i]] - z[0];
            j_sum += j.powf(2f64) / 2f64;
            if flag {
                write!(file2, "{}\n", format!("{} {}", t*1000 + i, j.powf(2f64) / 2f64))?;
            }

            // 中間→出力層結合w_kjの更新
            // m2: [1, 3]
            for l in 0..y.len() {
                m2[[0, l]] = m2[[0, l]] + eta * j * 1.0 * y[l];
            }

            // 入力→中間層結合w_jiの更新
            // m1: [3, 3]
            for l in 0..3 {
                for m in 0..3 {
                    let x = *input.get((i, m)).unwrap();
                    let diff = eta * m2[[0, l]] * j * 1.0 * sigmod_diff(u[l]) * x;

                    // let diff = eta *
                    //     (&m2 * j * 1.0) // [1, 3]
                    //     .dot(&u.map(|i| sigmod_diff(*i))) * // [3, 1]
                    //     (*input.get((i, m)).unwrap());

                    m1[[l, m]] = m1[[l, m]] + diff;
                }
            }
        }
        // write!(file2, "{}\n", format!("{} {}", t, j_sum))?;
        flag = false;
    }

    let filename2 = "result.txt";
    let mut file = File::create(filename2)?;

    // 学習した重みをもとにデータを出力
    for _ in 0..data_num {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let x = Array::from_shape_vec((3, 1), vec![x1, x2, 1.0]).unwrap();

        let u = m1.dot(&x); // (3, 3) * (3, 1)
        let y = u.map(|i| sigmod(*i)); // (3, 1)
        let v = m2.dot(&y); // (1, 3) * (3, 1)
        let z = v;

        write!(file, "{}\n", format!("{} {} {}", x1, x2, z[[0, 0]]))?;
    }

    Ok(())
}

fn func(x1: f64, x2: f64) -> f64 {
    (x1.powf(2.0) + x2.powf(2.0)).sqrt()
}

fn sigmod(u: f64) -> f64 {
    1. / (1. + (-u).exp())
    // 1. / (1. + f64::exp(-u))
}

fn sigmod_diff(u: f64) -> f64 {
    sigmod(u) * (1. - sigmod(u))
}
