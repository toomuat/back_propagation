use ndarray::prelude::*;
use rand::Rng;
use std::fs::File;
#[allow(unused_imports)]
use std::io::{BufRead, BufReader, Error, Write};

pub struct Network {
    w: Vec<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>>,
    in_num: usize,
    out_num: usize,
    calc: Vec<(Vec<f64>, Vec<f64>)>,
    rng: rand::rngs::ThreadRng,
}

impl Network {
    pub fn new(in_num: usize, out_num: usize) -> Self {
        Network {
            w: vec![],
            in_num: in_num,
            out_num: out_num,
            calc: vec![],
            rng: rand::thread_rng(),
        }
    }

    pub fn add_hidden_layer(mut self, node_num: usize) -> Self {
        if node_num != self.w[1].len() {
            panic!("invalid node num");
        }

        if self.w.len() == 0 {
            let w1 = Array::from_shape_vec((node_num, self.in_num),
                (0..(node_num * self.in_num))
                    .map(|_| self.rng.gen_range(-5.0..5.0))
                    .collect::<Vec<f64>>())
                    .unwrap();
            let w2 = Array::from_shape_vec((self.out_num, node_num),
                (0..(self.out_num * node_num))
                    .map(|_| self.rng.gen_range(-5.0..5.0))
                    .collect::<Vec<f64>>())
                    .unwrap();

            self.w.push(w1);
            self.w.push(w2);
        } else {
            let w = Array::from_shape_vec((node_num, node_num),
                (0..(node_num * node_num))
                    .map(|_| self.rng.gen_range(-5.0..5.0))
                    .collect::<Vec<f64>>())
                    .unwrap();

            self.w.insert(1, w);
        }

        self
    }

    pub fn forward(self, input: &Vec<f64>) -> Vec<f64> {
        let mut u;
        let mut x = Array::from_shape_vec(
        (1, input.len()), input.clone()).unwrap();

        for i in 0..self.w.len() {
            u = self.w[i].dot(&x);
            if i != self.w.len() - 1 {
                x = u.map(|i| sigmod(*i));
                // self.calc.push((x, u));
            } else {
                x = u;
            }
        }

        x.into_shape(self.out_num)
            .unwrap()
            .to_vec()
    }

    pub fn backward(&mut self, iter: usize, input: &Vec<f64>) -> Self {
        // 隠れ層→出力層
        let mut w_out = *self.w.last().unwrap();
        for i in 0..w_out.shape()[0] {
            for j in 0..w_out.shape()[1] {
                // w_out[[i, j]] = w_out[[i, j]] + 0.01 * (tk - zk) * (zk * (1 - zk)) * yi;
            }
        }

        // 入力層→隠れ層

        for i in 0..iter {
            for j in 0..input.len() {
                unimplemented!()
            }
        }

        self
    }
}

fn main() -> std::io::Result<()> {
    let mut network = Network::new(3, 1)
        .add_hidden_layer(10)
        .add_hidden_layer(10);

    let mut rng = rand::thread_rng();

    // 乱数で初期の重みを設定
    let mat_w_mid_to_out = Array::from_shape_vec((1, 10),
        (0..10)
            .map(|_| rng.gen_range(-5.0..5.0))
            .collect::<Vec<f64>>())
            .unwrap();
    let mat_w_in_to_mid = Array::from_shape_vec((10, 3),
        (0..30)
            .map(|_| rng.gen_range(-5.0..5.0))
            .collect::<Vec<f64>>())
            .unwrap();

    // 訓練用データを生成してファイルに格納する
    let filename = "data.txt";
    let mut file = File::create(filename)?;
    let data_num = 1000;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut train: Vec<f64> = Vec::new();

    for _ in 0..data_num {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let z = func(x1, x2);

        data.push(vec![x1, x2]);
        train.push(z);

        write!(file, "{}\n", format!("{} {} {}", x1, x2, z))?;
    }

    let eta = 0.1; // 学習率
    let times = 1000;

    let filename = "loss.txt";
    let mut file = File::create(filename)?;

    // 教師データ
    let train_mat = Array::from_shape_vec((1, data_num), train).unwrap(); // [1, N]
    // 訓練データ
    let input: Array2<f64> = data // [1000, 3]
        .iter()
        .map(|v| [v[0], v[1], 1.0])
        .collect::<Vec<_>>()
        .into();
    let mut m1 = mat_w_in_to_mid; // [10, 3]
    let mut m2 = mat_w_mid_to_out; // [1, 10]

    let mut _flag = true;

    for t in 0..times {
        let mut j: f64;
        let mut j_sum: f64 = 0.0;

        for i in 0..data_num {
            let x: ArrayBase<ndarray::OwnedRepr<f64>, _> = input
                .slice(s![i, ..])
                .to_owned();
            let u: ArrayBase<ndarray::OwnedRepr<f64>, _> = m1.dot(&x); // [10, 3] * [3, 1]
            let y = u.map(|i| sigmod(*i)); // [10, 1]
            let v = m2.dot(&y); // [1, 10] * [10, 1]
            let z = v; // [1, 1]

            j = train_mat[[0, i]] - z[0];
            j_sum += j.powf(2f64) / 2f64;

            // if t == 0 && i < 100 {
            //     let x2: ArrayBase<ndarray::OwnedRepr<f64>, _> = input
            //         .slice(s![0, ..])
            //         .to_owned();
            //     let u2: ArrayBase<ndarray::OwnedRepr<f64>, _> = m1.dot(&x2); // [10, 3] * [3, 1]
            //     let y2 = u2.map(|i| sigmod(*i)); // [10, 1]
            //     let v2: Array1<f64> = m2.dot(&y2); // [1, 10] * [10, 1]
            //     let z2 = v2;
            //     let j2  = train_mat[[0, 0]] - z2[0];

            //     write!(file, "{}\n", format!("{} {}", i, j2.powf(2f64) / 2f64))?;
            // }

            // 中間→出力層結合w_kjの更新
            // m2: [1, 10]
            for l in 0..m2.len() {
                m2[[0, l]] = m2[[0, l]] + eta * j * 1.0 * y[l];
            }

            // 入力→中間層結合w_jiの更新
            // m1: [10, 3]
            for l in 0..x.len() { // 0..3
                for m in 0..m1.shape()[0] { // 0..10
                    let x = *input.get((i, l)).unwrap();
                    let diff = eta * m2[[0, m]] * j * 1.0 * sigmod_diff(u[m]) * x;

                    m1[[m, l]] = m1[[m, l]] + diff;
                    // m1[[l, m]] = m1[[l, m]] + diff;
                }
            }
        }
        if t < 500 {
            write!(file, "{}\n", format!("{} {}", t, j_sum))?;
        }
        _flag = false;
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
    // (x1.powf(2.0) + x2.powf(2.0)).sqrt()
    8.0 * (-x1.powf(2.0) - x2.powf(2.0)).exp() * (0.1 + x1 * (x2 - 0.5))

    /*
    splot [-2:2] [-2:2] sqrt(x**2 + y**2)
    splot [-2:2] [-2:2] 8 * exp(-x**2 - y**2) * (0.1 + x*(y - 0.5))
    */
}

fn sigmod(u: f64) -> f64 {
    1. / (1. + (-u).exp())
    // 1. / (1. + f64::exp(-u))
}

fn sigmod_diff(u: f64) -> f64 {
    sigmod(u) * (1. - sigmod(u))
}
