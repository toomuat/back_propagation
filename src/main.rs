use ndarray::prelude::*;
use rand::Rng;
use std::fs::File;
use std::io::{Write, BufReader, BufRead, Error};

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();

    //let values: Vec<f64> = (0..6).map(|_| rng.gen_range(-5.0..5.0)).collect();
    //println!("{:?}", values);
    
    //let matrix = Array::from_shape_vec((2, 3), values).unwrap();
    //println!("{:?}", matrix);
    // println!("{}", matrix);
    
    let w_mid_to_out: Vec<f64> = (0..3).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let w_in_to_mid: Vec<f64> = (0..6).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let mat_w_mid_to_out = Array::from_shape_vec((1, 3), w_mid_to_out).unwrap();
    let mat_w_in_to_mid = Array::from_shape_vec((3, 2), w_in_to_mid).unwrap();

    let filename = "data.txt";
    let mut file = File::create(filename)?;

    for _ in 0..10 {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let z = func(x1, x2);

        write!(file, "{}\n", format!("{} {} {}", x1, x2, z))?;
    }

    //println!("{:?}", file.metadata()?);
    
    let input = File::open(filename)?;
    let buffered = BufReader::new(input);
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut train: Vec<f64> = Vec::new();

    for line in buffered.lines() {
        let line_string: String = line.unwrap().to_string();
        let split = line_string.split_whitespace();
        let vec_line: Vec<&str> = split.collect::<Vec<&str>>();
        //println!("{:?}", vec_line);
        let vec_f64: Vec<f64> = vec_line.iter()
            .enumerate()
            .filter(|&(i, _v)| i != 2 as usize)
            .map(|(_i, v)| v.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        //println!("{:?}", vec_f64);

        data.push(vec_f64);
        train.push(vec_line[2].parse::<f64>().unwrap());
    }

    //println!("{:?}", data);
    //println!("{:?}", train);

    let eta = 0.01;
    //println!("{}", sigmod(eta));
    //println!("{}", sigmod_diff(eta));

    let times = 1;

    for _ in 0..times {
    }

    Ok(())
}

fn func(x1: f64, x2: f64) -> f64 {
    (x1.powf(2.0) + x2.powf(2.0)).sqrt()
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn sigmod(u: f64) -> f64 {
    1. / (1. + u.exp())
}

fn sigmod_diff(u: f64) -> f64 {
    sigmod(u) * (1. - sigmod(u))
}
