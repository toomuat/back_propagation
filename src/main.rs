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

    let filename = "data.txt";
    let mut file = File::create(filename)?;

    let mut x1;
    let mut x2;
    let mut z;
    for _ in 0..10 {
        x1 = rng.gen_range(-2.0..2.0);
        x2 = rng.gen_range(-2.0..2.0);
        z = func(x1, x2);
        // println!("{}", z);

        write!(file, "{}\n", format!("{} {} {}", x1, x2, z))?;
    }

    println!("{:?}", file.metadata()?);
    
    let input = File::open(filename)?;
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        //println!("{}", line?);
        //println!("{}", line.unwrap());
        //println!("{}", line.unwrap().to_string());

        //split = line?;
        //split = line.unwrap();
        //print_type_of(&split); // alloc::string::String
        //Ok(())

        //split = line?.split_whitespace();
        //split = line.unwrap().split_whitespace();
        //split = line.unwrap().to_string().split_whitespace();
        
        let line_string = line.unwrap().to_string();
        //println!("{}", line_string);
        let split = line_string.split_whitespace();
        let vec_line = split.collect::<Vec<&str>>();
        println!("{:?}", vec_line);
        //println!("{:?}", split.collect::<Vec<&str>>());
    }

    Ok(())
}

fn func(x1: f64, x2: f64) -> f64 {
    (x1.powf(2.0) + x2.powf(2.0)).sqrt()
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

