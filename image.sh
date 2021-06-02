#!/bin/bash


gnuplot << EOF
    set terminal png
    set output "img/data.png"
    splot "data.txt" with points pt 7 ps 0.5
EOF

gnuplot << EOF
    set terminal png
    set output "img/result.png"
    splot "result.txt" with points pt 7 ps 0.5
EOF

gnuplot << EOF
    set terminal png
    set output "img/loss.png"
    plot "loss.txt" with lines
EOF

