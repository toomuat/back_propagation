set term gif animate delay 1
set output "img/data2.gif"
do for [i = 1:360] {
    set view 60,i,1,1
    splot "data.txt" with points pt 7 ps 0.5
}
set output

set term gif animate delay 1
set output "img/result2.gif"
do for [i = 1:360] {
    set view 60,i,1,1
    splot "result.txt" with points pt 7 ps 0.5
}
set output
