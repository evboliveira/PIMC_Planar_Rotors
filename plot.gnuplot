reset
set encoding utf8
set nokey
set terminal pngcairo
date = system("date +%F.%H.%M.%S")
set output 'results/plots/graph '.date.'.png'
# set terminal latex
# set xrange [0:1]
# set yrange [0:1]
set grid
set title "MC steps = 1000, N = 3, m_{max} = 5, g = 1.0, T = 0.1" 
set xlabel "tau"
set ylabel "K"
set size square 
# set term post eps enhanced color
# set out 'graf.eps'
# f(x)=-0.52929 # for N=2
# f(x)=-1.17123 # for N=3
f(x)=0.947591 # for N=3
plot f(x) w l lw 2, "results/main.txt" w lp lw 2
replot