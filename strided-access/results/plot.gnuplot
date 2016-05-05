set terminal postscript enhanced color eps

set style data lines
set style line 1  linetype -1 linewidth 3 lc rgb "#AA0000"
set style line 2  linetype -1 linewidth 3 lc rgb "#0000AA"
set style line 3  linetype -1 linewidth 3 lc rgb "#000000"
set style line 4  linetype -1 linewidth 3 lc rgb "#00AA00"
set style line 5  linetype  2 linewidth 3 lc rgb "#00AA00"
set style line 6  linetype -1 linewidth 3 lc rgb "#00AA00"
set style line 7  linetype  2 linewidth 3 lc rgb "#000000"
set style line 8  linetype -1 linewidth 3 lc rgb "#000000"
set style increment user

set size 0.75,0.75
#set size ratio 0.66
set border lw 2

set key top right Right
set grid
set logscale y
set xrange [1:16]

#######

set output "strided-access.eps"
set title "Memory Bandwidth for Strided Array Access\n{/*0.7 x[i*stride] = y[i*stride] + z[i*stride]}"
set ylabel "Memory Bandwidth (GB/sec)"
set xlabel "Stride (4 Bytes per Element)"
plot 'w9100.txt'           using 1:3 with linesp ls 1 pt  5 ps 1.5 title "AMD FirePro W9100", \
     'xeon-e5-2670v3.txt'  using 1:3 with linesp ls 2 pt  7 ps 1.5 title "1x INTEL Xeon E5-2670v3", \
     'xeon-phi-7120.txt'   using 1:3 with linesp ls 3 pt  9 ps 2 title "INTEL Xeon Phi 7120", \
     'k20m.txt'            using 1:3 with linesp ls 4 pt 11 ps 2   title "NVIDIA Tesla K20m"
     


