#!/bin/bash

cd $(dirname $0)
mkdir PGF2SVGRUN
cp standalone.tex PGF2SVGRUN
cp $1.pgf PGF2SVGRUN
cd PGF2SVGRUN
eval sed 's/FILENAME/$1.pgf/1' standalone.tex > job.tex
pdflatex job.tex --shell-escape
#pdfcrop --margins="0 5 0 5" job.pdf ../$1.pdf
inkscape job.pdf -o ../$1.svg -D
inkscape job.pdf -o ../$1.png -D
cd ..
rm -r PGF2SVGRUN
