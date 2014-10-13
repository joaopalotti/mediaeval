
file=main

all: compile run
	
compile:
	#latex ${file}.tex
	#bibtex ${file}
	#latex ${file}.tex
	#latex ${file}.tex
	#dvipdf ${file}.dvi
	pdflatex ${file}.tex
	rm -f ${file}.dvi ${file}.log ${file}.aux

run:
	evince ${file}.pdf &

clean:
	rm -f ${file}.dvi ${file}.log ${file}.aux ${file}.bbl *.log


