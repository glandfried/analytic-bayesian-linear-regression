all: submodule main.pdf clean

submodule: tex/.git biblio/.git

tex/.git:
	git submodule update --init tex/
biblio/.git:
	git submodule update --init biblio/

main.pdf:
	pdflatex main.tex

clean: 
	- rm -f *.log
	- rm -f *.soc
	- rm -f *.toc
	- rm -f *.aux
	- rm -f *.out
	- rm -f main.idx
	- rm -f *.bbl
	- rm -f *.bbg
	- rm -f *.dvi
	- rm -f *.blg
	- rm -f *.lof
	- rm -f *.nav
	- rm -f *.snm
	- rm -f *~

