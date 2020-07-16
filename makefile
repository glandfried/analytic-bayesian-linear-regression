all: figures doc

figures: 
	make -C figures/

doc:
	make -C doc/
	
#README.md:
#	pandoc -s doc/bayesianLinearRegression.tex -o bayesianLinearRegression.md
#	sed -i 's/..\/figures/figures/g' bayesianLinearRegression.md
#	sed -i 's/pdf/png/g' bayesianLinearRegression.md
#	sed -i 's/{width="\\textwidth"}//g' bayesianLinearRegression.md
#	sed -i 's/{width="0\.[0-9][0-9]cm"}/                /g' bayesianLinearRegression.md
#	sed -i 's/\[0\.[0-9][0-9]\]{}/ /g' bayesianLinearRegression.md


setup: /usr/local/lib/python3.6/dist-packages/ablr-0.0.0-py3.6.egg

/usr/local/lib/python3.6/dist-packages/ablr-0.0.0-py3.6.egg:
	#https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
	sudo python3 setup.py install

mirrors:
	git remote set-url --add origin git@github.com:BayesDeLasProvinciasUnidasDelSur/analytic-linear-regression.git
	git remote set-url --add origin git@git.exactas.uba.ar:bayes/analytic-linear-regression.git
	



