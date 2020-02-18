all: example1 example2

figures: 
	make -C figures/


setup: /usr/local/lib/python3.6/dist-packages/ablr-0.0.0-py3.6.egg

/usr/local/lib/python3.6/dist-packages/ablr-0.0.0-py3.6.egg:
	#https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
	sudo python3 setup.py install
