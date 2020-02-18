all: example1.pdf example2.pdf mixedModel.pdf

example1.pdf:
	python3 example1.py

example2.pdf:
	python3 example2.py

mixedModel.pdf:
	python3 mixedModel.py

png:
	sudo mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.off
	convert -density 300 example1.pdf png/example1-%d.png
	convert -density 300 example2.pdf png/example2-%d.png
	convert -density 300 mixedModel.pdf png/mixedModel-%d.png
	sudo mv /etc/ImageMagick-6/policy.xml.off /etc/ImageMagick-6/policy.xml


