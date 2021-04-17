build:
	docker build -t jupyter .
run:
	docker run -it -v ${PWD}:/home/jovyan -p 8090:8090 jupyter jupyter notebook --port 8090