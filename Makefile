
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
GRAPH_FILENAME = graph

.PHONY: all
all: prereqs graphnet

.PHONY: graphnet
graphnet: 
	@echo "\nmaking graphnet"
	(cd pretrained_models/caffe/SSD_MobileNet; make compile; cd ../../../; cp pretrained_models/caffe/SSD_MobileNet/graph ./${GRAPH_FILENAME};) 

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py

.PHONY: run_py
run_py: prereqs graphnet
	@echo "\nmaking run_py"
	python3 ./vehicle_platooning.py


.PHONY: run
run: run_py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make graph - makes and copies the compiled graph file";
	@echo "  make run_py - runs the python application";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}


