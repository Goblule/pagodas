
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y pagodas || :
	@pip install -e .

run_preprocess:
	python -c 'from interface.main import preprocess; preprocess()'

reset_local_files:
	rm -rf ./preproc_data
	mkdir -p preproc_data
