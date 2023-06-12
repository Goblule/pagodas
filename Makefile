default :
	@echo ' '
	@echo 'To install pagodas package, type:'
	@echo 'make reinstall_package'
	@echo ' '
	@echo 'Package options'
	@echo '  reset_local_files  clean the preproc_data storage'
	@echo '  run_preprocess     run preprocessing on raw data'
	@echo '  run_predict        run prediction on new data'
	@echo '  <option>           description'
	@echo ' '

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y pagodas || :
	@pip install -e .

reset_local_files:
	rm -rf ./preproc_data
	mkdir -p preproc_data

run_preprocess:
	python -c 'from interface.main import preprocess; preprocess()'

run_predict:
	python -c 'from interface.main import predict; predict()'
