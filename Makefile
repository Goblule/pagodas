
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y pagodas || :
	@pip install -e .
