@echo off

setlocal
echo Looking for proto files..

for %%F in (object_detection\protos\*.proto) do (
	echo %%F
	protoc %%F --python_out=.
)