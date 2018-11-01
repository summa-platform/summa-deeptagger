#!/bin/sh

cd `dirname $0`

# --name deeptagger

if [ "$DATA" = "" ]; then
	if [ -d "data" ]; then
		DATA="$PWD/data"
	else
		DATA="$PWD"
	fi
fi

# nvidia-docker run $@ -v $PWD:/deeptagger -v $DATA:/opt/app/model -p 6000:6000 -it deeptagger /deeptagger/deeptagger.py --rest --data-dir /opt/app/model
echo "Host data directory: $DATA"

if [ "`which nvidia-docker`" != "" ]; then
	echo "Using nvidia-docker"
	nvidia-docker run $@ -v $PWD:/deeptagger -v $DATA:/opt/app/model -p 6000:6000 -it deeptagger /deeptagger/run_rest.sh /opt/app/model
else
	docker run $@ -v $PWD:/deeptagger -v $DATA:/opt/app/model -p 6000:6000 -it deeptagger /deeptagger/run_rest.sh /opt/app/model
fi
