# Coronavirus-Sentiment

## Docker Container

Docker container is set up to build the container image use:

`$ docker build -t wray27/coronavirus . `

To run the container image (output given below):

```
$ docker run --rm -it -v "$PWD:/src" -p 8889:8888 --name ads wray27/coronavirus

To access the notebook, open this file in a browser:
file:///home/jovyan/.local/share/jupyter/runtime/nbserver-6-open.html
Or copy and paste one of these URLs:
http://01499ba7ede3:8888/?token=d42eb98a853087b8c1f1906e7f11dad3fab671a87107c972
or http://127.0.0.1:8888/?token=d42eb98a853087b8c1f1906e7f11dad3fab671a87107c972
```

The container will run jupyter notebook to access it on your machine, type  **localhost:8889/** in to your browser to open it up.

The jupyter notebook requires a token by default enter in the token in the final link given in the output.

