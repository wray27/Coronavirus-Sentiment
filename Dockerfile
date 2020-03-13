# ubuntu base
FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev

# create an src working directory and copy the entire directory over to it—data, notebook, and all. Once it is started, the container will have an exact copy of what you have locally.
RUN mkdir src
WORKDIR src/
COPY . .
COPY ./data src/data
COPY ./notebooks src/notebooks

# installing aall of the python requirements
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

# install all the rewuiremnets necessary for the python application
RUN pip3 install -r requirements.txt

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# command thats run when container is started
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]