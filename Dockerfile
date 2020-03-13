# ubuntu base
FROM jupyter/minimal-notebook

MAINTAINER Aaron Wray <aw16997@bristol.ac.uk>

# create an src working directory
COPY requirements.txt /src/requirements.txt
WORKDIR /src

# installing aall of the python requirements
RUN pip install -r requirements.txt


# command thats run when container is started
CMD ["jupyter", "notebook",  "--ip=0.0.0.0","--no-browser","--allow-root"]
#  ,"--NotebookApp.password=coronavirus"