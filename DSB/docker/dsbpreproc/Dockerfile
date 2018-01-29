FROM continuumio/anaconda

WORKDIR /dsb3

RUN mkdir -p /dsb3/preproc

RUN conda install opencv
RUN pip install pydicom
RUN conda install scipy
RUN conda install pandas

ADD helpers.py .
ADD settings.py .
ADD helpers_dicom.py .
ADD step0_preprocess.py .

ADD preprocess.sh .

# the following only gets executed during docker run
CMD [ "/bin/sh", "preprocess.sh"]

VOLUME /dsb3/preproc
