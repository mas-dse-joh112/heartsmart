FROM alpine

RUN mkdir -p /dsb3/data

WORKDIR /dsb3/data

ADD data_kaggle/train ./data_kaggle/train/
ADD data_kaggle/test ./data_kaggle/test/
ADD data_kaggle/validate ./data_kaggle/validate/
ADD train_validate.csv .

VOLUME /dsb3/data

# just for displaying during building
RUN ls -lR /dsb3/data && pwd

# just for displaying during runtime
CMD ["ls", "-l", "./data_kaggle/test"]
