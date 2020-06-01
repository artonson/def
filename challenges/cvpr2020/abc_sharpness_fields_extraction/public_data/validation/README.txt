To download the files:

1. Create the two folders "npy" and "txt":

    mkdir -p npy txt

2. The validation chunk can then be downloaded for example with wget or curl.
The following command downloads all files with 4 requests (maximum) in parallel into
the folders "npy" and "txt":

    cat val_file_urls.txt | xargs -n 2 -P 4 sh -c 'wget --no-check-certificate $0 -O $1'

