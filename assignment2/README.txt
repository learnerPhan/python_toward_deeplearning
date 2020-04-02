######################################################################
######################################################################
###### Missing OpenSSL when installing Python3.7.x ###################


1. Install openssl
https://help.dreamhost.com/hc/en-us/articles/360001435926-Installing-OpenSSL-locally-under-your-username

2. Install python3.7.x
(https://stackoverflow.com/questions/41328451/ssl-module-in-python-is-not-available-when-installing-package-with-pip3)

sudo wget https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tar.xz
sudo tar xf Python-3.7.1.tar.xz
cd Python-3.7.1

2+1. Change and uncomment SSL setup  in Module/Setup.dist
SSL=/usr/local/openssl

3. export environment variable before running configure

export PATH=/usr/local/openssl/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openssl/lib
export LC_ALL="en_US.UTF-8"
export LDFLAGS="-L/usr/local/openssl/lib -Wl,-rpath,/usr/local/openssl/lib"

4.
sudo ./configure --with-ensurepip=yes --with-openssl=/usr/local/openssl CFLAGS="-I/usr/local/openssl/include" LDFLAGS="-L/usr/local/openssl/lib -Wl,-rpath,/usr/local/openssl/lib"


5. compiling
sudo make
6.sudo make altinstall


7. create virtual environment for tensorflow
cd assignment2
virtualenv -p python3 .tf_20_env       # Create a virtual environment (python3)
# Note: you can also use "virtualenv .env" to use your default python (please note we support 3.6)
source .env/bin/activate         # Activate the virtual environment

8.  sudo -H pip3.7 install -r requirements.txt

######################################################################
######################################################################
###### SSL: CERTIFICATE_VERIFY_FAILED ################################

https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error

$ sudo update-ca-certificates --fresh
$ export SSL_CERT_DIR=/etc/ssl/certs

######################################################################
######################################################################
###### ImportError: /lib64/libm.so.6: version `GLIBC_2.23' not found ################################

1. virtualenv -p python3 .t
2. source .tf/bin/activate

#important set up environment for tensorflow
3. sudo -H pip3.7 install --upgrade tensorflow

4. jupyter notebbok
