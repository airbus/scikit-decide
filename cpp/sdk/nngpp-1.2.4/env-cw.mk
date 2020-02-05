
CXX = /usr/local/gcc-8.3.0/bin/g++

LIBRARIES = /mnt/c/Dev
LINUX_LIBS = /home/chris/lib

CATCH_INC = $(LIBRARIES)/Catch2/single_include

MBEDTLS_DIR = $(LINUX_LIBS)/mbedtls-2.13.0
MBEDTLS_LIB = $(MBEDTLS_DIR)/lib
MBEDTLS_LNK = $(MBEDTLS_LIB)/libmbedtls.a $(MBEDTLS_LIB)/libmbedx509.a $(MBEDTLS_LIB)/libmbedcrypto.a

NNG_DIR = $(LINUX_LIBS)/nng
NNG_INC = $(NNG_DIR)/include
NNG_LNK = $(NNG_DIR)/lib/libnng.a $(MBEDTLS_LNK)
