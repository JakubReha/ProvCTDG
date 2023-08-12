# ShadeWatcher pre-processing
This code is taken from [link](https://github.com/jun-zeng/ShadeWatcher) and slightly modified.
## Environment
ShadeWatcher runs on the 16.04.3 LTS Ubuntu Linux 64-bit distribution.


## Parser Setup
We implement the audit log parser using C++. The setup has been tested using g++ 8.4.0. The required packages are as follow:

1. Installation Path: "LIB_INSTALL_PATH"
```bash
mkdir parse/lib
cd parse/lib
LIB_INSTALL_PATH=$PWD
```

2. g++ (optional)
```bash
wget https://ftp.gnu.org/gnu/gcc/gcc-8.4.0/gcc-8.4.0.tar.gz
tar xzvf gcc-8.4.0.tar.gz
cd gcc-8.4.0
contrib/download_prerequisites
./configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --prefix=$LIB_INSTALL_PATH/lib -enable-checking=release --enable-languages=c,c++,fortran --disable-multilib
make -j8
make install
cd ..
```


3. libconfig
```bash
wget https://hyperrealm.github.io/libconfig/dist/libconfig-1.7.2.tar.gz
tar xzvf libconfig-1.7.2.tar.gz
cd libconfig-1.7.2/
./configure --prefix=$LIB_INSTALL_PATH
make -j8
make install
cd ../
```

4. jsoncpp
```bash
sudo apt-get install libjsoncpp-dev
```

5. nlohmann json
```bash
cd $LIB_INSTALL_PATH/include
wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
cd ../
```

6. xxhash
```bash
sudo apt install libxxhash-dev libxxhash0
```


Setup system system library path
```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CPLUS_INCLUDE_PATH=$LIB_INSTALL_PATH/include:$CPLUS_INCLUDE_PATH
export PATH=$LIB_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$LIB_INSTALL_PATH/lib:$LIB_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
```

Make Sure you have successfully compiled driverdar (for DARPA) under `Shadewatcher/parse`.

```
make clean
make
```


Assumed file structure for the data:

```
<dataset_path>
│
└─── a
│       *.json 
│   
└─── b
        *.json 

```

## How to Use

```bash
./driverdar -h
```


Parse audit records from DARPA TC dataset (multiple threads)
```bash
cd parse
./driverdar -dataset e3_trace -trace <dataset_path> -storefile -multithread 8
./driverdar -dataset e3_theia -trace <dataset_path> -storefile -multithread 8
```
Expected Result:
```
darpa file: <dataset_path>/ta1-trace-e3-official.json
Multi-thread Configure file: ../config/multithread.cfg
Reduce noisy events
	collecting temporary file
	collecting ShadowFileEdge
	collecting ShadowProcEdge
	collecting MissingEdge
	collecting Library
	deleting nosiy events
Reduce Noise Events runtime overhead: 11.2711

KG construction runtime overhead: 25.4445

KG Statistics
#Events: 726072596
#Edge: 12661091
#Noisy events: 692252902
#Proc: 39318718
#File: 1085043
#Socket: 4051386
#Node: 44455147
#Node(44455147) = #Proc(39318718) + #File(1085043) + #Socket(4051386)

Store KG information in ../data/encoding/e3_trace/
        storing process entity
        storing file entity
        storing socket entity
        storing edges
Store KG to files runtime overhead: 144.614
```

## Aditional
printID creates the hash_id the same way that is done in the parsing code 

```
# original ShadeWatcher hashing function (hash collision)
g++ -o printID_old printID_old.cpp -ljsoncpp
```

```
# new hash function
g++ -o printID printID.cpp -lxxhash
```

For nodes the input is the UUID:
```
./printID C13A910B-8966-7C95-549F-6EACF06F2429
```

For edges the input is the UUID+timestamp (BF04D0B5-9878-6984-99F5-513732C62662 + 1522775655508000000):
```
./printID BF04D0B5-9878-6984-99F5-513732C626621522775655508000000
```