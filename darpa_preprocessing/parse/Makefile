# target
TARGET = driverdar

# dir
SRC_DIR = $(shell pwd)
#PROJ_DIR = $(shell dirname $(SRC_DIR))
#LIB_DIR = $(PROJ_DIR)/lib/lib
#INC_DIR = $(PROJ_DIR)/lib/include
LIB_DIR = $(SRC_DIR)/lib/lib
INC_DIR = $(SRC_DIR)/lib/include
PARSER_DIR = $(SRC_DIR)/parser
UTIL_DIR = $(SRC_DIR)/util
DARPA_DIR = $(PARSER_DIR)/darpa

# src (Todo: polish graphviz)
PARSER_SRC = $(wildcard $(PARSER_DIR)/*.cpp)
UTIL_SRC = $(filter-out $(UTIL_DIR)/graphviz.cpp, $(wildcard $(UTIL_DIR)/*.cpp))
DARPA_SRC = $(wildcard $(DARPA_DIR)/*.cpp)
DRIVERDAR_SRC = $(SRC_DIR)/driverdar.cpp

# obj
UTIL_OBJ = $(patsubst %.cpp, %.o, $(UTIL_SRC))
PARSER_OBJ = $(patsubst %.cpp, %.o, $(PARSER_SRC))
DARPA_OBJ = $(patsubst %.cpp, %.o, $(DARPA_SRC))
OBJS = $(DB_OBJ) $(UTIL_OBJ) $(PARSER_OBJ)
DAR_OBJS = $(OBJS) $(DARPA_OBJ)
DRIVERDAR_OBJ = $(patsubst %.cpp, %.o, $(DRIVERDAR_SRC))

# Option (debug/progress)
export OPT = -std=c++17 
VER = normal
ifeq ($(VER), debug)
	OPT += -O0 -g -DDEBUG
else
	OPT += -O3
endif
# display progress bar for KG Construction
ifeq ($(BAR),1)
	OPT += -DBAR
endif

# Path
export INC_PATH = -I$(INC_DIR) -I$(SRC_DIR)
JSONCPP_PATH =  $(shell pkg-config jsoncpp --cflags)
LIB_PATH = -L$(LIB_DIR)

# Libs
CONFIG_LIB = -lconfig++
JSON_LIB = -ljsoncpp
THREAD_LIB = -pthread
GRAPHVIZ_LIB = -lgvc
HASH_LIB = -lxxhash  # external lib to avoid hash collisions
LIBS += $(LIB_PATH) $(strip $(CONFIG_LIB) $(JSON_LIB) $(THREAD_LIB) $(HASH_LIB))

# Compile
CC = g++
WARNINGS = -Wall -Wextra
CXXFLAGS  = $(WARNINGS) $(OPT)

all: $(TARGET)


driverdar: $(DRIVERDAR_OBJ) $(DAR_OBJS)
	$(CC) $(CXXFLAGS) $^ -o $@ $(LIBS) $(INC_PATH)


$(DRIVERDAR_OBJ): $(DRIVERDAR_SRC)
	$(CC) $(CXXFLAGS) -c $< $(INC_PATH) $(JSONCPP_PATH)

$(UTIL_OBJ): %.o:%.cpp
	(cd $(UTIL_DIR) && $(MAKE) all)

$(PARSER_OBJ): %.o:%.cpp
	(cd $(PARSER_DIR) && $(MAKE) $(PARSER_OBJ))

$(DARPA_OBJ): %.o:%.cpp
	(cd $(PARSER_DIR) && $(MAKE) $(DARPA_OBJ))

.PHONY: clean

clean:
	rm -f *.o *.a $(TARGET)
	(cd $(UTIL_DIR) && $(MAKE) clean)
	(cd $(PARSER_DIR) && $(MAKE) clean)
