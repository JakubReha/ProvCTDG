#include <iostream>
#include <unordered_map>
#include <sstream>
#include <jsoncpp/json/json.h>

std::string Jval2str(const Json::Value jval) {
	Json::FastWriter fastWriter;
	std::string tmp = fastWriter.write(jval);
	//std::cout << "json value: "<< tmp <<"\n";
	if (tmp[0] == '\"') {
		return tmp.substr(1, tmp.rfind("\"") - 1);
	}
	else {
		return tmp.substr(0, tmp.rfind("\n"));
	}
}


void printID(std::string name) {
    //std::cout << name <<"\n";
	name = Jval2str(name);
	std::hash<std::string> hasher;
    std::size_t hash = hasher(name);
	printf("%zu\n",hash);
}

int main(int argc, char *argv[])
{
	std::string name(argv[1]);
	printID(name);
}