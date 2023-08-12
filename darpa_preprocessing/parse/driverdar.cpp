#include "driverdar.h"

int main(int argc, char **argv) {
	// parse command line arguments
	Config cfg(argc, argv);
	cfg.ConfigDar();
	// Knowledge graph for darpa
	KG *infotbl = new KG(darpa);
	
	// define Local File in KG (no computation)
	LocalStore ls(cfg.embed_data_path, infotbl);


	// KG Construction
	// 1. load kg from db/file 
	// 2. parse logs to construct kg
	if (cfg.loadentity) {
		// load system entities from local files
		EntityLoadFromFile(cfg.embed_data_path, infotbl);
		// print KG information
		infotbl->PrintKG();
	}

	if (cfg.loadfromfile) {
		// load knowledge graph from local files
		KGLoadFromFile(cfg.embed_data_path, infotbl);
		// print KG information
		infotbl->PrintKG();
	}
	else {
		// collect darpa files under darpa_data_dir
		auto darpa_files = CollectJsonFile(cfg.darpa_data_dir);

		int file_id = 0;
		for (auto darpa_file: darpa_files) {
			// construct KG (work_threads shows #threads to parse dataset)
			KGConstruction(darpa_file, infotbl, cfg);

			// print KG information
			infotbl->PrintKG();
			
			/*
			if (cfg.storeentity) {
				ls.EntityStoreToFile();
			}
			*/

			if (cfg.storetofile) {
				ls.KGStoreToFile(file_id);
				file_id += 1;
				}


			// to save memory, we clean up KG after parsing a darpa file
			infotbl->FreeInteraction();
		}

		/*
		if (cfg.storetofile || cfg.storeentity) {
			ls.DumpProcFileSocketEdge2FactSize(file_id - 1);
		}
		*/
	}
	ls.EntityStoreToFile();
	infotbl->FreeInteraction();
	infotbl->FreeNode();
	delete (infotbl);
	return 0;
}
