from gqlalchemy import Memgraph
from datetime import datetime
import itertools
import time as t
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import pandas as pd



data_path = "/mnt/vdc/DATA/darpa_trace_0to210"
nodes = pd.read_csv(os.path.join(data_path, "attributed_nodes.csv"))
port = 7688
memgraph = Memgraph(host='127.0.0.1', port=port)
print(nodes.head())
firefox_fp_src = memgraph.execute_and_fetch('MATCH (m:Node)-[r:EDGE]->(n) WHERE toFloat(r.TGN_no_memory_seed_0_false_positive) < 1 and (n.name="firefox") RETURN n.hash_id ORDER BY r.timestamp')
firefox_fp_dst = memgraph.execute_and_fetch('MATCH (m:Node)-[r:EDGE]->(n) WHERE toFloat(r.TGN_no_memory_seed_0_false_positive) < 1 and (m.name="firefox") RETURN m.hash_id ORDER BY r.timestamp')
print("Done")
