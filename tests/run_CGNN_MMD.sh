#!/bin/bash

source activate py35

for h in {40,30,20,10,50,60,80,100,200,1000}
do
for a in {0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001}
do
python run_CGNN_decomposable_graph_embedded.py graph/G2_v1_numdata.tab graph/G2_v1_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G2_v2_numdata.tab graph/G2_v2_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G2_v3_numdata.tab graph/G2_v3_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G2_v4_numdata.tab graph/G2_v4_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G2_v5_numdata.tab graph/G2_v5_target.tab $a $h

python run_CGNN_decomposable_graph_embedded.py graph/G3_v1_numdata.tab graph/G3_v1_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G3_v2_numdata.tab graph/G3_v2_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G3_v3_numdata.tab graph/G3_v3_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G3_v4_numdata.tab graph/G3_v4_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G3_v5_numdata.tab graph/G3_v5_target.tab $a $h

python run_CGNN_decomposable_graph_embedded.py graph/G4_v1_numdata.tab graph/G4_v1_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G4_v2_numdata.tab graph/G4_v2_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G4_v3_numdata.tab graph/G4_v3_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G4_v4_numdata.tab graph/G4_v4_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G4_v5_numdata.tab graph/G4_v5_target.tab $a $h

python run_CGNN_decomposable_graph_embedded.py graph/G5_v2_numdata.tab graph/G5_v2_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G5_v5_numdata.tab graph/G5_v5_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G5_v1_numdata.tab graph/G5_v1_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G5_v3_numdata.tab graph/G5_v3_target.tab $a $h
python run_CGNN_decomposable_graph_embedded.py graph/G5_v4_numdata.tab graph/G5_v4_target.tab $a $h
done
done


