TRAINNUMBER="300243"
OUTPUTPATH="/alice/cern.ch/user/a/alihyperloop/outputs/0030/300243/53091"
DATASETTAG="LHC22o_pass7"
SUFFIX="K0s_Lambda"

mkdir -p datasets/$DATASETTAG/Train$TRAINNUMBER

alien_cp $OUTPUTPATH/AO2D.root file:datasets/$DATASETTAG/Train$TRAINNUMBER/AO2D.root

echo "datasets/$DATASETTAG/Train$TRAINNUMBER/AO2D.root" >> files_to_merge.txt
o2-aod-merger --input files_to_merge.txt --max-size 10000000000000 --output datasets/$DATASETTAG/Train$TRAINNUMBER/Tree_"$DATASETTAG"_$SUFFIX.root

rm files_to_merge.txt
rm datasets/$DATASETTAG/Train$TRAINNUMBER/AO2D.root
