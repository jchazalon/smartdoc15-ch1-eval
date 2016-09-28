#!/bin/bash

timestamp=$(date +%Y-%m-%d_%H%M%S)

# Path where scripts are stored
export SDC_TOOLS="/data/competitions/2015-ICDAR-smartdoc/challenge1/smartdoc-c1-eval"

# Path to the root of the directory used to store evaluation files
export SDC_ROOT="/data/competitions/2015-ICDAR-smartdoc/challenge1"
# Place participants / methods outputs here (ex: meth1/background1/datasheet001.segresult.xml, etc.)
export SDC_PART="${SDC_ROOT}/03-participants_outputs"
# Place where ground truth is stored
export SDC_GT="${SDC_ROOT}/04-ground_truth"
# Place where evaluation results will be stored
export SDC_EVAL="${SDC_ROOT}/05-evaluation"
# Place where analysis results will be stored
export SDC_ANALYSIS="${SDC_ROOT}/06-analysis"

export SDC_BACKGROUNDS=$(seq -f 'background0%.0f' 5)
export SDC_DOCUMENTS=$(echo {datasheet,letter,magazine,paper,patent,tax}{001,002,003,004,005})


# Create output directories
find ${SDC_PART} -mindepth 1 -maxdepth 1 -type d -print0 | \
parallel -0 -a - \
    mkdir -p ${SDC_EVAL}/{1/}/{2} \
    ::: $SDC_BACKGROUNDS


# Check outputs completeness
find ${SDC_PART} -mindepth 1 -maxdepth 1 -type d -print0 | \
parallel -0 -a - \
    'test -e "${SDC_PART}/{1/}/{2}/{3}.segresult.xml" || echo MISSING "${SDC_PART}/{1/}/{2}/{3}.segresult.xml"' \
       ::: $SDC_BACKGROUNDS \
       ::: $SDC_DOCUMENTS \
   2>&1 | tee ${SDC_ROOT}/00-missing_outputs_${timestamp}.log


# Evaluate segmentation outputs
find ${SDC_PART} -mindepth 1 -maxdepth 1 -type d -print0 | \
parallel -0 -a - \
    python $SDC_TOOLS/eval_seg.py -d \
        ${SDC_GT}/{2}/{3}.gt.xml \
        ${SDC_PART}/{1/}/{2}/{3}.segresult.xml \
        -o ${SDC_EVAL}/{1/}/{2}/{3}.segeval.xml \
       ::: $SDC_BACKGROUNDS \
       ::: $SDC_DOCUMENTS \
   2>&1 | tee ${SDC_ROOT}/01-eval_seg_${timestamp}.log


# To merge segmentation results
find ${SDC_EVAL} -mindepth 1 -maxdepth 1 -type d -print0 | \
parallel -0 -a - \
    'find ${SDC_EVAL}/{1/}/{2} -type f -iname "*.segeval.xml" | \
        python $SDC_TOOLS/merge_evalres.py -d \
        -f - \
        -o ${SDC_EVAL}/{1/}/{2}.evalsummary.xml' \
       ::: $SDC_BACKGROUNDS \
   2>&1 | tee ${SDC_ROOT}/02-merge_evalres_${timestamp}.log

find ${SDC_EVAL} -mindepth 1 -maxdepth 1 -type d -print0 | \
parallel -0 -a - \
    'find ${SDC_EVAL}/{1/} -maxdepth 1 -type f -iname "*.evalsummary.xml" | \
        python $SDC_TOOLS/merge_evalres.py -d \
        -f - \
        -o ${SDC_EVAL}/{1/}/BACKGROUND-ALL.evalsummary.xml' \
   2>&1 | tee ${SDC_ROOT}/03-merge_evalres_${timestamp}.log


# Generate CSV summary from results
find ${SDC_EVAL} -mindepth 1 -maxdepth 1 -type d -print0 | \
parallel -0 -a - \
    'find ${SDC_EVAL}/{1/} -iname "*.evalsummary.xml" | \
        sort | \
         python $SDC_TOOLS/evalsum_to_csv.py  -f - -o ${SDC_ANALYSIS}/{1/}.summary.csv' \
   2>&1 | tee ${SDC_ROOT}/04-evalsum_to_csv_${timestamp}.log


# Extract Jaccard Index for each frame, for each method

find ${SDC_PART} -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | \
parallel "grep -r '<jaccard_index_smartdoc>' ${SDC_EVAL}/{} | \
          sed 's:^${SDC_EVAL}/::g; 
               s/.segeval.xml://g;
               s:<jaccard_index_smartdoc>::g;
               s:</jaccard_index_smartdoc>::g;
               s/      /\t/g;
               s:/:\t:g;' \
          > ${SDC_ANALYSIS}/{}.ji-all-frames.csv"

find ${SDC_PART} -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | \
sort | \
parallel -j1 -X cat "${SDC_ANALYSIS}/{}.ji-all-frames.csv" | \
sort  | \
awk 'BEGIN { 
      frame=0;
      prev_doc="";
      OFS="\t"
    }
    { 
      if (prev_doc == $3) {
        frame+=1;
      } else {
        frame=1; 
        prev_doc=$3
      }; 
      print $1, $2, $3, frame, $4;  
    }' > ${SDC_ANALYSIS}/GLOBAL.ji-all-frames.csv

find ${SDC_PART} -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | \
parallel -X rm "${SDC_ANALYSIS}/{}.ji-all-frames.csv"

python $SDC_TOOLS/smartdoc_ji_overview.py "${SDC_ANALYSIS}/GLOBAL.ji-all-frames.csv" "${SDC_ANALYSIS}"


