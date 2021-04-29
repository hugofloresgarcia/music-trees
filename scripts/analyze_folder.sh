# first argument should be the parent folder for all runs to be run
# second argument should be the output name of this analysis
for dir in "${1}"/*
do
  echo ${dir}
  python3.7 music_trees/analyze.py ${dir} ${2}$
done