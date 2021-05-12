# first argument should be the parent folder for all runs to be run
# second argument should be the CUDA device to use
for dir in "${1}"/*
do
  echo ${dir}
  export CUDA_VISIBLE_DEVICES=${2}$ && python music_trees/eval.py --exp_dir ${dir}/version_0 
done
