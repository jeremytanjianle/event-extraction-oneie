# Download the raw RAMS dataset and run through the pre-processing script 
# and put them in the `data` folder
# Usage: From the main project folder (/oneie), run `bash preprocessing/rams/get_rams.sh`

# Need to explicitly export path because
export PATH=$PATH

echo $PATH

out_dir=./data/rams
process_path=./preprocessing/rams

mkdir -p $out_dir
mkdir {$out_dir/raw,$out_dir/processed-data,$out_dir/processed-data/json}

# Download.
wget https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz -P $out_dir

# Decompress.
tar -xf $out_dir/RAMS_1.0b.tar.gz -C $out_dir/raw

# Install jsonlines (which is not part of Dygiepp's requirements.txt)
pip install jsonlines

# Install pandas because pandas not installed to host
pip install pandas

# Process
python $process_path/parse_rams_to_dygiepp.py

# Clean up.
rm $out_dir/*.tar.gz
