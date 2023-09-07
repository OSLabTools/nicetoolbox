
# exit on error rather than trying to continue
set -e

# testing whether conda is available
echo -n "Ensuring conda is available...   "
if which conda >/dev/null; then
    echo "OK."
else
    echo "Please make sure a conda version (miniconda or anaconda) is available."
    exit 1
fi
echo ""


### NODDING PIGEON
conda_env_name="nodding_pigeon"

# test whether the conda environment already exists, create it if it doesn't
echo "Ensuring conda environment '$conda_env_name' is available... "

# create a conda environment
if conda info --env | grep -w "$conda_env_name" >/dev/null; then
    echo "Already exists."
else
    echo "Creating minimal environment (this might take a while) ... "
    conda create --name $conda_env_name python=3.8  pip --yes >/dev/null
    conda install --name $conda_env_name --file ./detectors/nodding/requirements.txt --yes
    echo "Created successfully."
fi


# activate the conda environment
# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh

echo "\n\nInstallation finished."
