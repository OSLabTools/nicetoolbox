#!/bin/bash


# install toml-cli if toml is not already installed
if ! command -v toml &> /dev/null
then
    echo "Installing package 'toml-cli'..."
    python -m pip install toml-cli -y
    echo "Installation done."
else 
    echo "Installation of 'toml' found."
fi

# define the output directory
output_directory="/is/sg2/cschmitt/pis/experiments/20240613_run_tests"
# create the output_directory if needed
mkdir -p $output_directory 

# create a tmp folder
tmp_folder="$output_directory/tmp"
mkdir -p $tmp_folder 

# Log file to store timings and folder sizes
log_file="$output_directory/log.txt"

# Save working directory to logfile
echo -e "\nRunning all tests defined in   './configs/tests/*.toml' and" > $log_file
echo -e "Saving the results all runs to '$output_directory'." >> $log_file
echo -e "\n\n------------------------------------\n\n" >> $log_file

# copy the machine_specific_paths.toml
machine_specifics="$tmp_folder/machine_specific_paths.toml"
cp ./configs/machine_specific_paths.toml $machine_specifics

for run_file in $(find ./configs/tests/*.toml)
do
    # skip any detector config
    if grep -q "detectors" <<< "$run_file"
    then
        continue
    fi

    # find the experiment_name
    experiment_name=$(toml get --toml-path $run_file io.experiment_name)

    # create a copy of the run_file
    run_config="$tmp_folder/"$experiment_name".toml"
    cp $run_file $run_config
    run_config_check="$tmp_folder/"$experiment_name"_check.toml"
    cp ./configs/run_file_check.toml $run_config_check

    # overwrite the experiment_folder used by the isa-tool to save this run's results
    experiment_folder="$output_directory/$experiment_name"
    toml set --toml-path $run_config io.out_folder $experiment_folder

    # copy the detectors_config.toml
    number_detectors=$(find ./configs/tests/ -type f -name "*detector*.toml" -iname "*$experiment_name*" | wc -l)
    if [ $number_detectors -eq "1" ] #&& [ -d $detectors_config] 
    then
        input_detector=$(find ./configs/tests/ -type f -name "*detector*.toml" -iname "*$experiment_name*")
    else
        input_detector="./configs/tests/detectors_config.toml"
    fi
    detectors_config="$tmp_folder/"$experiment_name"_detectors.toml"
    cp $input_detector $detectors_config

    # add log entry for this run file
    echo -e "Start running experiment '$experiment_name'.\n" >> $log_file
    echo -e "Base run_file:           $run_file" >> $log_file
    echo -e "Base detectors_config:   $input_detector         ($number_detectors found)" >> $log_file
    # log input files
    echo -e "Experiment folder path:  $experiment_folder" >> $log_file
    echo -e "Run config path:         $run_config" >> $log_file
    echo -e "Detectors config path:   $detectors_config" >> $log_file
    echo -e "Machine specifics path:  $machine_specifics\n" >> $log_file

    # -------------------------------------------------------------------------

    # Record start time
    start_time=$(date +%s)
    echo "Start time:   $(date -d @$start_time)" >> $log_file

    # run the experiment
    python main.py --run_config $run_config --detectors_config $detectors_config --machine_specifics $machine_specifics

    # Record end time
    end_time=$(date +%s)
    echo "End time:     $(date -d @$end_time)" >> $log_file

    # Calculate and log the duration
    duration=$((end_time - start_time))
    hours=$(($duration / 3600))
    minutes=$(( ($duration - ($hours * 3600)) / 60))
    seconds=$(( $duration - ($hours * 3600) - ($minutes * 60)))
    echo "Duration:     "$hours"h "$minutes"m "$seconds"s" >> $log_file

    # If the experiment created an output folder, get its size
    if [ -d $experiment_folder ]
    then 
        # Get the size of the folder    
        folder_size=$(du -sh $experiment_folder | cut -f1)
        echo "Folder size:  $folder_size" >> $log_file
    else
        echo -e "\n>>>>>>>>>> ERROR <<<<<<<<<<<<" >> $log_file
        echo -e "Directory '$experiment_folder' does not exist!" >> $log_file
        echo -e ">>>>>>>>>> ERROR <<<<<<<<<<<<" >> $log_file
    fi

    echo -e "\n\n------------------------------------\n\n" >> $log_file
done

echo "All runs completed. Results saved in $log_file."


# unused commands
#-----------------

    # skip the machine_specifics file we just created
    # if [[ "$run_file" == "$machine_specifics" ]]
    # then
    #     continue
    # fi

# update the output_directory
# toml set --toml-path $machine_specifics pis_folder_path $output_directory

    # experiment_folder=$(toml get --toml-path $run_config io.out_folder)
    # experiment_folder="${experiment_folder/<experiment_name>/"$experiment_name"}"
    # experiment_folder="${experiment_folder/<pis_folder_path>/"$output_directory"}"