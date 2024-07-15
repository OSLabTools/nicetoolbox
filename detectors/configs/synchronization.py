import os

import configs.config_handler as confh


remote_datasets_folder_path = "/ps/project/datasets/oslab"


def sync_dataset_configs(filename='dataset_properties.toml'):
    def listdir_absolut_paths(path):
        return [name for name in sorted(os.listdir(path))
                if os.path.isdir(os.path.join(path, name))]

    def get_user_input(text, options):
        valid_user_input = False
        while not valid_user_input:
            user_input = input(text)
            valid_user_input = user_input in options

        return user_input

    machine_specifics_file = "configs/machine_specific_paths.toml"
    machine_config = confh.load_config(machine_specifics_file)
    local_dataset_path = machine_config['datasets_folder_path']
    remote_dataset_path = remote_datasets_folder_path

    local_dataset_names = set(listdir_absolut_paths(local_dataset_path))
    remote_dataset_names = set(listdir_absolut_paths(remote_dataset_path))
    dataset_names_cut = local_dataset_names.intersection(remote_dataset_names)

    for dataset_name in dataset_names_cut:
        print(f"\nSynchronizing dataset {dataset_name}...")
        remote_file = os.path.join(remote_dataset_path, dataset_name, filename)
        if not os.path.exists(remote_file):
            print(f"{dataset_name}: Found no {filename} file.")
            continue

        local_file = os.path.join(local_dataset_path, dataset_name, filename)
        copy_remote = False
        if not os.path.exists(local_file):
            print(f"{dataset_name}: File {filename} was found on the remote "
                  f"but not locally.")
            user_input = get_user_input("Copy it (y/n)?", ['y', 'n'])
            copy_remote = user_input == 'y'

        else:
            remote = confh.load_config(remote_file)
            local = confh.load_config(local_file)
            keys_differ, values_differ = confh.compare_configs(
                    remote, local, log_fct=print, config_names=filename)
            if keys_differ or values_differ:
                print(f"{dataset_name}: Detected differences in the remote "
                      f"and local {filename} files.")
                user_input = get_user_input("Update the local file (y/n)?", ['y', 'n'])
                copy_remote = user_input == 'y'
            else:
                print(f"{dataset_name}: remote & local files {filename} match.")

        if copy_remote:
            os.system(f'cp {remote_file} {os.path.dirname(local_file)}')

    print(f"\nSynchronization completed.\n")


if __name__ == '__main__':
    sync_dataset_configs()

