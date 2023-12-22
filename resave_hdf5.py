import h5py

def compress_dataset(dataset: h5py.File, output_file: h5py.File, compression_algorithm = 'gzip', compression_level = 9) -> None:

    # Iterate through datasets and apply compression settings
    for key in dataset.keys():
        input_item = dataset[key]
        # print(key)
        
        if isinstance(input_item, h5py.Dataset):
            # Create a corresponding dataset in the output file with compression options
            output_dataset = output_file.create_dataset(
                key,
                shape=input_item.shape,
                dtype=input_item.dtype,
                chunks=input_item.chunks,
                compression=compression_algorithm,
                compression_opts=compression_level
            )
            
            # Copy data from the input dataset to the output dataset
            output_dataset[...] = input_item[...]

        elif isinstance(input_item, h5py.Group):
            output_file.create_group(key)
            compress_dataset(input_item, output_file[key], compression_algorithm, compression_level)


def compress_h5py(input_file_path, output_file_path, compression_algorithm = 'gzip', compression_level = 9)  -> None:

    # Open the input file in read mode
    with h5py.File(input_file_path, 'r') as input_file:
        # Create or open the output file in write mode with compression options
        with h5py.File(output_file_path, 'w') as output_file:
            # Copy the attributes from the input file to the output file
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value

            # Compress the datasets
            compress_dataset(input_file, output_file, compression_algorithm, compression_level)

    print(f'File "{input_file_path}" has been resaved with updated compression settings.')

if __name__ == '__main__':
    import os
    import shutil

    task_name = "push"
    
    DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data/" + task_name + "/"
    UNCOMPRESSED_DATA_DIR = DATA_DIR[:-1] + "_uncompressed/"

    for num_sims in [25, 50, 100, 200, 400]:
        if not os.path.exists(DATA_DIR + str(num_sims)):
            os.mkdir(DATA_DIR + str(num_sims))
        for run_num in range(1, 6):
            if not os.path.exists(DATA_DIR + str(num_sims) + "/run" + str(run_num)):
                os.mkdir(DATA_DIR + str(num_sims) + "/run" + str(run_num))
            if not os.path.exists(DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/data"):
                os.mkdir(DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/data")
            shutil.copy(UNCOMPRESSED_DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/meta_data.json", DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/meta_data.json")
            for h5py_file in os.listdir(UNCOMPRESSED_DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/data"):
                if h5py_file.endswith(".hdf5"):
                    print("compressing", num_sims, run_num, h5py_file)
                    compress_h5py(UNCOMPRESSED_DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/data/" + h5py_file, DATA_DIR + str(num_sims) + "/run" + str(run_num) + "/data/" + h5py_file)
