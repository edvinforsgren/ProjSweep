from CGAN_train import run_training


def read_file_lines(path_to_config):
    with open(path_to_config) as f:
        lines = f.readlines()

    rm_lines = []
    for line in lines:
        if line.strip().startswith("#"):
            rm_lines.append(line)
    for rm_line in rm_lines:
        lines.remove(rm_line)
    return lines


# Specify your path to your config.txt
path_to_config = "./config.txt"
lines = read_file_lines(path_to_config)
save_dir, input_path, target_path, n_epochs = [line[:-1] for line in lines]
run_training(save_dir=save_dir, target_path=target_path, input_path=input_path, n_epochs=n_epochs)
