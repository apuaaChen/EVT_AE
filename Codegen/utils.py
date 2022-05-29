import os


def generate_code(template, parameter, file_name="config.h"):
    if not os.path.isdir("scratch_space"):
        os.mkdir("sratch_space")
    config_file_path = os.path.join("scratch_space", file_name)
    # clean previous configuration
    try:
        os.remove(config_file_path)
    except OSError:
        pass
    file = open(config_file_path, 'w+')
    if parameter is None:
        file.write(template)
    else:
        parameters = tuple(parameter)

        file.write(template % parameters)


def get_kernel_name(config):
    name = ""
    if config["lhs_format"] == "row_sp":
        name += "SpMM_n"
    
    if config["rhs_format"] == "row":
        name += "n"
    elif config["rhs_format"] == "col":
        name += "t"
    
    if config["out_format"] == "row":
        name += "n"
    elif config["out_format"] == "col":
        name += "t"
    
    return name