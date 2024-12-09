import os.path as osp

concepts = ["nudity", "church", "garbage_truck", "parachute", "tench",  "vangogh"]
models = ["vanilla", "EraseDiff", "ESD", "AdvUnlearn", "SPM", "Salun", "Scissorhands", "FMN", "UCE", "AC"]

if __name__ == "__main__":
    base_dir = "project_page_data/images/generations"
    width = 0.15
    print_str = ""
    for model in models:
        model_print_name = model[0].upper() + model[1:]
        print_str += r"\raisebox{0.075\linewidth}{" + f"{model_print_name}}} & "
        for idx, concept in enumerate(concepts):
            simple_path = osp.join(base_dir, f"{concept}_{model}_simple.png")
            memory_path = osp.join(base_dir, f"{concept}_{model}_memory.png")
            if osp.exists(simple_path):
                print_str += f'\includegraphics[width={width}\linewidth]{{{simple_path.replace("project_page_data/images/", "")}}}'
            print_str += "& "
            if osp.exists(memory_path):
                print_str += f'\includegraphics[width={width}\linewidth]{{{memory_path.replace("project_page_data/images/", "")}}}'
            if idx != len(concepts) - 1:
                print_str += "& "
            else:
                print_str += "\\tabularnewline\n"
    print(print_str)
            