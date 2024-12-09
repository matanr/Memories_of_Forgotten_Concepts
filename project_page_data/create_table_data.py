import os.path as osp

concepts = ["nudity", "church", "garbage_truck", "parachute", "tench",  "vangogh"]
models = ["vanilla", "EraseDiff", "ESD", "AdvUnlearn", "SPM", "Salun", "Scissorhands", "FMN", "UCE", "AC"]

if __name__ == "__main__":
    base_dir = "project_page_data/images/generations"
    print_str = "<tbody>\n"
    for model in models:
        model_print_name = model[0].upper() + model[1:]
        print_str += f'<tr>\n<td style="vertical-align: middle; font-weight: bold;">{model_print_name}</td>\n'
        for concept in concepts:
            simple_path = osp.join(base_dir, f"{concept}_{model}_simple.webp")
            memory_path = osp.join(base_dir, f"{concept}_{model}_memory.webp")
            print_str += "<td>\n"
            if osp.exists(simple_path):
                print_str += f'<img class="table_img is-responsive" src="{simple_path}" loading="lazy" alt="{model} {concept} generation simple">\n'
            print_str += "</td>\n<td>\n"
            if osp.exists(memory_path):
                print_str += f'<img class="table_img is-responsive" src="{memory_path}" loading="lazy" alt="{model} {concept} generation memory">\n'
            print_str += "</td>\n"
        print_str += "</tr>\n"
    print_str += "</tbody>"
    print(print_str)
            