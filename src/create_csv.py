import pandas as pd


def create_csv_artist_small():
    csv_path = "data/concept_csv/artist_30.csv"
    cartoon_character = {}
    cartoon_character["prompt"] = []
    cartoon_character["evaluation_seed"] = []
    cartoon_character["case_number"] = []
    cartoon_character["concept"] = []
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        if index % 25 < 10:
            cartoon_character["prompt"].append(row.prompt)
            cartoon_character["evaluation_seed"].append(row.evaluation_seed)
            cartoon_character["case_number"].append(index)
            cartoon_character["concept"].append(row.concept)
    df = pd.DataFrame(cartoon_character)
    df.to_csv("data/concept_csv/artist_30_small.csv")
    return


def create_csv(data_name, csv_name, seed_num):
    cartoon_character = {}
    cartoon_character["prompt"] = []
    cartoon_character["evaluation_seed"] = []
    cartoon_character["case_number"] = []
    cartoon_character["concept"] = []
    data_path = f"data/concept_text/{data_name}.txt"
    csv_path = f"data/concept_csv/{csv_name}.csv"
    format_path = f"data/concept_text/10_others_template.txt"
    prompt_formats = []
    with open(format_path, "r") as f:
        for line in f:
            prompt_formats.append(line.strip())
    with open(data_path, "r") as f:
        i = 0
        for line in f:
            concept = line.strip("\n")
            for prompt_format in prompt_formats:
                for seed_tem in range(seed_num):
                    prompt = prompt_format.format(concept, "{}")
                    cartoon_character["prompt"].append(prompt)
                    cartoon_character["evaluation_seed"].append(seed_tem)
                    cartoon_character["case_number"].append(i)
                    cartoon_character["concept"].append(concept)
                    i += 1
    df = pd.DataFrame(cartoon_character)
    df.to_csv(csv_path)


def change_csv(csv_path, csv_name, edit_prompt):
    cartoon_character = {}
    prompt_formats = [
        "{} {} sits on the chair",
        "{} {} stand on the grassland",
        "Full body shot of {} {}"
    ]
    cartoon_character["prompt"] = []
    cartoon_character["evaluation_seed"] = []
    cartoon_character["case_number"] = []
    cartoon_character["concept"] = []
    csv_path = f"data/{csv_name}.csv"
    df = pd.read_csv(csv_path)
    i = 0
    for _, row in df.iterrows():
        cartoon_character["evaluation_seed"].append(row.evaluation_seed)
        cartoon_character["case_number"].append(row.case_number)
        cartoon_character["concept"].append(row.concept)
        cartoon_character["prompt"].append(prompt_formats[i % len(prompt_formats)].format(row.concept, edit_prompt))
        i += 1
    new_df = pd.DataFrame(cartoon_character)
    new_df.to_csv(f"data/{csv_name}_{edit_prompt}.csv")

def create_result_csv():
    concept_path = "data/concept_text/IP_character_concept_10.txt"
    concept_list = []
    with open(concept_path, "r") as f:
        for line in f:
            concept_list.append(line.strip())

    return
if __name__ == "__main__":
    data_name = "generate_IP_10_others"
    csv_name = "10_others"
    method = "create_result_csv"
    # create_csv(data_name, csv_name, 5)

    create_csv(data_name=data_name, csv_name=csv_name, seed_num=20)
    # change_csv(csv_name,"wearing sunglasses")
