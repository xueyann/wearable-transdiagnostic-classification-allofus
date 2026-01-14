import yaml
import pandas as pd
import os

output_folder = 'results/formatted_outputs_20260114/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Helper to flatten YAML into a DataFrame
def flatten_yaml(yaml_data, code_system):
    rows = []
    for category, conditions in yaml_data.get(code_system, {}).items():
        for condition, code in conditions.items():
            rows.append({
                "Category": category.replace("_", " ").title(),
                "Condition": condition,
                "Code": code
            })
    return pd.DataFrame(rows)
    

def get_icd_excl_latex():
    # Load YAML
    with open("conf/icd_excl.yaml", "r") as f:
        data = yaml.safe_load(f)

    # Flatten both ICD-10 and ICD-9
    df_icd10 = flatten_yaml(data, "ICD10")
    df_icd9  = flatten_yaml(data, "ICD9")

    # Merge on Category + Condition (outer join to keep all)
    df = pd.merge(df_icd10, df_icd9, on=["Category", "Condition"], how="outer", suffixes=("_ICD10", "_ICD9"))

    # Combine codes into one column with line break
    df["Code"] = df["Code_ICD10"].fillna("N/A").apply(lambda x: f"ICD 10: {x}") + r" \newline " + df["Code_ICD9"].fillna("N/A").apply(lambda x: f"ICD 9: {x}")

    # Keep only relevant columns
    df = df[["Category", "Condition", "Code"]]

    df = df.set_index(["Category", "Condition"])

    # Convert to LaTeX
    latex_table = df.to_latex(index=True, 
                            column_format = r"p{4cm}>{\setstretch{0.9}\raggedright\arraybackslash}p{6cm}>{\setstretch{0.9}}p{6cm}",
                            escape = False, 
                            bold_rows = True,
                            multicolumn = True,
                            longtable=True)
    latex_table = r"{\footnotesize" + latex_table + "}"
    with open(f'{output_folder}/icd_excl.tex', 'w') as f:
        f.write(latex_table)


def get_inclusion():
    # This is based on phewas
    condition_map = yaml.safe_load(open('conf/create_study_cohort.yaml'))['total_cohort']['condition_map']
    # reverse key and value
    subcohort_map = {}
    for desc, subcohort in condition_map.items():
        if subcohort not in subcohort_map.keys():
            subcohort_map[subcohort] = [desc]
        else:
            subcohort_map[subcohort].append(desc)
    df = pd.DataFrame(pd.Series(subcohort_map, name = 'PheWAS descriptions included')).explode('PheWAS descriptions included')
    df.index.rename('Subcohort', inplace=True)
    df = df.reset_index()
    df["dummy"] = " "
    df = df.set_index(["Subcohort", "dummy"])
    print(df)

    latex_table = df.to_latex(index=True, 
                            column_format = r"p{3cm}p{0.5cm}>{\setstretch{0.9}\raggedright\arraybackslash}p{10cm}",
                            escape = False, 
                            bold_rows = True,
                            multicolumn = True,
                            multirow = True,
                            longtable=True).replace('dummy', '')
    latex_table = r"{\small" + latex_table + "}"
    with open(f'{output_folder}/phewas_desc_incl.tex', 'w') as f:
        f.write(latex_table)
    

if __name__ == '__main__':

    get_inclusion()
    get_icd_excl_latex()


    
    

