import pandas as pd
import numpy as np
import os, os.path
import glob

# Basic path (where the data is stored)
basic_path = "C:\\Users\\Riccardo\\Google Drive\\University\\Double Degree - Middlesex\\Middlesex Teaching Material\\CSD4444 - Ralph Moseley\\Data\\"
# Simulation folders
simulation_folder_path = basic_path + "Real data\\SimulationEQUAL\\"
# Output folder
output_folder_path = basic_path + "Manipulated data\\"
# Output merged single file
output_merged_csv_pathname = output_folder_path + "Merged\\all_merged.csv"

heights = ["50", "100", "150"]
widths = ["50", "100", "150"]


def create_csv_data(restore=False, selected_height="50", selected_width="50"):
    # output csv (merged) file
    output_merged_file_path = output_folder_path + "merged_equal_" + selected_height + "x" + selected_width + ".csv"

    if restore:
        # ----------------------------------------------------------
        # READ ALREADY EXTRACTED DATA FROM CSV
        # ----------------------------------------------------------
        dataframe = pd.read_csv(output_merged_file_path)
    else:
        # ----------------------------------------------------------
        # CREATE THE CSVs
        # ----------------------------------------------------------
        file_list = os.listdir(simulation_folder_path)  # dir is your directory path

        dictionary = {'HEIGHT': [], 'WIDTH': [], 'NODE': [], 'R0': [], '%AGGR': [], 'HET': [], 'HOM ENERGY': [],
                      'HOM RATE': [], 'REECHD FND': [], 'REECHD HND': [], 'HEED FND': [], 'HEED HND': [],
                      'ERHEED FND': [], 'ERHEED HND': [], 'FMUC FND': [], 'FMUC HND': []}

        for i in range(0, len(file_list)):
            current_file_name = file_list[i]

            width, height, hom_energy, hom_rate, perc_aggr, heterogeneity, transm_range_e = current_file_name.split("_")
            """
            width:              W...        [1:] 
            height:             H...        [1:] 
            hom_energy:         Henergy...  [7:] 
            hom_rate:           Hrate...    [5:]
            perc_aggr:          Agg...      [3:]
            heterogeneity:      Hperc...    [5:]
            transm_range_e:     R0...       [2:]
            """

            width = width[1:]
            height = height[1:]
            hom_energy = hom_energy[7:]
            hom_rate = hom_rate[5:]
            perc_aggr = perc_aggr[3:]
            if perc_aggr == "100percentAggregation":
                perc_aggr = 1
            elif perc_aggr == "70percentAggregation":
                perc_aggr = 0.7
            elif perc_aggr == "40percentAggregation":
                perc_aggr = 0.4
            elif perc_aggr == "NoAggregation":
                perc_aggr = 0
            heterogeneity = heterogeneity[5:]
            transm_range, extension = str(transm_range_e[2:]).split(".")

            if width == selected_width and height == selected_height:
                """
                print(current_file_name, " - ", hom_energy, ", ", hom_rate, ", ", perc_aggr, ", ", heterogeneity, ", ",
                      transm_range)
                """

                dictionary.get("HEIGHT").append(selected_height)
                dictionary.get("WIDTH").append(selected_width)
                dictionary.get("NODE").append(100)

                dictionary.get("R0").append(transm_range)
                dictionary.get("%AGGR").append(perc_aggr)
                dictionary.get("HET").append(heterogeneity)
                dictionary.get("HOM ENERGY").append(hom_energy)
                dictionary.get("HOM RATE").append(hom_rate)

                simulation_dataframe = pd.read_csv(simulation_folder_path + current_file_name, header=None)

                REECHD_FND, REECHD_HND = str(simulation_dataframe.iat[1, 0]).split(";")
                HEED_FND, HEED_HND = str(simulation_dataframe.iat[3, 0]).split(";")
                ERHEED_FND, ERHEED_HND = str(simulation_dataframe.iat[5, 0]).split(";")
                FMUC_FND, FMUC_HND = str(simulation_dataframe.iat[7, 0]).split(";")

                dictionary.get("REECHD FND").append(REECHD_FND)
                dictionary.get("REECHD HND").append(REECHD_HND)
                dictionary.get("HEED FND").append(HEED_FND)
                dictionary.get("HEED HND").append(HEED_HND)
                dictionary.get("ERHEED FND").append(ERHEED_FND)
                dictionary.get("ERHEED HND").append(ERHEED_HND)
                dictionary.get("FMUC FND").append(FMUC_FND)
                dictionary.get("FMUC HND").append(FMUC_HND)

        dataframe = pd.DataFrame(data=dictionary, columns=dictionary.keys())

        # ----------------------------------------------------------
        # STORING DATA TO A SIMPLEST CSV
        # ----------------------------------------------------------

        dataframe.to_csv(output_merged_file_path, index=False)
        print(dataframe)
        print("---------------------------------------------------------")


def merge_csv_files():
    all_files = glob.glob(os.path.join(output_folder_path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv(output_merged_csv_pathname, index=False)


def load_data():
    dataframe = pd.read_csv(output_merged_csv_pathname, header=None)

    column_names = list(dataframe.columns.values)

    input_cols = [column_names[0], column_names[1], column_names[2], column_names[3], column_names[4], column_names[5],
                  column_names[6], column_names[7]]
    output_cols = [column_names[8], column_names[9], column_names[10], column_names[11], column_names[12],
                   column_names[13], column_names[14], column_names[15]]

    inputs = dataframe.as_matrix(input_cols)[1:]
    outputs = dataframe.as_matrix(output_cols)[1:]
    in_headers = np.squeeze(dataframe.as_matrix(input_cols)[:1])
    out_headers = np.squeeze(dataframe.as_matrix(output_cols)[:1])

    return inputs, outputs, in_headers, out_headers


if __name__ == "__main__":
    for h in heights:
        for w in widths:
            create_csv_data(selected_height=h, selected_width=w)
    merge_csv_files()
