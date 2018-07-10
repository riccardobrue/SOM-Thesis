import pandas as pd
import numpy as np
import os, os.path
import glob
from sklearn.utils import shuffle as sh

nodes_number = 100

# Basic path (where the data is stored)
basic_path_desktop = "C:\\Users\\Riccardo\\Google Drive\\University\\Double Degree - Middlesex\\Middlesex Teaching Material\\CSD4444 - Ralph Moseley\\Data\\"
basic_path_laptop = "C:\\Users\\brux9\\Google Drive\\University\\Double Degree - Middlesex\\Middlesex Teaching Material\\CSD4444 - Ralph Moseley\\Data\\"

if os.path.isdir(basic_path_desktop):
    print("DESKTOP")
    basic_path = basic_path_desktop
else:
    print("LAPTOP")
    basic_path = basic_path_laptop

# Simulation folders
simulation_equal_folder_path = basic_path + "Real data\\SimulationEQUAL\\"
simulation_unequal_folder_path = basic_path + "Real data\\SimulationUNEQUAL\\"
# Output folder
output_equal_folder_path = basic_path + "Manipulated data\\Equal\\extended\\"
output_unequal_folder_path = basic_path + "Manipulated data\\Unequal\\extended\\"
# Output merged single file
output_merged_equal_csv_pathname = output_equal_folder_path + "..\\Merged\\all_merged_ext_equal.csv"
output_merged_unequal_csv_pathname = output_unequal_folder_path + "..\\Merged\\all_merged_ext_unequal.csv"

heights = ["50", "100", "150"]
widths = ["50", "100", "150"]


def create_csv_data(restore=False, selected_height="50", selected_width="50"):
    # output csv (merged) file
    output_merged_equal_file_path = output_equal_folder_path + "merged_ext_equal_" + selected_height + "x" + selected_width + ".csv"
    output_merged_unequal_file_path = output_unequal_folder_path + "merged_ext_unequal_" + selected_height + "x" + selected_width + ".csv"

    if restore:
        # ----------------------------------------------------------
        # READ ALREADY EXTRACTED DATA FROM CSV
        # ----------------------------------------------------------
        dataframe_equal = pd.read_csv(output_merged_equal_file_path)
        dataframe_unequal = pd.read_csv(output_merged_unequal_file_path)
    else:
        # ----------------------------------------------------------
        # CREATE THE CSVs
        # ----------------------------------------------------------
        # --------------------
        # EQUAL PROTOCOLS
        # --------------------
        file_list_equal = os.listdir(simulation_equal_folder_path)  # dir is your directory path

        dictionary_equal = {'HEIGHT': [], 'WIDTH': [], 'NODE': [], 'R0': [], '%AGGR': [], 'HET': [], 'HOM ENERGY': [],
                            'HOM RATE': [], 'DENSITY': [], 'REECHD AVG LAYERS': [], 'REECHD AVG CHxROUND': [],
                            'REECHD FND': [], 'REECHD HND': [], 'HEED AVG LAYERS': [], 'HEED AVG CHxROUND': [],
                            'HEED FND': [], 'HEED HND': [], 'ERHEED AVG LAYERS': [], 'ERHEED AVG CHxROUND': [],
                            'ERHEED FND': [], 'ERHEED HND': [], 'FMUC AVG LAYERS': [], 'FMUC AVG CHxROUND': [],
                            'FMUC FND': [], 'FMUC HND': []}

        for i in range(0, len(file_list_equal)):
            current_file_name = file_list_equal[i]

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

                dictionary_equal.get("HEIGHT").append(selected_height)
                dictionary_equal.get("WIDTH").append(selected_width)
                dictionary_equal.get("NODE").append(nodes_number)

                dictionary_equal.get("R0").append(transm_range)
                dictionary_equal.get("%AGGR").append(perc_aggr)
                dictionary_equal.get("HET").append(heterogeneity)
                dictionary_equal.get("HOM ENERGY").append(hom_energy)
                dictionary_equal.get("HOM RATE").append(hom_rate)

                density = nodes_number / (int(selected_height) * int(selected_width))
                dictionary_equal.get("DENSITY").append(density)

                simulation_dataframe = pd.read_csv(simulation_equal_folder_path + current_file_name, header=None)

                REECHD_FND, REECHD_HND = str(simulation_dataframe.iat[1, 0]).split(";")
                HEED_FND, HEED_HND = str(simulation_dataframe.iat[3, 0]).split(";")
                ERHEED_FND, ERHEED_HND = str(simulation_dataframe.iat[5, 0]).split(";")
                FMUC_FND, FMUC_HND = str(simulation_dataframe.iat[7, 0]).split(";")

                REECHD_AVG_LAYERS = str(simulation_dataframe.iat[9, 0]).split(";")[1]
                REECHD_AVG_CHxROUND = str(simulation_dataframe.iat[10, 0]).split(";")[1]

                HEED_AVG_LAYERS = str(simulation_dataframe.iat[12, 0]).split(";")[1]
                HEED_AVG_CHxROUND = str(simulation_dataframe.iat[13, 0]).split(";")[1]

                ERHEED_AVG_LAYERS = str(simulation_dataframe.iat[15, 0]).split(";")[1]
                ERHEED_AVG_CHxROUND = str(simulation_dataframe.iat[16, 0]).split(";")[1]

                FMUC_AVG_LAYERS = str(simulation_dataframe.iat[18, 0]).split(";")[1]
                FMUC_AVG_CHxROUND = str(simulation_dataframe.iat[19, 0]).split(";")[1]

                dictionary_equal.get("REECHD AVG LAYERS").append(REECHD_AVG_LAYERS)
                dictionary_equal.get("REECHD AVG CHxROUND").append(REECHD_AVG_CHxROUND)
                dictionary_equal.get("REECHD FND").append(REECHD_FND)
                dictionary_equal.get("REECHD HND").append(REECHD_HND)

                dictionary_equal.get("HEED AVG LAYERS").append(HEED_AVG_LAYERS)
                dictionary_equal.get("HEED AVG CHxROUND").append(HEED_AVG_CHxROUND)
                dictionary_equal.get("HEED FND").append(HEED_FND)
                dictionary_equal.get("HEED HND").append(HEED_HND)

                dictionary_equal.get("ERHEED AVG LAYERS").append(ERHEED_AVG_LAYERS)
                dictionary_equal.get("ERHEED AVG CHxROUND").append(ERHEED_AVG_CHxROUND)
                dictionary_equal.get("ERHEED FND").append(ERHEED_FND)
                dictionary_equal.get("ERHEED HND").append(ERHEED_HND)

                dictionary_equal.get("FMUC AVG LAYERS").append(FMUC_AVG_LAYERS)
                dictionary_equal.get("FMUC AVG CHxROUND").append(FMUC_AVG_CHxROUND)
                dictionary_equal.get("FMUC FND").append(FMUC_FND)
                dictionary_equal.get("FMUC HND").append(FMUC_HND)

        # --------------------
        # UNEQUAL PROTOCOLS
        # --------------------
        file_list_unequal = os.listdir(simulation_unequal_folder_path)  # dir is your directory path

        dictionary_unequal = {'HEIGHT': [], 'WIDTH': [], 'NODE': [], 'R0': [], '%AGGR': [], 'HET': [], 'HOM ENERGY': [],
                              'HOM RATE': [], 'CONTROL': [], 'DENSITY': [], 'UHEED AVG LAYERS': [],
                              'UHEED AVG CHxROUND': [], 'UHEED FND': [], 'UHEED HND': [], 'RUHEED AVG LAYERS': [],
                              'RUHEED AVG CHxROUND': [], 'RUHEED FND': [], 'RUHEED HND': []}

        for i in range(0, len(file_list_unequal)):
            current_file_name = file_list_unequal[i]

            width, height, control, hom_energy, hom_rate, perc_aggr, heterogeneity, transm_range_e = current_file_name.split(
                "_")

            """
            width:              W...        [1:] 
            height:             H...        [1:] 
            control             Control...  [7:]
            hom_energy:         Henergy...  [7:] 
            hom_rate:           Hrate...    [5:]
            perc_aggr:          Agg...      [3:]
            heterogeneity:      Hperc...    [5:]
            transm_range_e:     R0...       [2:]
            """

            width = width[1:]
            height = height[1:]
            control = control[7:]
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
                print(current_file_name, " - ", control, " - ", hom_energy, ", ", hom_rate, ", ", perc_aggr, 
                    ", ", heterogeneity, ", ",transm_range)
                """

                dictionary_unequal.get("HEIGHT").append(selected_height)
                dictionary_unequal.get("WIDTH").append(selected_width)
                dictionary_unequal.get("NODE").append(nodes_number)

                dictionary_unequal.get("R0").append(transm_range)
                dictionary_unequal.get("%AGGR").append(perc_aggr)
                dictionary_unequal.get("HET").append(heterogeneity)
                dictionary_unequal.get("HOM ENERGY").append(hom_energy)
                dictionary_unequal.get("HOM RATE").append(hom_rate)
                dictionary_unequal.get("CONTROL").append(control)

                density = nodes_number / (int(selected_height) * int(selected_width))
                dictionary_unequal.get("DENSITY").append(density)

                simulation_dataframe = pd.read_csv(simulation_unequal_folder_path + current_file_name, header=None)

                UHEED_FND, UHEED_HND = str(simulation_dataframe.iat[1, 0]).split(";")
                RUHEED_FND, RUHEED_HND = str(simulation_dataframe.iat[3, 0]).split(";")

                UHEED_AVG_LAYERS = str(simulation_dataframe.iat[5, 0]).split(";")[1]
                UHEED_AVG_CHxROUND = str(simulation_dataframe.iat[6, 0]).split(";")[1]

                RUHEED_AVG_LAYERS = str(simulation_dataframe.iat[8, 0]).split(";")[1]
                RUHEED_AVG_CHxROUND = str(simulation_dataframe.iat[9, 0]).split(";")[1]

                dictionary_unequal.get("UHEED AVG LAYERS").append(UHEED_AVG_LAYERS)
                dictionary_unequal.get("UHEED AVG CHxROUND").append(UHEED_AVG_CHxROUND)
                dictionary_unequal.get("UHEED FND").append(UHEED_FND)
                dictionary_unequal.get("UHEED HND").append(UHEED_HND)

                dictionary_unequal.get("RUHEED AVG LAYERS").append(RUHEED_AVG_LAYERS)
                dictionary_unequal.get("RUHEED AVG CHxROUND").append(RUHEED_AVG_CHxROUND)
                dictionary_unequal.get("RUHEED FND").append(RUHEED_FND)
                dictionary_unequal.get("RUHEED HND").append(RUHEED_HND)

        dataframe_equal = pd.DataFrame(data=dictionary_equal, columns=dictionary_equal.keys())
        dataframe_unequal = pd.DataFrame(data=dictionary_unequal, columns=dictionary_unequal.keys())

        # ----------------------------------------------------------
        # STORING DATA TO A SIMPLEST CSV
        # ----------------------------------------------------------

        dataframe_equal.to_csv(output_merged_equal_file_path, index=False)
        print(dataframe_equal)
        print("---------------------------------------------------------")
        dataframe_unequal.to_csv(output_merged_unequal_file_path, index=False)
        print(dataframe_unequal)


def merge_csv_files():
    all_files_equal = glob.glob(os.path.join(output_equal_folder_path, "*.csv"))
    all_files_unequal = glob.glob(os.path.join(output_unequal_folder_path, "*.csv"))

    df_from_each_file_equal = (pd.read_csv(f) for f in all_files_equal)
    df_from_each_file_unequal = (pd.read_csv(f) for f in all_files_unequal)

    concatenated_df_equal = pd.concat(df_from_each_file_equal, ignore_index=True)
    concatenated_df_unequal = pd.concat(df_from_each_file_unequal, ignore_index=True)

    concatenated_df_equal.to_csv(output_merged_equal_csv_pathname, index=False)
    concatenated_df_unequal.to_csv(output_merged_unequal_csv_pathname, index=False)


def load_data(type="equal", shuffle=False):
    if type == "equal":
        dataframe = pd.read_csv(output_merged_equal_csv_pathname, header=None)

        nt_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        avg_layers_cols = [9, 13, 17, 21]
        avg_chxrounds_cols = [10, 14, 18, 22]
        sim_cols = [11, 12, 15, 16, 19, 20, 23, 24]
    else:
        dataframe = pd.read_csv(output_merged_unequal_csv_pathname, header=None)
        nt_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        avg_layers_cols = [10, 14]
        avg_chxrounds_cols = [11, 15]
        sim_cols = [12, 13, 16, 17]

    headers = dataframe[:1]
    data = dataframe[1:]

    if shuffle:
        data = sh(data)

    nt = data.as_matrix(nt_cols)
    avg_layers = data.as_matrix(avg_layers_cols)
    avg_chxrounds = data.as_matrix(avg_chxrounds_cols)
    sim = data.as_matrix(sim_cols)

    headers_nt = np.squeeze(headers.as_matrix(nt_cols))
    headers_avg_layers = np.squeeze(headers.as_matrix(avg_layers_cols))
    headers_avg_chxrounds = np.squeeze(headers.as_matrix(avg_chxrounds_cols))
    headers_sim = np.squeeze(headers.as_matrix(sim_cols))

    return nt.astype(float), avg_layers.astype(float), avg_chxrounds.astype(float), \
           sim.astype(float), headers_nt, headers_avg_layers, \
           headers_avg_chxrounds, headers_sim


if __name__ == "__main__":
    for h in heights:
        for w in widths:
            create_csv_data(selected_height=h, selected_width=w)
    merge_csv_files()
